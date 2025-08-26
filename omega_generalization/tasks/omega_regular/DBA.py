import functools
from collections import defaultdict

import jax
from jax import numpy as jnp
import jax.nn as jnn

from omega_generalization.tasks import task

import spot
import signal
import logging
from contextlib import contextmanager

@contextmanager
def timeout_handler(seconds):
    """Context manager to handle timeouts using signals."""
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Reset the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class DBA(task.GeneralizationTask):
    def __init__(self, num_states, alphabet_size, transition_matrix, initial_state, accepting_states, special_symbol, symbol_index):
        """
        Initialize a (deterministic) Büchi Automaton with JAX arrays for efficient computation.
        
        Args:
            num_states: Number of states in the automaton
            alphabet_size: Size of the alphabet (e.g., 2 for binary)
            transition_matrix: JAX array of shape (num_states, alphabet_size) where
                              transition_matrix[state, symbol] gives the next state
            initial_state: The starting state (integer)
            accepting_states: JAX array of booleans indicating which states are accepting
            special_symbol: The special symbol that triggers acceptance checking
        """
        super().__init__()
        self.num_states = num_states
        self.alphabet_size = alphabet_size
        self.symbol_index = symbol_index
        self.special_symbol = special_symbol
        
        self.transition_matrix = transition_matrix
        self.state_transition_matrix = jnp.zeros((num_states, num_states), dtype=jnp.bool)  # Transition matrix for SCCs
        for state in range(num_states):
            for symbol in range(alphabet_size - 1):
                next_state = int(self.transition_matrix[state, symbol])
                self.state_transition_matrix = self.state_transition_matrix.at[state, next_state].set(True)

        self.initial_state = initial_state
        self.accepting_states = accepting_states
        
        self.rejecting_sink_states = jnp.zeros(num_states, dtype=jnp.bool)  
        self.accepting_sink_states = jnp.zeros(num_states, dtype=jnp.bool)

        # Identify rejecting sink states (states with no outgoing transitions)
        for state in range(num_states):
            if jnp.all(self.transition_matrix[state, :-1] == state):
                if (1 - accepting_states[state]):
                    self.rejecting_sink_states = self.rejecting_sink_states.at[state].set(True)
                else:
                    self.accepting_sink_states = self.accepting_sink_states.at[state].set(True)

        # Compute SCCs and store transition matrices for accepting SCCs
        self.sccs_mask = jnp.zeros((num_states, num_states), dtype=jnp.bool) 
        self.scc_info = self._compute_scc_info()
        self.acc_sccs_mask = jnp.any(self.sccs_mask & self.accepting_states, axis=1)  
        self.rejecting_transition_matrix = self.state_transition_matrix
        for i in range(num_states):
            for j in range(num_states):
                if self.state_transition_matrix[i, j] and self.accepting_states[j]:
                    self.rejecting_transition_matrix = self.rejecting_transition_matrix.at[i, j].set(False)
        state_transitions = jnp.sum(self.state_transition_matrix)

        logging.info("Prepared Büchi Automaton with the following parameters:")
        logging.info(f"Number of states: {self.num_states}, Alphabet size: {self.alphabet_size}, Initial state: {self.initial_state}")
        logging.info(f"State Transitions: {state_transitions}, Accepting states: {self.accepting_states.sum()}")    

    def __call__(self):
        return self
    
    def seq_to_formula(self, sequence) -> str:
        """
        Convert a sequence of symbols to a Spot omega-word.
        
        Args:
            sequence: JAX array of symbols (integers)
            
        Returns:
            A Spot formula string representing the sequence
        """
        # Convert symbols to their string representation
        prefix, suffix = [], []
        past_special = False
        for symbol in sequence:
            if symbol != self.special_symbol:
                if past_special:
                    suffix.append(self.symbol_index[int(symbol)])
                else:
                    prefix.append(self.symbol_index[int(symbol)])
            else:
                past_special = True
        prefix_str = "; ".join(prefix).strip() if len(prefix) > 0 else None
        suffix_str = "; ".join(suffix).strip()
        
        return f"{prefix_str}; cycle{{{suffix_str}}}" if prefix_str else f"cycle{{{suffix_str}}}"
    
    def _compute_scc_info(self):
        """
        Compute strongly connected components and their transition matrices for accepting states.
        
        Returns:
            Dictionary containing SCC information
        """
        # First, find all SCCs using Tarjan's algorithm
        sccs = self._tarjan_scc()
        scc_ids = jnp.arange(len(sccs))

        for scc_idx in scc_ids:
            for state in sccs[scc_idx]:
                mask = jnp.isin(jnp.arange(self.num_states), jnp.array(sccs[scc_idx])) | self.sccs_mask[state]
                self.sccs_mask = self.sccs_mask.at[state].set(mask)  # Mark accepting states in the SCC
                
    def _tarjan_scc(self):
        """
        Tarjan's algorithm for finding strongly connected components.
        
        Returns:
            List of SCCs, where each SCC is a list of states
        """
        adj_list = defaultdict(set)
        for state in range(self.num_states):
            for symbol in range(self.alphabet_size - 1):
                next_state = int(self.transition_matrix[state, symbol])
                adj_list[state].add(next_state)
        
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []
        
        def strongconnect(v):
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True
            
            for w in adj_list[v]:
                if w not in index:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack[w]:
                    lowlinks[v] = min(lowlinks[v], index[w])
            
            if lowlinks[v] == index[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                sccs.append(component)
        
        for v in range(self.num_states):
            if v not in index:
                strongconnect(v)
        
        return sccs
    
    def transition_step(self, state, symbol):
        return self.transition_matrix[state, symbol]
    
    def state_mapping(self, sequence, mask=None):
        """
        Produce a mapping of initial states to final states when sequence is applied to the automaton.
        
        Args:
            sequence: JAX array of symbols (integers)"""
        if mask is None:
            mask = jnp.ones_like(sequence, dtype=jnp.bool_)

        mapping = jnp.zeros((self.num_states, self.num_states), dtype=jnp.bool_)
        mapping_traverses_accepting_state = jnp.zeros((self.num_states, self.num_states), dtype=jnp.bool_) 
        def scan_step(carry, input):
            state, path_has_acc_state = carry
            symbol, mask = input
            next_state = jnp.where(mask, self.transition_step(state, symbol), state)
            path_has_acc_state = jnp.logical_or(path_has_acc_state, self.accepting_states[next_state])
            return (next_state, path_has_acc_state), None

        initial_states = jnp.arange(self.num_states)
        initial_carry = (initial_states, self.accepting_states)  

        (final_states, traversed_acc_state), _ = jax.lax.scan(scan_step, initial_carry, (sequence, mask))

        mapping = mapping.at[initial_states, final_states].set(True)
        mapping_traverses_accepting_state = mapping_traverses_accepting_state.at[initial_states, final_states].set(traversed_acc_state)
        return mapping, mapping_traverses_accepting_state
    
    def run_sequence(self, sequence):
        """
        Run the Büchi automaton on a sequence.
        
        Args:
            sequence: JAX array of symbols (integers) representing the input sequence
            
        Returns:
            Tuple of (final_state, is_accepted, states_trace)
        """
        def scan_prefix(carry, symbol):
            state, special_state, special_idx = carry
            # Check if current state is accepting when we see the special symbol
            is_special = symbol == self.special_symbol
            special_idx = jnp.where(jnp.logical_and(jnp.logical_not(is_special), special_state == -1), special_idx + 1, special_idx)
            
            # Update special_accepting_state when we encounter special symbol in accepting state
            # Use -1 to indicate no accepting state has been found yet with special symbol
            special_state = jnp.where(
                is_special,
                state,  # Record this accepting state
                special_state  # Keep previous state
            )
            next_state = self.transition_step(state, symbol)
            
            return (next_state, special_state, special_idx), next_state
        # Use -1 to indicate no special accepting state found yet
        initial_carry = (self.initial_state, -1, 0)
        
        # Run the scan
        (_, special_state, special_idx), states_trace = jax.lax.scan(
            scan_prefix, initial_carry, sequence
        )

        suffix_mask = jnp.arange(sequence.shape[0]) > special_idx
        state_mapping, mapping_traverses_acc_state = self.state_mapping(sequence, suffix_mask)

        reachability = state_mapping.astype(jnp.float32)
        accepting_reachability = mapping_traverses_acc_state.astype(jnp.float32)

        def matrix_power_step(carry, _):
            R, A = carry
            R_new = jnp.clip(jnp.dot(R, R) + R, 0, 1)  # Boolean OR of paths
            # For accepting paths: A_new[i,j] = 1 if there's any path from i to j that traverses accepting states
            A_new = jnp.clip(jnp.dot(A, R) + jnp.dot(R, A) + A, 0, 1)
            return (R_new, A_new), None

        (reachability, accepting_reachability), _ = jax.lax.scan(
            matrix_power_step, (reachability, accepting_reachability), jnp.arange(self.num_states)
        )
        
        accepting_cycles = jnp.sum(accepting_reachability * jnp.eye(self.num_states), axis=1)  # Check if any cycles exist
        reachable_from_special = reachability[special_state, :] > 0
        accepting_cycles = reachable_from_special & (accepting_cycles > 0)
        is_accepted = jnp.any(accepting_cycles)
        return state_mapping, is_accepted, states_trace
    
    def run_batch(self, sequences, lengths=None):
        """
        Run the Büchi automaton on a batch of sequences.
        
        Args:
            sequences: JAX array of shape (batch_size, max_seq_length)
            lengths: JAX array of shape (batch_size,) with the actual length of each sequence,
                    or None if all sequences are the same length
            
        Returns:
            Tuple of (final_states, is_accepted)
        """
        # Define the scan function for a single sequence
        def process_sequence(args):
            if lengths is not None:
                sequence, length = args
                # Create a mask for valid positions
                mask = jnp.arange(sequence.shape[0]) < length
                # Apply the mask (replace invalid positions with a safe value, e.g., 0)
                sequence = jnp.where(mask, sequence, jnp.zeros_like(sequence))
            else:
                sequence = args
            
            _, is_accepted, _ = self.run_sequence(sequence)
            return is_accepted
        
        # Prepare input for vmap
        if lengths is not None:
            vmap_input = (sequences, lengths)
        else:
            vmap_input = sequences
        
        # Vectorize over the batch
        return jax.vmap(process_sequence)(vmap_input)

    def _generate_random_path(self, rng, input_seq, start_idx, end_idx, initial_state,
                             no_rejecting_sinks=True, no_accepting_sinks=False, amplify_rejecting_sinks=False):
        """Generate a random path by sampling symbols on the fly, with optional masked transitions."""
        
        sequence_length = len(input_seq)
        
        # Ensure indices are valid
        start_idx = jnp.clip(start_idx, 0, sequence_length)
        end_idx = jnp.clip(end_idx, start_idx, sequence_length)
        
        # Create position mask for which positions to modify
        position_mask = (jnp.arange(sequence_length) >= start_idx) & (jnp.arange(sequence_length) < end_idx)
        state_mask = self.state_transition_matrix
        state_mask = jnp.where(no_rejecting_sinks, jnp.logical_and(state_mask, ~self.rejecting_sink_states[None, :]), state_mask)
        # We make any non accepting state a sink state
        state_mask = jnp.where(no_accepting_sinks, jnp.logical_and(state_mask, ~self.accepting_sink_states[None, :]), state_mask)

        state_mask = jnp.where(amplify_rejecting_sinks, jnp.logical_and(state_mask, self.rejecting_transition_matrix), state_mask)
        def masked_step_fn(carry, inputs):
            state, rng_key = carry
            original_symbol, pos_idx = inputs
            
            # Check if we should sample a new symbol at this position
            should_sample = position_mask[pos_idx]
            
            def sample_valid_symbol(state, rng_key):
                # Get valid state transitions from the mask matrix
                state_rng, symbol_rng = jax.random.split(rng_key)
                valid_state_transitions = state_mask[state, :]  # Boolean mask of valid state transitions
                valid_state_transitions = jnp.where(jnp.sum(valid_state_transitions) == 0, jnp.ones_like(valid_state_transitions), valid_state_transitions)  # Avoid division by zero
                random_state = jax.random.choice(state_rng, jnp.arange(self.num_states), p=valid_state_transitions.astype(float) / jnp.sum(valid_state_transitions))
                valid_symbols_mask = self.transition_matrix[state, :-1] == random_state  # Mask for valid symbols that lead to the random state
                # if valid symbols empty then sample from all to ensure we can always sample.
                valid_symbols_mask = jnp.where(jnp.sum(valid_symbols_mask) == 0, jnp.ones_like(valid_symbols_mask), valid_symbols_mask)  # Avoid division by zero

                sampled_idx = jax.random.choice(symbol_rng, jnp.arange(self.alphabet_size - 1), p=valid_symbols_mask.astype(float) / jnp.sum(valid_symbols_mask))
                return sampled_idx
            
            # Split RNG for this step
            rng_key, subkey = jax.random.split(rng_key)
            
            # Choose symbol: sample if in range, otherwise use original
            symbol = jnp.where(should_sample, 
                            sample_valid_symbol(state, subkey),
                            original_symbol)
            
            # Transition to next state
            next_state = jnp.where(should_sample, self.transition_matrix[state, symbol], state)
            
            return (next_state, rng_key), (symbol, next_state)
        
        # Prepare inputs: (original_symbols, position_indices)
        position_indices = jnp.arange(sequence_length)
        inputs = (input_seq, position_indices)
        
        # Choose which step function to use based on whether mask is provided
        # if scc:
        (final_state, _), (modified_sequence, states_trace) = jax.lax.scan(
                masked_step_fn, 
                (initial_state, rng), inputs
            )
        
        # Include initial state in the trace
        full_states_trace = jnp.concatenate([jnp.array([initial_state]), states_trace])
        
        return modified_sequence, full_states_trace, final_state

    def generate_positives(self, rng: jax.random.PRNGKey, batch_size: int, length: int,  samples: int=1024) -> jnp.ndarray:
        symbol_length = length - 1
        finite_prefix_lengths = jax.random.randint(rng, (samples,), 0, symbol_length - 1) # 0 to length - 1 (suffix must be at least 1)
        prefix_rng, suffix_rng = jax.random.split(rng, num=2)
        raw_sequences = jnp.zeros((samples, length), dtype=jnp.int32) + self.special_symbol  # Initialize with special symbol

        raw_sequences, _, loop_states = jax.vmap(functools.partial(self._generate_random_path, initial_state=self.initial_state, no_rejecting_sinks=True, no_accepting_sinks=False), in_axes=(0, 0, None, 0))(jax.random.split(prefix_rng, num=samples), raw_sequences, 0, finite_prefix_lengths)
        raw_sequences, _, _ = jax.vmap(functools.partial(self._generate_random_path, no_rejecting_sinks=True, no_accepting_sinks=False), in_axes=(0, 0, 0, None, 0))(jax.random.split(suffix_rng, num=samples), raw_sequences, finite_prefix_lengths + 1, length, loop_states)
        is_accepted = self.run_batch(raw_sequences)

        sorted_order = jnp.argsort(is_accepted, stable=True)[::-1]
        raw_sequences = raw_sequences[sorted_order]
        
        return raw_sequences[:batch_size], is_accepted[sorted_order][:batch_size]  # Return only the first `batch_size` sequences

    def generate_negatives(self, rng: jax.random.PRNGKey, batch_size: int, length: int, samples: int=1024) -> jnp.ndarray:
        symbol_length = length - 1 # special symbol will sandwich the two parts of the U.P. w-word.
        finite_prefix_lengths = jax.random.randint(rng, (samples,), 0, symbol_length - 1) # 0 to length - 1 (suffix must be at least 1)
        prefix_rng, suffix_rng = jax.random.split(rng, num=2)
        raw_sequences = jnp.zeros((samples, length), dtype=jnp.int32) + self.special_symbol  # Initialize with special symbol

        raw_sequences, _, loop_states = jax.vmap(functools.partial(self._generate_random_path, initial_state=self.initial_state, no_rejecting_sinks=False, no_accepting_sinks=True), in_axes=(0, 0, None, 0))(jax.random.split(prefix_rng, num=samples), raw_sequences, 0, finite_prefix_lengths)

        negatives, _, _ = jax.vmap(functools.partial(self._generate_random_path, no_rejecting_sinks=False, no_accepting_sinks=True, amplify_rejecting_sinks=True), in_axes=(0, 0, 0, None, 0))(jax.random.split(suffix_rng, num=samples), raw_sequences, finite_prefix_lengths + 1, length, loop_states)
        is_accepted = self.run_batch(negatives)

        sorted_order = jnp.argsort(is_accepted, stable=True)  # Sort in ascending order, as we want the least accepted sequences
        negatives = negatives[sorted_order]

        return negatives[:batch_size], is_accepted[sorted_order][:batch_size]  # Return only the first `batch_size` sequences and their acceptance status

    @functools.partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def sample_batch(self, rng, batch_size, length, samples=512):
        """Returns a batch of strings and the expected class."""
        pos_rng, neg_rng, swap_rng = jax.random.split(rng, 3)
        half_size = batch_size // 2
        positives, pos_labels = self.generate_positives(pos_rng, half_size, length, samples)
        negatives, neg_labels = self.generate_negatives(neg_rng, batch_size - half_size, length, samples)
        inputs = jnp.concatenate([positives, negatives])
        labels = jnp.concatenate([pos_labels, neg_labels]).astype(int)
        # jax.debug.print("Negative ratio: {}", jnp.mean(labels))  # Debugging: print the labels for negative samples
        # Convert to one-hot representation
        one_hot_strings = jnn.one_hot(inputs, self.alphabet_size)
        ans = jnn.one_hot(labels, 2)
        
        indices = jax.random.permutation(swap_rng, batch_size)
        one_hot_strings = one_hot_strings[indices]
        ans = ans[indices]
        
        return {
            'input': one_hot_strings,
            'output': ans,
        }
    
    def evaluate_balance(self, rng, min_length, max_length, step=1, batches=8, batch_size=256, samples=512):
        """
        Evaluate the balance of positive and negative samples generated by the DBA.
        
        Args:
            min_length: Minimum length of sequences to generate
            max_length: Maximum length of sequences to generate
            batches: Number of batches to sample
            samples: Number of samples per batch
        """
        lengths = range(min_length, max_length + 1, step)
        results = {x:0 for x in lengths} | {"mean": 0}
        aggregate = 0
        rngs = jax.random.split(rng, len(lengths))
        for length in lengths:
            len_agg = 0
            jax.clear_caches()  # Clear JAX caches to avoid memory issues
            rngs_len = jax.random.split(rngs[length - min_length], batches)
            for i in range(batches):
                # Generate a batch of samples
                sample = self.sample_batch(rng=rngs_len[i], batch_size=batch_size, length=length, samples=samples)
                labels = sample['output'].argmax(axis=-1)
                subbatch_total = jnp.sum(labels)
                aggregate += subbatch_total
                len_agg += subbatch_total
            results[length] = len_agg / (batches * batch_size)
            logging.info(f"Length: {length}, Ratio: {results[length]}")
        results["mean"] = aggregate / (batches * batch_size * len(lengths))
        return results
    
    @property
    def input_size(self) -> int:
      """Returns the input size for the models."""
      return self.alphabet_size

    @property
    def output_size(self) -> int:
      """Returns the output size for the models."""
      return 2
    

def spot_dba(formula_str: str, timeout: int=600) -> DBA:
    """
    Create a Büchi automaton from a Spot formula string.
    
    Args:
        formula_str: A string representing the Spot formula (e.g., "F(a)")
        
    Returns:
        An instance of DBA representing the Büchi automaton
    """
    # Parse the formula using Spot
    logging.info("Translating LTL formula to Büchi automaton")
    # Set the timeout for Spot operations
    dba = None
    try:
        with timeout_handler(timeout):
            ba = spot.formula(formula_str).translate('buchi', 'complete')
            dba = spot.tgba_determinize(ba)
            dba = dba.postprocess('sbacc', 'det', 'complete')
            dba = spot.split_edges(dba)
            logging.info("Constructed DBA (spot) from LTL formula")
            if not dba.is_deterministic():
                raise ValueError("The resulting DBA is not deterministic. Nondeterminism cannot be handled.")
            # Extract parameters from the DBA
            num_states = dba.num_states()
            if num_states > 200:
                logging.warning(f"The resulting DBA has {num_states} states which can take prohibitively long to sample sequences from.")
            bdict = dba.get_dict()

            propositions = set()
            for t in dba.edges():
                proposition = spot.bdd_format_formula(bdict, t.cond)
                proposition = proposition.replace("!", "")
                if "&" in proposition:
                    # If the proposition is a conjunction, we can split it into individual propositions
                    for prop in proposition.split("&"):
                        propositions.add(prop.strip())
                # propositions.add(proposition)
            # print("props", propositions)
            alphabet_size = 2 ** len(propositions) + 1 # if just one symbol then we have that and its negation, two symbols we have 
            transition_matrix = jnp.zeros((num_states, alphabet_size), dtype=jnp.int32)
            transition_matrix = transition_matrix.at[:, -1].set(jnp.arange(num_states))  # Set the last column to self-loops

            # This is all combinations of the propositions and their negations --- do not include a proposition and its negation
            propositions = sorted(propositions)  # Ensure consistent ordering
            symbols = []
            for i in range(2 ** len(propositions)):
                parts = []
                for j, prop in enumerate(propositions):
                    if (i >> j) & 1:
                        parts.append(prop)
                    else:
                        parts.append("!" + prop)
                symbols.append(" & ".join(parts))
            
            symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols)} | {"$": len(symbols)}
            index_to_symbol = {idx: symbol for symbol, idx in symbol_to_index.items()}
            # print("symbol to index", symbol_to_index)
            accepting_states = jnp.zeros(num_states, dtype=bool)
            # Fill the transition matrix
            for t in dba.edges():
                symbol = spot.bdd_format_formula(bdict, t.cond)
                if t.acc.has(0):
                    # If the transition is accepting, mark the state as accepting
                    accepting_states = accepting_states.at[t.src].set(True)
                symbol_index = symbol_to_index[symbol]
                transition_matrix = transition_matrix.at[t.src, symbol_index].set(t.dst)
            initial_state = dba.get_init_state_number()
            special_symbol = len(symbols)  # Assuming 'a' is represented by 1
    except TimeoutError:
        logging.error(f"Spot translation timed out after {timeout} seconds for formula: {formula_str}")
        raise
    logging.info("Constructed DBA (object) from LTL formula")
    return DBA(num_states, alphabet_size, transition_matrix, initial_state, accepting_states, special_symbol, symbol_index=index_to_symbol)

if __name__ == "__main__":
    # Example usage
    formula_str = "(((((!godown)&(!goup))&(!ws)) & (G((!godown)|(!goup))) & (G(F(!ws))) & (G(((!ws)&(ss))->(X(!ws)))) & (G((ws)->((((X(!ws))|(X(X(!ws))))|(X(X(X(!ws)))))|(X(X(X(X(!ws))))))))) -> ((G((goup)->(F(!ss)))) & (G((godown)->(F(ss)))) & (G(!((ws)&(X(ss)))))))"
    # Compute our DBA object from formula
    dba = spot_dba(formula_str)

    key = jax.random.PRNGKey(0)
    ba = spot.formula(formula_str).translate('buchi', 'complete')
    aut = spot.tgba_determinize(ba)
    aut = aut.postprocess('sbacc', 'det', 'complete')
    
    for l in range(20, 501, 40):
        print(f"Testing sequences of length {l}...")
        samples = dba.sample_batch(key, batch_size=256, length=l, samples=1024)
        sequences = samples["input"].argmax(axis=-1) 
        targets = samples["output"].argmax(axis=-1)
        ground_truth = []
        for sequence in sequences:
            # Convert sequence to formula
            formula = dba.seq_to_formula(sequence)
            # Treat the omega-word as an automaton to intersect the two
            w = spot.parse_word(formula).as_automaton()
            ground_truth.append(w.intersects(aut))

        print(ground_truth)
        for i, (seq, gt, test) in enumerate(zip(sequences, ground_truth, targets)):
            print(f"Sequence {i}: {seq}, Ground Truth: {gt}, Tested: {test}")
            if gt != test:
                print("Mismatch found!")
                print("Sequence:", dba.seq_to_formula(seq))
                print("Ground Truth:", gt)
                print("Tested:", test)
                exit()

    print("All sequences tested correctly.")


