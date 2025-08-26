# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Training loop for base generalization experiments."""

import os
import dataclasses
import functools
import random
from typing import Tuple, List, Callable, Mapping, Optional, Any
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict
from tensorboardX import SummaryWriter

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import logging

from jax.lib import xla_bridge

from omega_generalization.tasks import task as task_lib
from omega_generalization.training import curriculum as curriculum_lib
from omega_generalization.training import range_evaluation

import jax.numpy as jnp
from typing import Any, Dict


_LossMetrics = Optional[Mapping[str, jnp.ndarray]]
_LossFn = Callable[[chex.Array, chex.Array], Tuple[float, _LossMetrics]]
_AccuracyFn = Callable[[chex.Array, chex.Array], float]
_ModelApplyFn = Callable[..., chex.Array]
_MAX_RNGS_RESERVE = 1e10

optimizer_from_string = {"adam": optax.adam, "amsgrad": optax.amsgrad}

@dataclasses.dataclass
class ClassicTrainingParams:
  """Parameters needed to train classical architectures."""
  seed: int 
  model_init_seed: int 
  training_steps: int
  log_frequency: int

  task: task_lib.GeneralizationTask
  length_curriculum: curriculum_lib.Curriculum
  batch_size: int

  model: hk.Transformed
  eval_model: hk.Transformed
  loss_fn: Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, _LossMetrics]]
  learning_rate: float
  task_str: str
  init_learning_rate: float = 1e-8
  warmup_frac: float = 0.2

  max_grad_norm: float = 1.0
  optimizer: str = "amsgrad"

  compute_full_range_test: bool = False
  range_test_total_batch_size: int = 512
  range_test_sub_batch_size: int = 64
  max_range_test_length: int = 100
  l2_lambda: float = 1e-3
  weight_decay: float = 0.0
  architecture_params: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  accuracy_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                 jnp.ndarray]] = None
  
  use_tensorboard: bool = False # log to tensorboard

def compute_l2_loss(params, module_name=None):
    l2_sum = 0.0
    if module_name is not None:
        for path, param in params.items():
            # Skip if "layer_norm" is in the parameter path
            if "layer_norm" in path.lower():
                continue
            if module_name in path and isinstance(param, jnp.ndarray):
                l2_sum += jnp.sum(jnp.square(param))
    else:
        for path, param in params.items():
            # Skip if "layer_norm" is in the parameter path
            if "layer_norm" in path.lower():
                continue
            if isinstance(param, jnp.ndarray):
                l2_sum += jnp.sum(jnp.square(param))
    return l2_sum

def _apply_loss_and_metrics_fn(
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    l2_lambda: float = 0.0,
) -> Tuple[float, Tuple[_LossMetrics, float]]:
  """Computes the model output and applies the loss function.

  Depending on whether a model is autoregressive or not, it will have a
  different number of input parameters (i.e., autoregressive models also require
  the targets as an input).

  Args:
    params: The model parameters.
    rng_key: The prng key to use for random number generation.
    batch: The data (consists of both inputs and outputs).
    model_apply_fn: The model function that converts inputs into outputs.
    loss_fn: A function that computes the loss for a batch of logits and labels.
    accuracy_fn: A function that computes the accuracy for a batch of logits and
      labels.

  Returns:
    The loss of the model for the batch of data, extra loss metrics and the
    accuracy, if accuracy_fn is not None.
  """
  outputs = model_apply_fn(params, rng_key, batch["input"])

  loss, loss_metrics = loss_fn(outputs, batch["output"])

  l2_loss = 0.5 * l2_lambda * compute_l2_loss(params)
  total_loss = loss + l2_loss
  
  if accuracy_fn is not None:
    accuracy = accuracy_fn(outputs, batch["output"])
  else:
    accuracy = None
  return total_loss, (loss_metrics, accuracy, outputs)

@functools.partial(
    jax.jit,
    static_argnames=(
        "model_apply_fn",
        "loss_fn",
        "accuracy_fn",
        "optimizer",
    ),
)
def _update_parameters(
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    l2_lambda: float = 0.0,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
    """Applies a single SGD update step to the model parameters.

    Args:
        params: The model parameters.
        rng_key: The prng key to use for random number generation.
        batch: The data (consists of both inputs and outputs).
        model_apply_fn: The model function that converts inputs into outputs.
        loss_fn: A function that computes the loss for a batch of logits and labels.
        accuracy_fn: A function that computes the accuracy for a batch of logits and labels.
        optimizer: The optimizer that computes the updates from the gradients of the
            `loss_fn` with respect to the `params` and the previous `opt_state`.
        opt_state: The optimizer state, e.g., momentum for each variable when using Adam.

    Returns:
        The updated parameters, the new optimizer state, and the loss, loss metrics,
        accuracy, and logits.
    """
    # Compute loss and gradients
    (loss, (metrics, accuracy, logits)), grads = jax.value_and_grad(
        _apply_loss_and_metrics_fn,
        has_aux=True)(params, rng_key, batch, model_apply_fn, loss_fn,
                     accuracy_fn, l2_lambda)
    
    def centralize_gradients(grads):
        def _centralize(g):
            if len(g.shape) > 1:  # Only apply to weight matrices, not biases
                mean = jnp.mean(g, axis=tuple(range(1, len(g.shape))), keepdims=True)
                return g - mean
            return g
        return jax.tree.map(_centralize, grads)

    grads = centralize_gradients(grads)

    # Compute optimizer update
    updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)

    # Compute optimization metrics
    param_norm = optax.global_norm(params)
    grad_norm = optax.global_norm(grads)
    update_norm = optax.global_norm(updates)
    
    # Compute gradient statistics
    grad_squares = jax.tree.map(lambda x: jnp.square(x), grads)
    grad_vars = jax.tree.map(lambda x: jnp.var(x), grad_squares)
    avg_grad_var = jnp.mean(jnp.array([jnp.mean(v) for v in jax.tree_util.tree_leaves(grad_vars)]))
    
    # Compute gradient norm ratio
    grad_layer_norms = jnp.array([optax.global_norm(g) for g in jax.tree_util.tree_leaves(grads)])
    grad_norm_ratio = jnp.max(grad_layer_norms) / (jnp.min(grad_layer_norms) + 1e-8)

    # Update metrics dictionary with optimizer metrics
    metrics.update({
        "param_norm": param_norm,
        "grad_norm": grad_norm,
        "update_norm": update_norm,
        "grad_norm_ratio": grad_norm_ratio,
        "avg_grad_var": avg_grad_var
    })

    return new_params, new_opt_state, (loss, metrics, accuracy, logits, grads)

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (bool, str, float, int)):
            items.append((new_key, v))
    return dict(items)

def to_dict(obj: Any) -> Dict:
    if not is_dataclass(obj):
        return {}
    
    result = {}
    for field in obj.__dataclass_fields__:
        value = getattr(obj, field)
        if isinstance(value, (bool, str, float, int)):
            result[field] = value
        elif is_dataclass(value):
            nested_dict = to_dict(value)
            result[field] = nested_dict
        elif isinstance(value, dict):
          result = {**result, **flatten_dict(value)}
    return result

def log_dataclass_hparams(writer: SummaryWriter, params: Any, prefix: str = '') -> None:
    """Logs all parameters from a dataclass (including nested dataclasses) to TensorBoard."""
    if not is_dataclass(params):
        raise ValueError("Input must be a dataclass instance")
    
    params_dict = to_dict(params)
    flat_params = flatten_dict(params_dict)
    if prefix:
        flat_params = {f"{prefix}.{k}": v for k, v in flat_params.items()}
    
    hparams = {k: str(v) if not isinstance(v, (bool, str, float, int)) else v 
               for k, v in flat_params.items()}
    
    writer.add_hparams(hparams, {})

class TrainingWorker:
  """Training worker."""

  def __init__(self,
               training_params: ClassicTrainingParams,
               use_tqdm: bool = False):
    """Initializes the worker.

    Args:
      training_params: The training parameters.
      use_tqdm: Whether to add a progress bar to stdout.
    """
    self._training_params = training_params
    self._use_tqdm = use_tqdm
    self._log_tensorboard = training_params.use_tensorboard
    exp_name = training_params.task.__class__.__name__
    if exp_name == "DBA":
       exp_name = training_params.task_str
    if self._log_tensorboard:
        from datetime import datetime
        today = datetime.now()
        today = today.strftime("%d%m%y_%H%M_%f")
        exp_name = exp_name + "/" + today
        extra = 1
        file_prefix = "tensorboard"
        while os.path.exists(f"{file_prefix}/{exp_name}"):
            # Check if the path already has a number suffix
            parts = exp_name.split("-")
            if len(parts) > 1 and parts[-1].isdigit():
                # If it does, increment that number
                exp_name = "_".join(parts[:-1]) + f"-{extra}"
            else:
                # If it doesn't, add the number suffix
                exp_name = exp_name + f"-{extra}"
            extra += 1
        self.name = exp_name
        print("Beginning run",self.name)
        self._writer = SummaryWriter(log_dir=f"{file_prefix}/{self.name}")
        log_dataclass_hparams(self._writer, training_params,)

  def run(
      self
  ) -> Tuple[List[Mapping[str, Any]], Optional[List[Mapping[str, Any]]],
             chex.ArrayTree]:
    """Trains the model with the provided config.

    Returns:
      Results (various training and validation metrics), module parameters
      and router parameters.
    """
    training_params = self._training_params
    rngs_reserve = min(_MAX_RNGS_RESERVE, 3 * training_params.training_steps + 2)

    random.seed(training_params.seed)
    np.random.seed(training_params.seed)
    rng_seq = hk.PRNGSequence(training_params.seed)
    rng_seq.reserve(rngs_reserve)

    results = []
    model = training_params.model
    eval_model = training_params.eval_model
    task = training_params.task
    length_curriculum = training_params.length_curriculum

    if training_params.warmup_frac == 0.0:
        optimizer = optimizer_from_string[training_params.optimizer](learning_rate=training_params.learning_rate)
    else:
        warmup_scheduler = optax.warmup_constant_schedule(init_value=training_params.init_learning_rate,
                                                       peak_value=training_params.learning_rate, 
                                                       warmup_steps=int(training_params.training_steps*training_params.warmup_frac))
        optimizer = optimizer_from_string[training_params.optimizer](learning_rate=warmup_scheduler)

    optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(training_params.weight_decay), 
            optimizer,
        )

    dummy_batch = task.sample_batch(next(rng_seq), length=training_params.max_range_test_length+1, batch_size=4)
    eval_batch = task.sample_batch(next(rng_seq), length=training_params.max_range_test_length, batch_size=1024, samples=4096)
    
    model_init_rng_key = jax.random.PRNGKey(training_params.model_init_seed)
    
    params = model.init(model_init_rng_key, dummy_batch["input"])

    opt_state = optimizer.init(params)
    self._params, self._step = params, 0

    model_apply_fn = jax.jit(model.apply)
    steps = range(training_params.training_steps + 1)
    if self._use_tqdm:
      steps = tqdm.tqdm(steps, dynamic_ncols=True, ascii=True)
    try:
      for step in steps:
        length = length_curriculum.sample_sequence_length(step)
        train_batch = task.sample_batch(next(rng_seq), length=length, batch_size=training_params.batch_size)
        
        params, opt_state, (train_loss, train_metrics, train_accuracy, predictions, grads) = _update_parameters(
                params=params,
                rng_key=next(rng_seq),
                batch=train_batch,
                model_apply_fn=model_apply_fn,
                loss_fn=training_params.loss_fn,
                accuracy_fn=training_params.accuracy_fn,
                optimizer=optimizer,
                opt_state=opt_state,
                l2_lambda=training_params.l2_lambda,
        )
        
        self._params, self._step = params, step

        logging.info('progress: {:.2f}, loss: {}, acc: {}'.format(step/training_params.training_steps, float(train_loss), float(train_accuracy)))
        if self._log_tensorboard:
          self._writer.add_scalar("training/loss", float(train_loss), step)
          self._writer.add_scalar("training/accuracy", float(train_accuracy), step)
          for key, value in train_metrics.items():
            self._writer.add_scalar(f"training_metrics/{key}", float(value), step)
        log_freq = training_params.log_frequency

        if (log_freq > 0) and (step % log_freq == 0):
            log_data = {
                "step": step,
                "train_loss": float(train_loss),
            }
            if training_params.accuracy_fn is not None:
                log_data["train_accuracy"] = float(train_accuracy)
            for key, value in train_metrics.items():
                log_data[".".join(["train_metrics", key])] = np.array(value)
            eval_loss, (_, eval_accuracy, predictions) = _apply_loss_and_metrics_fn(
                params=params,
                rng_key=next(rng_seq),
                batch=eval_batch,
                model_apply_fn=jax.jit(eval_model.apply),
                loss_fn=training_params.loss_fn,
                accuracy_fn=training_params.accuracy_fn,
                l2_lambda=training_params.l2_lambda,
            )
            if self._log_tensorboard:
                self._writer.add_scalar("validation/loss", float(eval_loss), step)
                self._writer.add_scalar("validation/accuracy", float(eval_accuracy), step)
            results.append(log_data)
        if not rng_seq._subkeys:  # pylint: disable=protected-access
          rng_seq.reserve(rngs_reserve)
          
    except KeyboardInterrupt:
      print("Training interrupted proceeding to evaluation.")

    eval_results = None
    if training_params.compute_full_range_test:
      eval_params = range_evaluation.EvaluationParams(
          model=eval_model,
          params=params,
          accuracy_fn=training_params.accuracy_fn,
          sample_batch=task.sample_batch,
          max_test_length=training_params.max_range_test_length,
          total_batch_size=training_params.range_test_total_batch_size,
          sub_batch_size=training_params.range_test_sub_batch_size,
      )
      eval_results = range_evaluation.range_evaluation(
          eval_params, use_tqdm=False, writer=self._writer if self._log_tensorboard else None)

    return results, eval_results, params
