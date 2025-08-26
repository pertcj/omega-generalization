def generate_lift_ltl_spec(num_floors):
    """
    Generate LTL specifications for safe lift behavior.
    
    Args:
        num_floors (int): Number of floors in the building (must be >= 2)
    
    Returns:
        list: List of LTL formulas as strings
    """
    if num_floors < 2:
        raise ValueError("Number of floors must be at least 2")
    
    specs = []
    spec_counter = 1
    
    # 1. The lift is only at one floor at a time (mutual exclusion)
    for i in range(num_floors):
        for j in range(i + 1, num_floors):
            specs.append(f"G(f{i} -> !f{j})")
            spec_counter += 1
    
    # 2. Initial values
    specs.append(f"!u")
    spec_counter += 1
    specs.append(f"f0")  # Start at floor 0
    spec_counter += 1
    
    # Initial button states (all unpressed)
    for i in range(num_floors):
        specs.append(f"!b{i}")
        spec_counter += 1
    
    specs.append(f"!up")  # Initially not moving up
    spec_counter += 1
    
    # 3. The lift and the users take turns
    specs.append(f"G(u -> !Xu)")
    spec_counter += 1
    specs.append(f"G((!Xu) -> u)")
    spec_counter += 1
    
    # 4. When it is the users' turn, then the floor remains unchanged
    for i in range(num_floors):
        specs.append(f"G(u -> (f{i} -> Xf{i}))")
        spec_counter += 1
        specs.append(f"G(u -> ((Xf{i}) -> f{i}))")
        spec_counter += 1
    
    # 5. The lift can move at most to one neighboring floor in one step
    for i in range(num_floors):
        neighbors = []
        if i > 0:  # Can go down
            neighbors.append(f"f{i-1}")
        neighbors.append(f"f{i}")  # Can stay
        if i < num_floors - 1:  # Can go up
            neighbors.append(f"f{i+1}")
        
        neighbor_formula = " | ".join(neighbors)
        specs.append(f"G(f{i} -> X({neighbor_formula}))")
        spec_counter += 1
    
    # 6. When it is the lift's turn, then the buttons remain unchanged
    for i in range(num_floors):
        specs.append(f"G((!u) -> (b{i} -> Xb{i}))")
        spec_counter += 1
        specs.append(f"G((!u) -> ((Xb{i}) -> b{i}))")
        spec_counter += 1
    
    # 7. The buttons remain pressed while the corresponding floor has not been reached
    for i in range(num_floors):
        specs.append(f"G((b{i} & !f{i}) -> Xb{i})")
        spec_counter += 1
    
    # 8. up is true if the lift moves up, false if it moves down, 
    #    and unchanged if it does not move
    for i in range(num_floors):
        # If staying at the same floor, up direction unchanged
        specs.append(f"G((f{i} & (Xf{i})) -> (up -> Xup))")
        spec_counter += 1
        specs.append(f"G((f{i} & (Xf{i})) -> ((Xup) -> up))")
        spec_counter += 1
        
        # If moving up from floor i
        if i < num_floors - 1:
            # Apply fix - Towards a notion of unsatisfiable and unrealizable cores for LTL
            specs.append(f"G((f{i} & (Xf{i+1})) -> X(up))")
            spec_counter += 1
        
        # If moving down from floor i
        if i > 0:
            # Apply fix - Towards a notion of unsatisfiable and unrealizable cores for LTL
            specs.append(f"G((f{i} & (Xf{i-1})) -> X(!up))")
            spec_counter += 1
    
    # 9. sb is true iff any button except ground floor (b0) is pressed
    if num_floors > 1:
        # sb is true if any upper floor button is pressed
        upper_buttons = " | ".join([f"b{i}" for i in range(1, num_floors)])
        specs.append(f"G(sb -> ({upper_buttons}))")
        spec_counter += 1
        
        # If any upper floor button is pressed, sb is true
        for i in range(1, num_floors):
            specs.append(f"G(b{i} -> sb)")
            spec_counter += 1
    
    # 10. When not used, the lift parks at floor 0
    for i in range(num_floors):
        specs.append(f"G((f{i} & !sb) -> (f{i} U (sb R ((Ff0) & !up))))")
        spec_counter += 1
    
    # 11. Each request will eventually be served
    for i in range(num_floors):
        specs.append(f"G(b{i} -> Ff{i})")
        spec_counter += 1
    
    specs = [f"({spec})" for spec in specs]  # Enclose each spec in parentheses
    return " & ".join(specs)