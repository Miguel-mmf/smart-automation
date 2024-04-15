

def calc_error(
    set_point: float, 
    current_value: float
) -> float:
    
    return set_point - current_value


def calc_delta_error(
    error: float, 
    previous_error: float
) -> float:
    
    return error - previous_error


def calc_new_frequency(
    frequency: float, 
    delta_frequency: float
) -> float:
    
    return frequency + delta_frequency