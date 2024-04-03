def binary_to_decimal(binary):
    """This function converts a binary number to a decimal number.

    Args:
        binary (_type_): _description_

    Returns:
        int: The decimal number.
    """    
    return sum(
        [
            int(binary[-i-1])*2**i for i in range(len(binary))
        ]
    )