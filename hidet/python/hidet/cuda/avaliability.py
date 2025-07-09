import functools


@functools.cache
def available() -> bool:
    """
    Returns True if CUDA is available, False otherwise.

    Use ctypes to check if libcuda.so is available instead of calling cudart directly.

    Returns
    -------
    ret: bool
        Whether CUDA is available.
    """
    import ctypes.util

    if ctypes.util.find_library('cuda'):
        return True
    return False
