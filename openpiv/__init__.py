try:
    from .c_src import lib, process
except ImportError:
    from .py_src import lib, process
