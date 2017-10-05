try:
    from .c_src import *
except ImportError:
    from .src import *
