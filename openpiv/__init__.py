try:
    from .src import process, lib
except ImportError:
    from . import process
