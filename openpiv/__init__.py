from __future__ import print_function
from __future__ import absolute_import

try:
    import openpiv.lib
    import openpiv.process
except ImportError:
    from .py_src import lib
    from .py_src import process
