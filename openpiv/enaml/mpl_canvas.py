#------------------------------------------------------------------------------
# Copyright (c) 2013, Nucleic Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#------------------------------------------------------------------------------
import enaml
from enaml.qt.qt_application import QtApplication

import sys
sys.path.append('/Users/alex/Documents/OpenPIV/alexlib/openpiv-python')

# import openpiv.tools
# import openpiv.process
# import openpiv.scaling
import numpy as np

from atom.api import Atom, Unicode, Range, Bool, Value, Int, Tuple, observe



def main():


    with enaml.imports():
        from mpl_canvas import Main

    app = QtApplication()

    view = Main()
    view.show()

    # Start the application event loop
    app.start()


if __name__ == "__main__":
    main()