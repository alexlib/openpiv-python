import logging
import numpy

try:
    from setuptools import setup
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext
    # from Cython.Distutils import build_ext
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    from distutils.command.build_ext import build_ext

# from setuptools.extension import Extension
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

logging.basicConfig()
log = logging.getLogger(__file__)

ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError)

class BuildFailed(Exception):
    pass

def construct_build_ext(build_ext):
    class WrappedBuildExt(build_ext):
        # This class allows C extension building to fail.
        def run(self):
            try:
                build_ext.run(self)
            except DistutilsPlatformError as x:
                raise BuildFailed(x)

        def build_extension(self, ext):
            try:
                build_ext.build_extension(self, ext)
            except ext_errors as x:
                raise BuildFailed(x)
    return WrappedBuildExt

setup_args = {'name': 'openpiv', 'license': 'GPL', 'author': 'OpenPIV',
    'packages': ['openpiv', 'openpiv.py_src', 'openpiv.c_src'],
    'cmdclass': {}
    }
    
USE_CYTHON = False  # command line option, try-import, ...

ext = '.pyx' if USE_CYTHON else '.c'

ext_modules = [Extension(
                        name         = "openpiv.process",
                        sources      = ["openpiv/c_src/process"+ext],
                        include_dirs = [numpy.get_include()],
                        ),
                Extension(    name         = "openpiv.lib",
                        sources      = ["openpiv/c_src/lib"+ext],
                        include_dirs = [numpy.get_include()],
                    )
                ]

cmd_classes = setup_args.setdefault('cmdclass', {})

try:
    # try building with c code :
    setup_args['cmdclass']['build_ext'] = construct_build_ext(build_ext)
    setup(ext_modules=ext_modules, **setup_args)
except BuildFailed as ex:
    log.warn(ex)
    log.warn("The C extension could not be compiled")

    ## Retry to install the openpiv without C extensions :
    # Remove any previously defined build_ext command class.
    if 'build_ext' in setup_args['cmdclass']:
        del setup_args['cmdclass']['build_ext']
    if 'build_ext' in cmd_classes:
        del cmd_classes['build_ext']

    # If this new 'setup' call don't fail, the openpiv 
    # will be successfully installed, without the C extension :
    setup(**setup_args)
    log.info("Plain-Python installation succeeded.")