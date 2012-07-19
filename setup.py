from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("helper_routines", ["helper_routines.pyx"])]

setup(
  name = 'helper_routines',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

