from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("cython_test", ["cython_test.pyx"], language = 'c++'), Extension("cython_random_order", ["cython_random_order.pyx"], language = 'c++')]

setup(
  name = 'experts',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

