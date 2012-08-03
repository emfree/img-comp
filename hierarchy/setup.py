from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("experts", ["experts.pyx"]),
        Extension("pqueue", ["pqueue.pyx"]), Extension("aux", ["aux.pyx"]), Extension("hierarchy2", ["hierarchy2.pyx"]), Extension("cython_random_order", ["cython_random_order.pyx"])]

setup(
  name = 'experts',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

