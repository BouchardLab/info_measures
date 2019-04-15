"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.ccompiler import get_default_compiler
from Cython.Build import cythonize
# To use a consistent encoding
from codecs import open
from os import path
import numpy as np

import glob

here = path.abspath(path.dirname(__file__))


# cKDTree
ckdtree_src = glob.glob('spatial/ckdtree/src/*.cxx')
ckdtree_dep = glob.glob('spatial/*.pyx')

class custom_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.compiler is None:
            compiler = get_default_compiler()
        else:
            compiler = self.compiler


inc_dirs = [np.get_include(), path.join('spatial', 'ckdtree', 'src'),
            'spatial']
ext_modules = cythonize(
    [Extension('info_measures.spatial.ckdtree',
               sources=ckdtree_dep + ckdtree_src,
               depends=ckdtree_dep + ckdtree_src,
               include_dirs=inc_dirs)])

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

pkgs = find_packages()
print("found the following packages:", pkgs)
setup(
    name='info_measures',
    description='Information measures.',
    long_description=long_description,
    author='Jesse Livezey',
    author_email='jesse.livezey@gmail.com',
    packages=pkgs,
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext}
    )
