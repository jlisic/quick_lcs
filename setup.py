from setuptools import setup, Extension
import numpy as np

# Define the extension module
module = Extension(
    'quick_lcs.string_length_sum',
    sources=['quick_lcs/string_length_sum.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-fopenmp'],  # Add OpenMP flags for GCC
    extra_link_args=['-fopenmp']       # Add OpenMP flags for linking
)

# Setup function to build the package
setup(
    name='quick_lcs',
    version='0.1',
    description='A package for summing lengths of strings in NumPy arrays.',
    author='Jonathan Lisic',
    packages=['quick_lcs'],
    ext_modules=[module],
    install_requires=['numpy'],
)

