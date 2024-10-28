from setuptools import setup, Extension
import numpy

# Define the extension module
quick_lcs_module = Extension(
    'quick_lcs.string_length_sum',
    sources=['quick_lcs/string_length_sum.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp', '-std=gnu11'],
    extra_link_args=['-fopenmp'],
    language='c',
)

# Setup function
setup(
    name='quick_lcs',
    version='0.1',
    description='A package for string length sums using OpenMP',
    long_description=open('README.md').read(),  # Use README.md for long description
    long_description_content_type='text/markdown',  # Specify the format of the long description
    author='Jonathan Lisic',
    author_email='jlisic@gmail.com',
    url='https://github.com/jlisic/quick_lcs',  # Replace with your repository URL
    packages=['quick_lcs'],
    install_requires=['numpy'],  # Ensure NumPy is included as a dependency
    ext_modules=[quick_lcs_module],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.8',  # Specify supported Python versions
)
