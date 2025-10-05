from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import numpy


class BuildExt(build_ext):
    """Custom build_ext to handle compiler/platform differences."""

    def build_extensions(self):
        compiler = self.compiler.compiler_type
        for ext in self.extensions:
            if compiler == 'msvc':
                # Windows (MSVC)
                ext.extra_compile_args = ['/O2']
                ext.extra_link_args = []
            else:
                # GCC or Clang
                ext.extra_compile_args = ['-O3', '-std=gnu11']
                ext.extra_link_args = []

                # Try to enable OpenMP if available
                if self.has_openmp():
                    if sys.platform == 'darwin':
                        # macOS Clang needs special handling
                        ext.extra_compile_args += ['-Xpreprocessor', '-fopenmp']
                        ext.extra_link_args += ['-lomp']
                    else:
                        # Linux / Clang / GCC
                        ext.extra_compile_args += ['-fopenmp']
                        ext.extra_link_args += ['-fopenmp']

        build_ext.build_extensions(self)

    def has_openmp(self):
        """Detect if OpenMP is supported by the compiler."""
        import tempfile, textwrap, subprocess

        test_code = textwrap.dedent(
            r"""
            #include <omp.h>
            int main(void) {
                int n = omp_get_max_threads();
                return 0;
            }
            """
        )
        with tempfile.NamedTemporaryFile('w', suffix='.c', delete=False) as f:
            f.write(test_code)
            test_path = f.name

        try:
            cmd = self.compiler.compiler_so + [test_path, '-fopenmp', '-o', os.devnull]
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            return True
        except Exception:
            return False
        finally:
            try:
                os.remove(test_path)
            except OSError:
                pass


quick_lcs_module = Extension(
    'quick_lcs.string_length_sum',
    sources=['quick_lcs/string_length_sum.c'],
    include_dirs=[numpy.get_include()],
    language='c',
)

setup(
    name='quick_lcs',
    version='0.1',
    description='A package for string length sums using OpenMP',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jonathan Lisic',
    author_email='jlisic@gmail.com',
    url='https://github.com/jlisic/quick_lcs',
    packages=['quick_lcs'],
    install_requires=['numpy'],
    ext_modules=[quick_lcs_module],
    cmdclass={'build_ext': BuildExt},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.8',
)
