from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            import cmake
        except ImportError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        wd = os.getcwd()
        os.makedirs(self.build_temp, exist_ok=True)
        os.chdir(self.build_temp)
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir]
        self.spawn(['cmake', ext.sourcedir] + cmake_args)
        self.spawn(['cmake', '--build', '.', '--config', 'Release'])
        if os.name == 'nt':
            os.chdir(extdir)
            os.rename('Release/tfim_sampler.cp312-win_amd64.pyd', 'tfim_sampler.cp312-win_amd64.pyd')
            os.rmdir('Release')
        os.chdir(wd)

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

ext_modules = [CMakeExtension('tfim_sampler')]

setup(
    name='PyQrackIsing',
    version='1.3.3',
    author='Dan Strano',
    author_email='stranoj@gmail.com',
    description='Near-ideal closed-form solutions for transverse field Ising model (TFIM)',
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/vm6502q/PyQrackIsing",
    license="LGPL-3.0-or-later",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=["pybind11"],
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    packages=['PyQrackIsing'],
    zip_safe=False,
)
