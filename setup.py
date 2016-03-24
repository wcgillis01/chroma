from setuptools import setup, find_packages, Extension
import subprocess
import os

libraries = []
extra_objects = []

def check_output(*popenargs, **kwargs):
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd, output=output)
    return output

if 'VIRTUAL_ENV' in os.environ:
    include_dirs.append(os.path.join(os.environ['VIRTUAL_ENV'], 'include'))
try:
    import numpy.distutils
    include_dirs += numpy.distutils.misc_util.get_numpy_include_dirs()
except:
    pass # if numpy doesn't exist yet

setup(
    name = 'Chroma',
    version = '0.6',
    packages = find_packages(),
    include_package_data=True,
    setup_requires = [],
    install_requires = ['uncertainties','pyzmq-static','spnav', 'pycuda', 
                        'numpy>=1.6', 'pygame', 'nose', 'sphinx'],
    #test_suite = 'nose.collector',
    
)
