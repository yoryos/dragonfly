This code makes an explicit cython class that wraps the C++ class
created for the CSTMD1, exposing it in python.

to install:

$ python setup.py install --user

or simply use setup.sh


explanation: 

The C++ code inside the src folder is wrapped using wrapper.pyc, by
referencing the C-header file and converting the C-class into a cppclass
in python as well as redeclaring the used member functions

setup.py compiles the wrapper.pyx and the .cu file into an extension .so
file 

This .so extension is then imported in test.py as a library. The
functionality of the CSTMD1 module is then tested inside test.py

Note that the way this code is set up, it will install the extension
.so file created by setup.py where all your python libraries are, so
once you run the install command, you can use it from anywhere on your machine
