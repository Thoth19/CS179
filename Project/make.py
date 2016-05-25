# This file takes the place of a makefile because
# we are going to use tests.py to test the RNGs

import os

print "Compiling the CPU demon"
os.system("g++ --std=c++11 cpu_main.cc")
print "Running the code to create 1 000 000 random integers"
os.system("./a 1000000")
print "Running tests on those output files"
os.system("python tests.py")
print "DONE with demo."