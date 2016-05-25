# Tests the Pseudo Random Number Generator Project for
# CS 179 at Caltech.
# Author: Kyle Seipp

# This file opens the named files generated by cpu_main.cc and uses the dieharder
# tests on the data therein. It will tell us how "random," seeming the 
# pseudo random numbers generated are.

import os

print "Please be sure we are in the same folder as cpu_main was run."
print "We will now run the dieharder test suite on the generated text files."
print "This will take a long time."
print "Not all tests will pass because dieharder is a very robust suite. It is"
print "not designed to use text files of integers. I chose this method because"
print "it provides more reproduceable results which is important for a demo."
print
print "Testing the constant RNG."
# -a means all tests
# -g 202 means tells dieharder the format of the data (bitstream vs ints)
# -f constant_rng.txt means take text input from that file
os.system("dieharder -a -g 202 -f constant_rng.txt")
print "Done"
print
print "Testing the builtin RNG."
os.system("dieharder -a -g 202 -f builtin_rng.txt")
print "Done"
print
print "Testing the LCG RNG."
os.system("dieharder -a -g 202 -f LCG_rng.txt")
# On 1000000 integers it passes the diehard_birthdays, diehard_parking_lot
# 2dsphere, 3dsphere
# weakly passes diehard_runs
print "Done"
print
print "Testing the MY RNG."
os.system("dieharder -a -g 202 -f MY_rng.txt")
# On 1000000 integers it passes diehard_birthdays, diehard_bitstream, 
# diehard_dna, diehard_count_ls_str, diehard_parking_lot
# 2dsphere, 3dsphere

diehard_runs, and rgb permutations.
# It weakly passes sts_monobit.
print "Done"
print
# Takes 678 minutes of realtime to get to this point in testing.
print "Testing the XorShift  RNG."
os.system("dieharder -a -g 202 -f XOR_rng.txt")
# On 1000000 integers it passes diehard_birthdays, diehard_bistream, 
# diehard_dna, diehard_Count_ls_str, diehard_parking_lot
# 2dsphere, 3dsphere
print "Done"