# This file generates 1024 random 4-byte integers (c++ ints).
# Those ints will then be hard coded into
# the main.cpp file in order to seed any number of threads
# PRNGs since the maximum thread count for the architectures
# availble is 1024 or less.

import os

cmd = "od -vAn -N4 -tu4 < /dev/urandom ;  echo ','"
os.system("echo '{'")
for i in range(1024):
    os.system(cmd)
os.system("echo '}'")