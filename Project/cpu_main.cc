/* Runs the CPU implementation of my Pseudo Random Number Generator Project for
CS 179 at Caltech.
Author: Kyle Seipp

This file will demonstrate 5 algorithms for generating random numbers.
Before we start, we will check the tests by writing 0.5 to the whole file.
First, we will use the built-in random library for C++ to check the 
test libraries we are going to use and build.
Second, we will use values from /dev/random. This part will be supported
only on Linux machines. It may work on Mac and will not likely work on Windows
which does not have /dev/random.
The third algorithm will be ____.
The fourth algorithm will be ___.
The fifth algorithm will be ____.

We will use the DieHarder suite to test the quality of our RNGs. Instructions for 
its installation will be provided in the README. 
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <random>
#include <time.h>
#include <ctime>
#include <math.h>       /* sin */
#include <climits>
#include <stdint.h>

#define PI 3.14159265358979323846264338327950288419716939937510

using namespace std;

int main(int argc, char const *argv[])
{
    // First, we check that the number of arguments are correct.

    if (argc != 2){
        printf("Usage: (Number of floats)\n");
        std::exit(-1);
    }
    const unsigned int targetBytes = atoi(argv[1]);

    // Sets up a file object to send data to.
    ofstream output_file;

    // First we will check that the tests are working correctly by generating
    // a non-random set of digits.
    output_file.open("constant_rng.txt");
    cout << "Writing " << targetBytes << " equal floats to constant_rng.txt" << endl;
    output_file << "# generator constant seed = n/a\ntype: d\ncount: "<<targetBytes <<
        "\nnumbit: 32" << endl;
    clock_t begin = clock();
    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << "5\n";
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Second, we will use a builtin random number generator.
    output_file.open("builtin_rng.txt");
    srand (time(NULL));
    cout << """Writing floats from the builtin uniform real distribution """
        """to builtin_rng.txt""" << endl;
    output_file << "# generator builtin seed = based on time\ntype: d\ncount: "<<targetBytes <<
    "\nnumbit: 64" << endl;
    
    begin = clock();
    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << rand() <<"\n";
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Now, we will use /dev/random. Note that this might fail depending
    // on operating system.
    /*
    float rand_data[targetBytes];
    FILE *dev_rand = fopen("/dev/random","r");
    fread(&rand_data, 1, targetBytes, dev_rand);
    fclose(dev_rand);
    output_file.open("dev_random_rng.txt");
    cout << "Writing data from /dev/random to dev_random_rng.txt" << endl;
    output_file << rand_data;
    */

    // Weak Algo
    // We will implement a Linear Congruential Generator.
    // The formula is X_new = (a*X_old+c) mod m
    // It is known for being fast to compute and cheap in memory usage
    // however, the quality of randomness is low especially in lower bits
    // and it is not well suited to parallelization due to race conditions.
    // As mentioned in class, I intend to parallelize this RNG
    // by starting at multiple places and concatenating at the appropriate places
    // in the sequence.
    const unsigned int a = 1075731923;  // Approximately 2^30 with some digits changed
    const unsigned int c = 732971908;   // Approximately 2^29 with some digits changed
    const unsigned long int m = pow(2,63)-1; // We want to use a large number that isn't a power of two
    unsigned long int x = 7;
    
    output_file.open("LCG_rng.txt");
    srand (time(NULL));
    cout << """Writing ints via the LCG a=1075731923,c=732971908,m=2^31-1,x_0=7"""
        """to LCG_rng.txt""" << endl;
    output_file << "# generator LCG seed = 7\ntype: d\ncount: "<<targetBytes <<
    "\nnumbit: 64" << endl;
        
    begin = clock();
    output_file << x << "\n";
    for (int i = 1; i < targetBytes; ++i)
    {
        x = (a*x + c) % m;
        output_file << x <<"\n";
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // My Algo
    // I am going to use an algorithm based on the LCG that should be
    // "more random." In particular, it will be slower to compute, but 
    // should lead to a high degree of uniformity. The problems with 
    // parallelization will still exist and will be solved the same 
    // way. 
    // I originally intended to use Y_new = sin(sin(sin(Y_old)))
    // however, sine has a tendency to converge towards 0.
    // Thus, I instead propose Y_new = sin(y_old) + k
    // This will prevent convergence to 0 by adding a non-zero constant
    // large enough to force non-convergence. In addition, sine is periodic
    // with respect to 2*pi, so we can wrap around easily. As long as k is
    // not a multiple of pi, the period of the RNG function will be large.
    // We multiply by 2^63 because sine is limited to [0,1] and we need to
    // scale up.

    output_file.open("MY_rng.txt");
    cout << """Writing ints via my rng with k=19,y_0=7"""
        """to MY_rng.txt""" << endl;
    output_file << "# generator MY_RNG seed = 7\ntype: d\ncount: "<<targetBytes <<
    "\nnumbit: 64" << endl;

    const unsigned int k = 19;
    long double y = 7;
    unsigned long int y_prime;

    begin = clock();
    output_file << y << "\n";
    for (int i = 1; i < targetBytes; ++i)
    {
        y = pow(2,63) * sin(y) + k;
        y_prime = y;
        output_file << y_prime <<"\n";
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Stronger Algo
    // We will use an Xor-Shift Algorithm inspired by 
    // https://en.wikipedia.org/wiki/Xorshift
    // 
    // It should run faster than the LCG and MY algo because
    // it uses bit shifts and xors rather than arithmatic operations
    // In addition, it should pass many more tests. Standard xorshifts
    // will fail bigcrunch and other randomness tests, but by multiplying 
    // by a a constant that is invertible mod 2^64, we can avoid these problems
    // It is equivalent to choose a large value less than 2^64 that is 
    // relatively prime. 2685821657736338717 is one such number
    
     output_file.open("XOR_rng.txt");
    cout << """Writing ints via xor-shifting rng with z_0=11, multiplier of 2685821657736338717, and """
        """shifts 13,30, and 19 to XOR_rng.txt""" << endl;
    output_file << "# generator XOR_RNG seed = 7\ntype: d\ncount: "<<targetBytes <<
    "\nnumbit: 64" << endl;

    const unsigned int q = 4294867297; // multiplier
    uint64_t z=11; // Seeded random value

    begin = clock();

    output_file << z << "\n";
    for (int i = 1; i < targetBytes; ++i)
    {
        // xor shifting algo 
        z ^= z >> 13;
        z ^= z << 30;
        z ^= z >> 19;
        output_file << (unsigned int) z * UINT64_C(2685821657736338717) << "\n";
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    return 0;
}