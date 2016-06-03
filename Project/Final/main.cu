/* Runs the CPU and GPU implementations of my Pseudo Random Number Generator Project for
CS 179 at Caltech.
Author: Kyle Seipp

This file will demonstrate 4 algorithms for generating random numbers.
Before we start, we will check the tests by writing 0.5 to the whole file.
First, we will use the built-in random library for C++ to check the 
test libraries we are going to use and build.
The second algorithm will be Linear Congruential Generator.
The third algorithm will be My algorithm. Since it uses sine in the same
way that the LCG uses a linear function, I will call it a Sinusoidal
Congruential Generator.
The fourth algorithm will be an Xor-Shift Algorithm.

Then, we will run the same algorithms via a GPU.
Before starting, we will check that memcopy-ing works by transfering
0.5s to the GPU, and back.
First, we will use cuRAND.
Algorithms 2-4 will be implemented by finding N seeds, and starting
the PRNG in N different places. This requires careful thought.
    * We must generate a large sample to see speedup because we will
    be generating the seeds on the CPU. 
    * We could generate these seeds by seeding based on the current time
    but then we would see a lot of repetition. E.g.
    seed, f(seed), f(f(seed)), ..., f(seed),f(f(seed)), ..., f(f(seed)), ...
    where each of those sections would be the output of a different GPU thread.
    * We could generate the seeds using the C++ builtin RNG, but that feels impure
    because we shouldn't have to use any builtin RNGs.
    * We could seed by hand as long as I could generate enough random numbers
    by hand to satisfy sufficiently many seeds as for an upper bound of threads.
    Based on the architectures we have, 1024 seeds is likely sufficient.
    * With that in mind, we can pull 1024 seeds from /dev/random and hard code
    those, just as we hardcoded the seeds for the CPU demo for reproduceability.

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

// GPU REquirements
#include "ta_utilities.hpp"
#include <cuda.h>
#include <curand.h>

using namespace std;

// Kernel code
__global__
cudaConstantKernel(const unsigned int *d_input, unsigned long int *d_output, 
    unsigned int targetBytes, unsigned int const_val)
{
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while(thread_index < (unsigned int)targetBytes)
    {
        // Assigns the constant value everywhere
        d_output[thread_index] = const_val;
        // Update the thread index.
        thread_index += (blockDim.x * gridDim.x);
    }
}
__global__
cudaLCGKernel(const unsigned int *d_input, unsigned long int *d_output, unsigned int targetBytes, 
    unsigned int a, unsigned int c, unsigned int m)
{
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int x = d_input[thread_index];
    while(thread_index < (unsigned int)targetBytes)
    {
        // Assigns the value.
        d_output[thread_index] = x;
        x = (a*x + c) % m;
        // Update the thread index.
        thread_index += (blockDim.x * gridDim.x);
    }
}
__global__
cudaSCGKernel(const unsigned int *d_input, unsigned long int *d_output, unsigned int targetBytes, 
    unsigned int k)
{
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int y_prime = d_input[thread_index];
    double y = d_input[thread_index];
    while(thread_index < (unsigned int)targetBytes)
    {
        // Assigns the value.
        d_output[thread_index] = y_prime;
        y = pow(2,63) * sin(y) + k;
        y_prime = y;
        // Update the thread index.
        thread_index += (blockDim.x * gridDim.x);
    }
}

__global__
cudaXORKernel(const unsigned int *d_input, unsigned long int *d_output, unsigned int targetBytes, 
    unsigned int mult, unsigned int s1, unsigned int s2, unsigned int s3)
{
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int z = d_input[thread_index];
    while(thread_index < (unsigned int)targetBytes)
    {
        // Assigns the value. Note that the ultiplication is only for the final answer,
        // not part of computing the next iteration.
        d_input[thread_index] = (unsigned int) z * UINT64_C(mult);
        // Computes the next value
        z ^= z >> s1;
        z ^= z << s2;
        z ^= z >> s3;
        // Update the thread index.
        thread_index += (blockDim.x * gridDim.x);
    }
}

// Kernel Callers
void cudaCallConstantKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock, 
    const unsigned int *d_input,
    unsigned long int *d_output,
    const unsigned int targetBytes,
    const unsigned int const_val) {
        cudaConstantKernel<<<blocks, threadsPerBlock>>> (d_input, d_output, targetBytes, const_val);
}
void cudaCallLCGKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock, 
    const unsigned int *d_input,
    unsigned long int *d_output,
    const unsigned int targetBytes,
    const unsigned int a,
    const unsigned int c,
    const unsigned int m) {
        cudaLCGKernel<<<blocks, threadsPerBlock>>> (d_input, d_output, targetBytes, a,c,m);
}
void cudaCallSCGKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock, 
    const unsigned int *d_input,
    unsigned long int *d_output,
    const unsigned int targetBytes,
    const unsigned int k) {
        cudaSCGKernel<<<blocks, threadsPerBlock>>> (d_input, d_output, targetBytes, k);
}
void cudaCallXORKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock, 
    const unsigned int *d_input,
    unsigned long int *d_output,
    const unsigned int targetBytes,
    const unsigned int mult
    const unsigned int s1,
    const unsigned int s2,
    const unsigned int s3) {
        cudaXORKernel<<<blocks, threadsPerBlock>>> (d_input, d_output, targetBytes, mult,s1,s2,s3);
}

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

    /*********************************
    CPU SECTION
    We are still running this part for side by side comparisons
    *********************************/

    // First we will check that the tests are working correctly by generating
    // a non-random set of digits.
    output_file.open("constant_rng.txt");
    cout << "Writing " << targetBytes << " equal floats to constant_rng.txt" << endl;
    output_file << "# generator constant seed = n/a\ntype: d\ncount: "<<targetBytes <<
        "\nnumbit: 64" << endl;
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

    output_file.open("SCG_rng.txt");
    cout << """Writing ints via my rng with k=19,y_0=7"""
        """to SCG_rng.txt""" << endl;
    output_file << "# generator SCG_RNG seed = 7\ntype: d\ncount: "<<targetBytes <<
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

    /*********************************
    GPU SECTION
    Now for the fast part.
    *********************************/

    // First, we need all of the seeds. For our current architectures,
    // we can't have more than 1024 threads, so we only need that many
    // seeds.
    // These seeds were generated by seed_gen.py
    int seeds[1024] = {
     2097221417
    ,
     2662086441
    ,
      161921528
    ,
     1110552507
    ,
     1803310772
    ,
     3739158550
    ,
     4130710869
    ,
     2631142762
    ,
     3638735692
    ,
     3112991656
    ,
     3281222882
    ,
     3494342845
    ,
     4199891836
    ,
       14731281
    ,
      575359658
    ,
     3333656029
    ,
     2856374882
    ,
      938656246
    ,
     2561520986
    ,
     3516898075
    ,
     2175735638
    ,
      784332572
    ,
     4061571475
    ,
     3548293598
    ,
     1802552924
    ,
     4242430280
    ,
       96756922
    ,
     3534105824
    ,
     1007655828
    ,
     3097004058
    ,
     2026135474
    ,
     3261161145
    ,
     3761712042
    ,
      133092535
    ,
     1196763385
    ,
     1495904526
    ,
     3270524056
    ,
     2316613525
    ,
     1545802442
    ,
      432094012
    ,
     2483166042
    ,
     3177195615
    ,
      922354374
    ,
     2308679375
    ,
     4144519767
    ,
     2970557127
    ,
     1666459174
    ,
     4172156848
    ,
      903484490
    ,
     2529245207
    ,
     3283349229
    ,
     3191543775
    ,
      212651056
    ,
     1311380102
    ,
     2432817699
    ,
     1191626848
    ,
     3602414053
    ,
     1417305221
    ,
     2622196923
    ,
      359755950
    ,
     3975491234
    ,
     1682088008
    ,
     1709616175
    ,
     2886214004
    ,
     2753472423
    ,
     2015649497
    ,
     3141400130
    ,
     3759886216
    ,
     1801437764
    ,
      333232844
    ,
      297100574
    ,
     1272777297
    ,
      132710540
    ,
     1838860945
    ,
     3896043929
    ,
     1817241073
    ,
      470465540
    ,
     3860698972
    ,
     2618121736
    ,
     1386874649
    ,
     2015729935
    ,
      862616103
    ,
     3645598195
    ,
     3337520590
    ,
     1684769412
    ,
     1223149571
    ,
     1715066824
    ,
     1085525381
    ,
     1884505212
    ,
     3994853219
    ,
      500686960
    ,
     2498040968
    ,
     1819793791
    ,
     3453267347
    ,
     2441366531
    ,
     2647113302
    ,
     2020068339
    ,
     2787031405
    ,
     4028586939
    ,
     4258556833
    ,
      421070721
    ,
     4033195947
    ,
     2561951393
    ,
      662359512
    ,
      132368796
    ,
     2894563596
    ,
     2624056847
    ,
     1218174381
    ,
      186449823
    ,
     3606892269
    ,
     4118620858
    ,
     3172076987
    ,
      771707180
    ,
     3983438573
    ,
     2873028918
    ,
     2231052109
    ,
     2674515763
    ,
     2528161505
    ,
     2167854366
    ,
     3384536817
    ,
     1307214911
    ,
     2189312106
    ,
     3480866868
    ,
      833206155
    ,
     4034671195
    ,
     1772217295
    ,
      184807750
    ,
     1971751537
    ,
     4002527788
    ,
     4059850032
    ,
     2598945672
    ,
      166018048
    ,
     1878839160
    ,
      411119049
    ,
     2767796116
    ,
     2103297589
    ,
     2102768350
    ,
      816206023
    ,
     3327164455
    ,
     2869651910
    ,
     2107123438
    ,
     3181627298
    ,
     3174467418
    ,
     1618892304
    ,
     2838999882
    ,
      231012592
    ,
      242875153
    ,
     1117792943
    ,
     4095761412
    ,
     2088459346
    ,
     2242576702
    ,
     2495978839
    ,
     1104729138
    ,
     3849185296
    ,
     2393529227
    ,
      518979882
    ,
     1378907729
    ,
      810366733
    ,
     3234602943
    ,
      479415141
    ,
     2948433706
    ,
     3183590958
    ,
     3153731697
    ,
     3030852650
    ,
     3751835815
    ,
     2223122556
    ,
       24810180
    ,
     3199934390
    ,
     4216970090
    ,
     2760228181
    ,
     2767718537
    ,
     3251030177
    ,
     2300951637
    ,
     3728765056
    ,
     2522357925
    ,
      609820867
    ,
     3880254010
    ,
     3694956927
    ,
      490667520
    ,
     3075976260
    ,
      672118130
    ,
     3532178843
    ,
     4059059802
    ,
     1513038403
    ,
     1172610978
    ,
      637892580
    ,
     3314271939
    ,
     2794158950
    ,
      675406522
    ,
      318913214
    ,
     2934592035
    ,
     3440665108
    ,
      934453807
    ,
     1166361325
    ,
      698570688
    ,
     1161983951
    ,
      604382671
    ,
     1029764012
    ,
     3125914204
    ,
     3268021619
    ,
     1510040543
    ,
     2756016178
    ,
     3290591297
    ,
     2377401279
    ,
      552113414
    ,
     2297919955
    ,
     1547640831
    ,
     2544464835
    ,
      392372468
    ,
     4113425859
    ,
     3597168297
    ,
     4169846012
    ,
     3714605734
    ,
     3222106280
    ,
     2310270640
    ,
     1165269733
    ,
     3851729013
    ,
     4034627515
    ,
     3505431260
    ,
     4032699077
    ,
     1086306262
    ,
      225733497
    ,
     3917272184
    ,
       48744809
    ,
     3318603524
    ,
     3709575474
    ,
      915061080
    ,
     3253261723
    ,
     2723541811
    ,
     1764657673
    ,
     2087564283
    ,
     3672835055
    ,
     3374238867
    ,
     4057522925
    ,
     3775487531
    ,
      751749952
    ,
     4149100481
    ,
     1022020401
    ,
     1928724936
    ,
     4049682852
    ,
     1940417699
    ,
     4201312673
    ,
       62200342
    ,
     3175118824
    ,
      916777858
    ,
     1987257957
    ,
      287123385
    ,
      236809925
    ,
      276317734
    ,
     4008620124
    ,
     3190240842
    ,
      990430360
    ,
     2366156912
    ,
     2002140217
    ,
     1098328074
    ,
     3977198436
    ,
     3190876932
    ,
     1981540377
    ,
       97997321
    ,
     2065127163
    ,
     1925430166
    ,
     2792679414
    ,
     2895015109
    ,
     2329214984
    ,
       59043079
    ,
     3592088560
    ,
     1188334230
    ,
      265321693
    ,
      120115311
    ,
      512022997
    ,
     3566296678
    ,
     3579698546
    ,
     2988870099
    ,
     2698058454
    ,
     3411522971
    ,
     3502587921
    ,
      305369630
    ,
     1531152959
    ,
     1711852223
    ,
     3548079522
    ,
     3211539115
    ,
      610748876
    ,
      604152925
    ,
      686647277
    ,
     3549451372
    ,
     4080071134
    ,
     3390972876
    ,
     3791948494
    ,
     4272312083
    ,
     4136636776
    ,
      557177545
    ,
     4160611945
    ,
     4133064504
    ,
      959736146
    ,
     3440580783
    ,
       85087391
    ,
      603663430
    ,
     1957249276
    ,
     1093898090
    ,
     3401989923
    ,
     4267753434
    ,
     1525327258
    ,
     2532234817
    ,
     3585311325
    ,
      345154588
    ,
     4177762265
    ,
     3344454571
    ,
     1337572241
    ,
     1687140605
    ,
     1266594155
    ,
      266856963
    ,
     1958547908
    ,
      253374204
    ,
     1369757502
    ,
     1121693364
    ,
     3420955626
    ,
     1369696538
    ,
     3460644274
    ,
     2823798142
    ,
     1981123977
    ,
     3842009341
    ,
     1013913695
    ,
     2325594086
    ,
     2760847395
    ,
     2106312373
    ,
     3436715411
    ,
     2041279428
    ,
     1693320416
    ,
     3615110725
    ,
     2324099825
    ,
     1443158866
    ,
     2058661537
    ,
      115343987
    ,
     1805168000
    ,
     4128629917
    ,
     4275114707
    ,
     2925682793
    ,
     1995573346
    ,
      468499207
    ,
     2926781384
    ,
      758343637
    ,
     1917812604
    ,
      321447440
    ,
     4181616279
    ,
     2722593307
    ,
     3396174483
    ,
     3861545669
    ,
     2910216238
    ,
       69962269
    ,
     3523095348
    ,
     2532387036
    ,
     2059872046
    ,
     1295374790
    ,
     1610728791
    ,
     2774768279
    ,
     1442585662
    ,
      801650986
    ,
     1421521807
    ,
     3752064140
    ,
     3984830656
    ,
     1371383120
    ,
     4206614490
    ,
      113157623
    ,
     3495319026
    ,
     3526558741
    ,
     1568391662
    ,
      262126233
    ,
     2820526436
    ,
      269138987
    ,
     2625996222
    ,
     3182194565
    ,
      483717290
    ,
      365195900
    ,
       63053948
    ,
     1841689347
    ,
     2729105559
    ,
     4117291101
    ,
     1034716998
    ,
      887131876
    ,
     2356866069
    ,
      311762211
    ,
     2116036432
    ,
     1750861044
    ,
     2121495348
    ,
     2668852532
    ,
      475132999
    ,
     3136430735
    ,
     1834436303
    ,
      777257268
    ,
     3325634590
    ,
     3400074945
    ,
     1722620959
    ,
     4154967040
    ,
     3677427414
    ,
     3223052151
    ,
     3904089584
    ,
     1413990182
    ,
     2752921051
    ,
      408201131
    ,
     4171197370
    ,
      169819102
    ,
     2697364121
    ,
       97431694
    ,
     2678353982
    ,
       71884795
    ,
     2542887292
    ,
     1268513853
    ,
      168598175
    ,
     1610373876
    ,
     1376667052
    ,
     3786550407
    ,
     1329259036
    ,
       42711252
    ,
     2959990952
    ,
     3755291895
    ,
     1606074017
    ,
     2405822633
    ,
     2312356578
    ,
     3955359090
    ,
     2330310585
    ,
     3939095662
    ,
     1668401266
    ,
     1171444998
    ,
     2624560109
    ,
     2547001159
    ,
     1355515798
    ,
     2110872998
    ,
     1136321950
    ,
     1477478666
    ,
     1061680900
    ,
      673403574
    ,
     2237026286
    ,
     1411022725
    ,
     2216552898
    ,
     3240707901
    ,
     3174559495
    ,
      252201780
    ,
     2066679465
    ,
     1540651907
    ,
     2055751181
    ,
     3831639053
    ,
      493872003
    ,
     4199242027
    ,
      585670378
    ,
     1357471458
    ,
     1268163585
    ,
     2232240315
    ,
     2835398727
    ,
      833989325
    ,
      946530136
    ,
     1642765041
    ,
     3610646476
    ,
     1475714802
    ,
     2817305744
    ,
       43498201
    ,
     3532956492
    ,
     2493778514
    ,
     3759110448
    ,
     4209761103
    ,
     2393701721
    ,
     3347376891
    ,
      370484932
    ,
     1566923321
    ,
     2948802167
    ,
     1410098522
    ,
     1709227144
    ,
      501311192
    ,
       45020824
    ,
     1273009416
    ,
     1488573443
    ,
      743483708
    ,
     3455493164
    ,
     2736565383
    ,
     2832916954
    ,
      984950671
    ,
     1773030966
    ,
     3686682163
    ,
     2404189788
    ,
     1577851685
    ,
     2833473478
    ,
     1726960151
    ,
      713173908
    ,
     1499939683
    ,
     1467033073
    ,
     2928101805
    ,
     3460834068
    ,
      114797881
    ,
     1287404884
    ,
     3790819130
    ,
      556062663
    ,
      339209125
    ,
       21935861
    ,
      464738894
    ,
      924765083
    ,
     1223232798
    ,
     3241873927
    ,
     2666859999
    ,
      367609806
    ,
     3918575486
    ,
        6046253
    ,
     1822871265
    ,
     3728132944
    ,
     3507665477
    ,
     1657246751
    ,
     1274431050
    ,
     3278291279
    ,
     3199027873
    ,
     1382008297
    ,
     1886476213
    ,
     2588498347
    ,
       50085655
    ,
     2544047388
    ,
     1340316141
    ,
     1009533942
    ,
     3692037083
    ,
      821058824
    ,
     1099098807
    ,
     2755953006
    ,
     3102843538
    ,
      242806483
    ,
     2211335619
    ,
     1293753497
    ,
      158902395
    ,
      979079427
    ,
      305886184
    ,
     1181659564
    ,
      156100821
    ,
     1487497893
    ,
     3029076394
    ,
     1256394808
    ,
      133811610
    ,
     1467072152
    ,
     2265216101
    ,
      214971429
    ,
     1246741072
    ,
      558791877
    ,
     4274890297
    ,
      813360733
    ,
     3915099588
    ,
     4074508368
    ,
     2792625949
    ,
     2191600496
    ,
     1964626705
    ,
      970806822
    ,
     3402259295
    ,
       59346428
    ,
     1251315985
    ,
     2562849807
    ,
     1210556722
    ,
      768867552
    ,
     2494347868
    ,
      400538887
    ,
     2825975425
    ,
      970163984
    ,
      936336982
    ,
     3998364682
    ,
     4046049562
    ,
     1681347957
    ,
     3835598544
    ,
     3279540436
    ,
      964781587
    ,
      577204779
    ,
     2404346685
    ,
     4290172719
    ,
      365731653
    ,
      662975128
    ,
     1584701907
    ,
     1980410792
    ,
     3591671630
    ,
     3094910561
    ,
     1303452499
    ,
     1164495713
    ,
      370786986
    ,
     2708591125
    ,
     1128975064
    ,
      182350224
    ,
     2998399472
    ,
     2263773099
    ,
      504644276
    ,
     1848822682
    ,
     1466968886
    ,
     2842888334
    ,
     1914385392
    ,
      580222396
    ,
     3978696016
    ,
      149145834
    ,
       92436934
    ,
      285368391
    ,
     3414083856
    ,
     2974587265
    ,
     1680510752
    ,
      884838681
    ,
     2333323884
    ,
     3546175113
    ,
     1979781528
    ,
     3560185411
    ,
     3492445162
    ,
     1039793694
    ,
     2979471208
    ,
      912898999
    ,
     1414619566
    ,
     3092365594
    ,
     2091519930
    ,
     1617651279
    ,
     1941679218
    ,
     2372014434
    ,
     3314397171
    ,
     1804743672
    ,
        9485799
    ,
     3471295602
    ,
      701590628
    ,
     2735761867
    ,
     1994733805
    ,
     2187849122
    ,
     2747852357
    ,
     4272418548
    ,
     3491035326
    ,
      637743670
    ,
     3867478088
    ,
     1622661709
    ,
     1340171608
    ,
     2645532882
    ,
     1048030304
    ,
     1027802432
    ,
     3909705281
    ,
     4250721083
    ,
     1988709281
    ,
     1183152042
    ,
      611052063
    ,
     3583830722
    ,
     1634722379
    ,
      255448264
    ,
     3328676338
    ,
     4193443525
    ,
      484912970
    ,
     2228666780
    ,
      847922287
    ,
     1980803432
    ,
     1762707779
    ,
     3139629195
    ,
     1496830911
    ,
      159244175
    ,
     2333446484
    ,
     1112397988
    ,
     2883112783
    ,
     3206054534
    ,
     2299138922
    ,
      493990816
    ,
     1140582615
    ,
     3614128815
    ,
     4206397975
    ,
     2656062876
    ,
       59598131
    ,
     1439262316
    ,
     2143862017
    ,
      671882613
    ,
      785487552
    ,
      737864346
    ,
      647595473
    ,
     3623981082
    ,
     1251382798
    ,
     2249492680
    ,
     3402041334
    ,
      620160090
    ,
      882153870
    ,
     1704445570
    ,
     2699114342
    ,
     2060745967
    ,
      559598018
    ,
      170038577
    ,
     3554068365
    ,
      723342702
    ,
     1389460694
    ,
     1739216915
    ,
     2740638846
    ,
     2750534334
    ,
      548810098
    ,
      600103186
    ,
      931553784
    ,
     2796700388
    ,
      625476231
    ,
     1421754508
    ,
      585722364
    ,
      690835486
    ,
     3572269428
    ,
      315494236
    ,
       38060817
    ,
      589814176
    ,
      613074119
    ,
     1601065012
    ,
     1351969604
    ,
     2103765138
    ,
     2518215231
    ,
     2048157713
    ,
     2570416872
    ,
     2963541515
    ,
      104427604
    ,
     2174476051
    ,
     2488403130
    ,
      845934587
    ,
      599015033
    ,
      581097663
    ,
     1246133673
    ,
     4264299605
    ,
     3258172252
    ,
     3967554877
    ,
      342281340
    ,
     2803620372
    ,
     3638606054
    ,
     3635118953
    ,
     2946332484
    ,
     1122784276
    ,
     4039094674
    ,
      397696387
    ,
     2704584370
    ,
     3147190117
    ,
     2792539555
    ,
     1774659723
    ,
     3556681046
    ,
       44790713
    ,
     2770396005
    ,
      459632886
    ,
     4105530137
    ,
     1451742126
    ,
     4205859770
    ,
     1586699137
    ,
     1011334643
    ,
      120391935
    ,
     4288639580
    ,
     2043818369
    ,
      526502912
    ,
     3551879206
    ,
     4255470335
    ,
     2774889296
    ,
     2607521443
    ,
     2712962277
    ,
     1581873275
    ,
     3322067258
    ,
      213263181
    ,
      544621620
    ,
      919598497
    ,
      492008228
    ,
      509953102
    ,
     4064604125
    ,
      478948163
    ,
      906584989
    ,
     1340249624
    ,
      955751311
    ,
      269273296
    ,
     3777082134
    ,
     3000702959
    ,
     1265564725
    ,
     3874876513
    ,
      897337802
    ,
     1578380100
    ,
     3143938260
    ,
     2607427589
    ,
     3120966311
    ,
     1195711471
    ,
     3754058244
    ,
       42956621
    ,
      157648619
    ,
     2669339697
    ,
     1859771110
    ,
     2122567455
    ,
     1405892115
    ,
      799523958
    ,
      650142838
    ,
      794771308
    ,
     1818588600
    ,
     2263361994
    ,
     3705217203
    ,
     1084307471
    ,
     1710749663
    ,
      777525248
    ,
      731149812
    ,
      732156276
    ,
     4053477977
    ,
     2177729687
    ,
     3438600043
    ,
      832068461
    ,
     3664818420
    ,
     2747474304
    ,
      164052468
    ,
      598833122
    ,
     2243492136
    ,
     1875828996
    ,
     3589026375
    ,
     3794665432
    ,
      719447782
    ,
     3701789451
    ,
     3721095939
    ,
      567750527
    ,
      942228703
    ,
      248923598
    ,
     2800546236
    ,
     4083509634
    ,
     2232565489
    ,
     1356331474
    ,
      415905450
    ,
     2704210056
    ,
     1914656749
    ,
     1070282826
    ,
     1754190360
    ,
     1168426019
    ,
     3133894131
    ,
     2757269234
    ,
     1767321176
    ,
       21121250
    ,
      458538930
    ,
     4090025529
    ,
      101429864
    ,
     1161283946
    ,
     1430987573
    ,
     1885518397
    ,
      404606538
    ,
     2883112561
    ,
     1953349353
    ,
     1646523720
    ,
      127261972
    ,
     3263995444
    ,
      643419609
    ,
     3529027268
    ,
     1905895539
    ,
     3541650784
    ,
     1508335595
    ,
     3490482298
    ,
     3015311840
    ,
     3689169270
    ,
     2845624854
    ,
      151151182
    ,
     1010386061
    ,
     2442250553
    ,
      858759087
    ,
      297325967
    ,
     1822044397
    ,
     2320345751
    ,
     2704592845
    ,
     3275062926
    ,
     1317836200
    ,
     3044280932
    ,
      867597254
    ,
     2489859161
    ,
      127265531
    ,
     1786030662
    ,
     2940311190
    ,
     2086838088
    ,
     3172465646
    ,
     3171132993
    ,
     1641008520
    ,
       95585092
    ,
     2036620229
    ,
     2068172538
    ,
     1509546902
    ,
     2029006711
    ,
     1587621577
    ,
     1401038822
    ,
     2272886945
    ,
     3369975781
    ,
     4183972954
    ,
     2605528240
    ,
      414741663
    ,
     3754250150
    ,
     1604770253
    ,
     4123209195
    ,
     1381820126
    ,
     1253292357
    ,
     2476391749
    ,
     1200148608
    ,
     1806695378
    ,
     2112085669
    ,
     2123000746
    ,
     1168169765
    ,
     3934897399
    ,
     3937462453
    ,
     3857314085
    ,
     4264885251
    ,
     4001379018
    ,
     3633748973
    ,
     3169440261
    ,
     3554584104
    ,
     4026974749
    ,
     2494811324
    ,
     2629800632
    ,
     2458117610
    ,
     1556074063
    ,
     2805767611
    ,
     3442854852
    ,
      163471534
    ,
     2602256121
    ,
     3669083798
    ,
     2813453330
    ,
      619922621
    ,
     4134002083
    ,
     1977711700
    ,
     1026120996
    ,
     1621637442
    ,
     2924399799
    ,
     2778816627
    ,
      990800896
    ,
     1720917768
    ,
     3787103031
    ,
     4182938396
    ,
     3425496340
    ,
     3291579494
    ,
     4268434578
    ,
       63679130
    ,
     1650041002
    ,
       86410183
    ,
      122061286
    ,
     3386879152
    ,
       69490464
    ,
      360428378
    ,
       18609669
    ,
      427328703
    ,
     3232012458
    ,
     4193147631
    ,
     3143457199
    ,
     3915423892
    ,
      610152926
    ,
     3099079352
    ,
      270618440
    ,
     1879665021
    ,
     1753154144
    ,
      948478674
    ,
     1202703286
    ,
      347777614
    ,
     3778449341
    ,
     3291815802
    ,
     1165988657
    ,
     3662754934
    ,
     2615528349
    ,
     1479213331
    ,
     2998223486
    ,
     2940473991
    ,
     1527461799
    ,
     3868737096
    ,
     1799009272
    ,
     1290079590
    ,
     3210911389
    ,
      801667198
    ,
      777384901
    ,
     2014322897
    ,
     1507764831
    ,
     1020348756
    ,
       75828889
    ,
     1548209020
    ,
     1239081468
    ,
     2748131518
    ,
     1710617357
    ,
     3468807215
    ,
     4272029955
    ,
     3555887640
    ,
      468878643
    ,
     1345855678
    ,
      799026974
    ,
     1370639535
    ,
     3892543411
    ,
     2582498316
    ,
     2558303127
    ,
      514519919
    ,
     2102392072
    ,
     1089077914
    ,
      405818918
    ,
      869774153
    ,
     1734681413
    ,
     3847084506
    ,
      990015599
    ,
      921126939
    ,
     3699684231
    ,
     3532318500
    ,
     4101595454
    ,
     1387888433
    ,
     1781151900
    ,
     1311045748
    ,
     2389667890
    ,
     3499350427
    ,
     4125065333
    ,
     3298170311
    ,
     1262190929
    ,
     2699585865
    ,
     1107979468
    ,
      612698354
    ,
     2183432440
    ,
     3065655427
    ,
     1902083248
    ,
     3942245087
    ,
      187217937
    ,
     2712917408
    ,
      156433532
    ,
     2064445895
    ,
     3256123807
    ,
       50781490
    ,
     2337858779
    ,
      350354728
    ,
     1990192444
    ,
     3685873914
    ,
     1050878066
    ,
       31776283
    ,
     1594687412
    ,
      778636853
    ,
     2787557534
    ,
     3160180549
    ,
     3570138587
    ,
     4281311610
    ,
     1320975469
    ,
     2448516777
    ,
     3928493781
    ,
     2475902687
    ,
      416463868
    ,
      707767320
    ,
      546105199
    ,
     2829490564
    ,
     3693639448
    ,
     1501681957
    ,
     1561488911
    ,
     4210250125
    ,
     1782488305
    ,
     1406417617
    ,
     2299995644
    ,
     1210620648
    ,
     1692807536
    ,
     1806646184
    ,
     2579804289
    };
    // We also will simply define blocks and threadsPerBlock here rather than
    // take them as arguments. It wil be less confusing in the CPU code to
    // not require them.
    // Never raise above 1024, or we won't have enough seeds and the code will fail
    int blocks = 512;
    int local_size = 200;

    // Now that we've initialized all of the seeds, we can follow the same
    // format as most GPU code.

    // These functions allow you to select the least utilized GPU 
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these 
    // functions if you are running on your local machine.
    TA_Utilities::select_coldest_GPU();
    int max_time_allowed_in_seconds = 10;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    // Rather than use CUDA's timing code, I am using the standard c++
    // timer code so that there is consistency between the CPU and GPU
    // timers. 

    // Allocate host memory
    int *input = seeds;
    // We could use fewer seeds if we are generating less than 1024 numbers.
    // However, that is not the intended use-case of this PRNG. Generating 1024 random
    // integers can be done by the slowest PRNG in the CPU demo in 0.001657 seconds.
    // With that in mind, trying to use less than 1024 values will just end up using
    // the seeds themselves which come from /dev/urandom, making them acceptable.
    float *output = new long int[targetBytes];

    // First we will check that the tests are working correctly by generating
    // a non-random set of digits.
    output_file.open("constant_rng_GPU.txt");
    cout << "Writing " << targetBytes << " equal floats to constant_rng_GPU.txt" << endl;
    output_file << "# generator constant seed = n/a\ntype: d\ncount: "<<targetBytes <<
        "\nnumbit: 64" << endl;

    clock_t begin = clock();

    // Allocate device memory
    int *d_input;
    long int *d_output;
    gpuErrChk(cudaMalloc(&d_input, 1024 * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_output, targetBytes * sizeof(long int)));

    // Copy input to GPU
    gpuErrChk(cudaMemcpy(d_input, input, 1024 * sizeof(int), 
        cudaMemcpyHostToDevice));
    // Runs Kernel
    cudaCallConstantKernel(blocks, local_size, d_input, d_output, targetBytes, 5);
    // Copies data back.
    gpuErrChk(cudaMemcpy(output, d_output, targetBytes * sizeof(long int), 
                cudaMemcpyDeviceToHost));

    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << output[i];
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Note. We could keep the seeds on the GPU to speed things up. However,
    // part of the slowdowns of using a GPU is the memcopying. It would be unfair
    // to the first generator if copying the data was part of the time, and only it
    // had to do the copying. In addition, saving the data to a file should be part
    // of how long it takest to run. Therefore, so should the copying of data.
    // Free GPU memory.
    gpuErrChk(cudaFree(d_input));
    gpuErrChk(cudaFree(d_output));

    // BUILTIN
    // Now, we will use cuRAND to test against the builtin random genertor.
    output_file.open("builtin_rng_GPU.txt");
    cout << "Writing " << targetBytes << " equal floats to builtin_rng_GPU.txt" << endl;
    output_file << "# generator constant seed = n/a\ntype: d\ncount: "<<targetBytes <<
        "\nnumbit: 64" << endl;

    clock_t begin = clock();

    // Allocate device memory
    int *d_input;
    long int *d_output;
    gpuErrChk(cudaMalloc(&d_output, targetBytes * sizeof(long int)));

    // Runs Kernel
    // We will just use cuRand API calls instead for this one
    curandGenerator_t gen;
    /* Create pseudo-random number generator */
    curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);
    
    /* Set seed */
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    /* Generate n floats on device */
    curandGenerateLongLong(gen, d_output, targetBytes);

    /* Copy device memory to host */
    cudaMemcpy(output, d_output, targetBytes * sizeof(long int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << output[i];
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Free GPU memory.
    curandDestroyGenerator(gen);
    gpuErrChk(cudaFree(d_output));

    // LCG
    // This is the first PRNG implementation for the GPU.
    // Though the seeds will be different, the other constants will be
    // the same as those from the CPU implementation.
    // x = (a*x + c) % m;
    output_file.open("LCG_rng_GPU.txt");
    cout << """Writing ints via the LCG a=1075731923,c=732971908,m=2^31-1,x_0=7"""
        """to LCG_rng_GPU.txt""" << endl;
    output_file << "# generator LCG seed = n/a\ntype: d\ncount: "<<targetBytes <<
    "\nnumbit: 64" << endl;

    // Recall that
    // const unsigned int a = 1075731923;  // Approximately 2^30 with some digits changed
    // const unsigned int c = 732971908;   // Approximately 2^29 with some digits changed
    // const unsigned long int m = pow(2,63)-1; // We want to use a large number that isn't a power of two
    // unsigned long int x = 7;
    // and has not gone out of scope.
    
    clock_t begin = clock();

    // Allocate device memory
    int *d_input;
    long int *d_output;
    gpuErrChk(cudaMalloc(&d_input, 1024 * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_output, targetBytes * sizeof(long int)));

    // Copy input to GPU
    gpuErrChk(cudaMemcpy(d_input, input, 1024 * sizeof(int), 
        cudaMemcpyHostToDevice));
    // Runs Kernel
    cudaCallLCGKernel(blocks, local_size, d_input, d_output, targetBytes, a, c, m);
    // Copies data back.
    gpuErrChk(cudaMemcpy(output, d_output, targetBytes * sizeof(long int), 
                cudaMemcpyDeviceToHost));

    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << output[i];
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Free GPU memory.
    gpuErrChk(cudaFree(d_input));
    gpuErrChk(cudaFree(d_output));

    // SCG
    // This is the my PRNG algorithm's implementation for the GPU.
    // Though the seeds will be different, the other constants will be
    // the same as those from the CPU implementation.
    // y = pow(2,63) * sin(y) + k;

    output_file.open("SCG_rng_GPU.txt");
    cout << """Writing ints with with  SCG k=19,y_0=7"""
        """to SCG_rng_CPU.txt""" << endl;
    output_file << "# generator LCG seed = n/a\ntype: d\ncount: "<<targetBytes <<
    "\nnumbit: 64" << endl;

    // Recall that
    // const unsigned int k = 19;
    // long double y = 7;
    // unsigned long int y_prime;
    // and has not gone out of scope.

    clock_t begin = clock();

    // Allocate device memory
    int *d_input;
    long int *d_output;
    gpuErrChk(cudaMalloc(&d_input, 1024 * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_output, targetBytes * sizeof(long int)));

    // Copy input to GPU
    gpuErrChk(cudaMemcpy(d_input, input, 1024 * sizeof(int), 
        cudaMemcpyHostToDevice));
    // Runs Kernel
    cudaCallSCGKernel(blocks, local_size, d_input, d_output, targetBytes, k);
    // Copies data back.
    gpuErrChk(cudaMemcpy(output, d_output, targetBytes * sizeof(long int), 
                cudaMemcpyDeviceToHost));

    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << output[i];
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Free GPU memory.
    gpuErrChk(cudaFree(d_input));
    gpuErrChk(cudaFree(d_output));

    // XORSHIFT
    // This is the my PRNG algorithm's implementation for the GPU.
    // Though the seeds will be different, the other constants will be
    // the same as those from the CPU implementation.
    // z ^= z >> 13;
    // z ^= z << 30;
    // z ^= z >> 19;
    // (unsigned int) z * UINT64_C(2685821657736338717);

    output_file.open("XOR_rng_GPU.txt");
    cout << """Writing ints via xor-shifting rng with z_0=11, multiplier of 2685821657736338717, and """
        """shifts 13,30, and 19 to XOR_rng_GPU.txt""" << endl;
    output_file << "# generator XOR_RNG seed = n/a\ntype: d\ncount: "<<targetBytes <<
    "\nnumbit: 64" << endl;

    clock_t begin = clock();

    // Allocate device memory
    int *d_input;
    long int *d_output;
    gpuErrChk(cudaMalloc(&d_input, 1024 * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_output, targetBytes * sizeof(long int)));

    // Copy input to GPU
    gpuErrChk(cudaMemcpy(d_input, input, 1024 * sizeof(int), 
        cudaMemcpyHostToDevice));
    // Runs Kernel
    cudaCallXORKernel(blocks, local_size, d_input, d_output, targetBytes, 2685821657736338717, 13, 30, 19);
    // Copies data back.
    gpuErrChk(cudaMemcpy(output, d_output, targetBytes * sizeof(long int), 
                cudaMemcpyDeviceToHost));

    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << output[i];
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Free GPU memory.
    gpuErrChk(cudaFree(d_input));
    gpuErrChk(cudaFree(d_output));

    return 0;
}