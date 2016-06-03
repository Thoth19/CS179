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

// GPU Requirements
#include <cuda.h>
#include <curand.h>

using namespace std;

// Kernel code
__global__
void cudaConstantKernel(const unsigned long int *d_input, unsigned long int *d_output, 
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
void cudaLCGKernel(const unsigned long int *d_input, unsigned long int *d_output, unsigned int targetBytes, 
    unsigned long int a, unsigned long int c, unsigned long int m)
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
void cudaSCGKernel(const unsigned long int *d_input, unsigned long int *d_output, unsigned int targetBytes, 
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
void cudaXORKernel(const unsigned long int *d_input, unsigned long int *d_output, unsigned int targetBytes, 
    unsigned long int mult, unsigned int s1, unsigned int s2, unsigned int s3)
{
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int z = d_input[thread_index];
    while(thread_index < (unsigned int)targetBytes)
    {
        // Assigns the value. Note that the ultiplication is only for the final answer,
        // not part of computing the next iteration.
        d_output[thread_index] = z * mult;
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
    const unsigned long  int *d_input,
    unsigned long int *d_output,
    const unsigned int targetBytes,
    const unsigned int const_val) {
        cudaConstantKernel<<<blocks, threadsPerBlock>>> (d_input, d_output, targetBytes, const_val);
}
void cudaCallLCGKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock, 
    const unsigned long int *d_input,
    unsigned long int *d_output,
    const unsigned int targetBytes,
    const unsigned long int a,
    const unsigned long int c,
    const unsigned long int m) {
        cudaLCGKernel<<<blocks, threadsPerBlock>>> (d_input, d_output, targetBytes, a,c,m);
}
void cudaCallSCGKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock, 
    const unsigned long int *d_input,
    unsigned long int *d_output,
    const unsigned int targetBytes,
    const unsigned int k) {
        cudaSCGKernel<<<blocks, threadsPerBlock>>> (d_input, d_output, targetBytes, k);
}
void cudaCallXORKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock, 
    const unsigned long int *d_input,
    unsigned long int *d_output,
    const unsigned int targetBytes,
    const unsigned long int mult,
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
    const unsigned long int a = 1075731923;  // Approximately 2^30 with some digits changed
    const unsigned long int c = 732971908;   // Approximately 2^29 with some digits changed
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
    unsigned long int seeds[1024] = {
         15009086648056230862
        ,
          1386565886589582088
        ,
          3367698093339929639
        ,
          8442639455492736197
        ,
         10960571038789860670
        ,
          4382472817481091698
        ,
         13517369762482954398
        ,
          2232117127823227061
        ,
          7869210045519652651
        ,
          2751262079163232061
        ,
         11959675983418433134
        ,
           612419775625343251
        ,
          8795510778620040531
        ,
          6822367020645177951
        ,
          3948381899721689453
        ,
          4247563114722957162
        ,
          8401910170948285377
        ,
         18366339550946758155
        ,
         15514434962015269604
        ,
          7539959559249556907
        ,
          6248161516374715383
        ,
          6975072684335006393
        ,
         16997893873950856393
        ,
          1354758432964968477
        ,
         15456483340581219078
        ,
         12296950018663712799
        ,
          1077956030558981971
        ,
         16831002843353953996
        ,
         17980095519113297053
        ,
         10999976592696447901
        ,
          3982091147167892962
        ,
         12848725597482499350
        ,
         17191854214093390574
        ,
          8232945078171212022
        ,
          7006877048362836217
        ,
          2874094724074161733
        ,
         10666985705011766838
        ,
         16698677576040951679
        ,
           823366517066265101
        ,
          4716930161011406844
        ,
         15204038850938188648
        ,
          4381245316996745128
        ,
         14203178971771008722
        ,
          6684134859105771026
        ,
         10198198746345586387
        ,
         14215493679549521542
        ,
          3962707704961153953
        ,
          6667694773883259386
        ,
         14111510667873162386
        ,
           375403199221009688
        ,
          4974731756960764561
        ,
          3367237543247984496
        ,
         12737999711936806522
        ,
         16159312679929752234
        ,
          8841948002488488640
        ,
         14724704940953796532
        ,
         17404147299647151243
        ,
          3381943095397107140
        ,
         12666217121861465229
        ,
          6395326465895769294
        ,
          5895428537330612236
        ,
          1874243055670457588
        ,
         15457023963534878602
        ,
          5593545243681640314
        ,
          7981422021113919494
        ,
           734802194375220398
        ,
           399622418497547047
        ,
         10387528232606295447
        ,
          8531368883955030269
        ,
          9990416331043265840
        ,
          4638218464705524689
        ,
         12978606929236608162
        ,
         18136234671596037747
        ,
         15178036454280359076
        ,
         11882343470657543883
        ,
         18148974268439565782
        ,
          6679840546781892641
        ,
          1411166061243743220
        ,
         13054310395439953020
        ,
         10820983508501807803
        ,
          4332564311763644193
        ,
          9421023031916690362
        ,
          5211572614192304886
        ,
          3868757432234157603
        ,
         15590376020091981850
        ,
         18196036129597331333
        ,
          5239168282838426903
        ,
          7246959567183006837
        ,
         14640171347347391893
        ,
          9705964628233646155
        ,
          5136538901953499333
        ,
         17127156466461980151
        ,
          5850091237748261525
        ,
           421827855246491520
        ,
          3485532365405785748
        ,
          7226331863354320232
        ,
         17328681221497350048
        ,
          6891611993515316537
        ,
         15855276948503028595
        ,
         10621290709926028601
        ,
         12949831522461913942
        ,
          8572044002439356570
        ,
         15875949084218808854
        ,
          1880667152202773822
        ,
          4926499289424847405
        ,
         16575752365024361782
        ,
           820721419251465036
        ,
          2824394008966279771
        ,
         11680078967504010347
        ,
         11946422350233642894
        ,
          5644830768636638895
        ,
         10531675962928936483
        ,
          8408370056550069417
        ,
         16176904738380421345
        ,
         18242049240673418282
        ,
           309142400339535271
        ,
         11410818352426699816
        ,
         14847362330696741492
        ,
          1982694018844874392
        ,
         15462416137036517619
        ,
          3609228772539817828
        ,
          7753518430879613880
        ,
          2727445705515394371
        ,
          6416213867434588381
        ,
          5050360930393166868
        ,
          4963203774569068315
        ,
         16829226096394982267
        ,
          4816560883501339479
        ,
         10094672219521805256
        ,
          3477760038045501847
        ,
          4784052848707417433
        ,
          3710199748709442536
        ,
          7105828751982544235
        ,
          1170917741707525275
        ,
         14550384964284213914
        ,
          9994507326600780334
        ,
          7346796713318951517
        ,
          5811812217969210365
        ,
         18074478439909492930
        ,
         12679319564197389753
        ,
          2184803088026811554
        ,
          5676572081081773084
        ,
         14521412013665222303
        ,
         16189549618331553745
        ,
          6374414867642286672
        ,
          5087109004807834875
        ,
          3160076569473926305
        ,
          9430383795376543389
        ,
         13550102084445048726
        ,
          5166712065876557303
        ,
          4020571026917202718
        ,
         13187594882525956824
        ,
         10157391837263851216
        ,
           437569871202486056
        ,
         14004755817076531935
        ,
          5821207464352577799
        ,
           798044527018966208
        ,
           861371451701573941
        ,
          8949474274486976650
        ,
          1074784528415991342
        ,
         13342573985918661078
        ,
         15175127452028185647
        ,
         17989566484158532327
        ,
           260931675294039465
        ,
          5680207113917276363
        ,
          2856089455612565435
        ,
          6133326788964020304
        ,
         15729418347616165858
        ,
          7305612105021052126
        ,
         14820591568463104481
        ,
         16639047613571371969
        ,
          1368762713846381292
        ,
          4476340997810739031
        ,
         16634058261446704976
        ,
          8409030717808734297
        ,
         17134518147501423060
        ,
          6352352225642749163
        ,
          8121342473822317618
        ,
         14845284392754390407
        ,
         13121409786847913974
        ,
         14435895963032503405
        ,
          5311708512811707957
        ,
          6774727898930925178
        ,
          7667873836443889586
        ,
         15670102592999411412
        ,
          8721136452259161351
        ,
          7229013293622283759
        ,
          3184573509075676905
        ,
          5996054936515877189
        ,
          4755847825338355556
        ,
         11952934259811009206
        ,
          8590792479389289430
        ,
         11467750007665228233
        ,
          5349078977564905714
        ,
          9775919380935533618
        ,
         10977512204040909519
        ,
         13855624328749812189
        ,
          2655590300204180667
        ,
         15022334668391117482
        ,
          8874221973841189048
        ,
         16836523838870411184
        ,
         10392381564971722848
        ,
         13037609954087503728
        ,
         10584528579752654202
        ,
         17901972048299535391
        ,
         16382763616997159076
        ,
          8310319793341857975
        ,
          4885384215829871317
        ,
          8659217262971480692
        ,
         17766167400812025978
        ,
          7341243084835608096
        ,
          9095590878349322536
        ,
         15274176362661718913
        ,
         13420048837923366634
        ,
          1515931931605547456
        ,
         16287073113329776929
        ,
         14074562865204135585
        ,
         14768521943923005937
        ,
         11183798665273324013
        ,
          5782086880738223353
        ,
         11079354041142335616
        ,
          5289265178306913331
        ,
         17927657205069259001
        ,
          4547729806694688689
        ,
         13048598167857237322
        ,
         16224021433672679910
        ,
         10474782667963362365
        ,
          4758684226954328544
        ,
         11055896243487965354
        ,
          4926168259553998303
        ,
         17785566039793645160
        ,
          4133316158985164878
        ,
          4305758170622007501
        ,
          5949504733768296589
        ,
         10241668992609417801
        ,
         14605941936489561199
        ,
          4577680371875115518
        ,
         12339016096445776570
        ,
          2914316266117065863
        ,
         13301726820780823856
        ,
         18154724932915521000
        ,
         11732354857777551775
        ,
         12631188738778303223
        ,
         16260780708965995624
        ,
          9168475420672151970
        ,
          1306736152755902206
        ,
         17339480731497059540
        ,
         17240814577531720195
        ,
         16982484841081778880
        ,
          2093202021154522421
        ,
         10500855667561581507
        ,
           123864805529042730
        ,
          9268832303886816536
        ,
         12576480157252280466
        ,
          2857123682714846642
        ,
         13853747870222217217
        ,
         12543776373048408900
        ,
         13083956518712584850
        ,
          9285997962124025771
        ,
          3572118044337482118
        ,
         13916670211202108367
        ,
         14013455049523601069
        ,
          7384132207970395991
        ,
          6871877182889319639
        ,
          6389042588027674057
        ,
          5543831890880521542
        ,
          4511856169451192321
        ,
          9809527341931268826
        ,
         17680178702919896733
        ,
         11148508549374769988
        ,
          9494431609798654880
        ,
          6837168068006309870
        ,
         11303217681660871964
        ,
           605942987825681800
        ,
         13964859593565923850
        ,
         15709471330482124066
        ,
         16747483829525744457
        ,
         14384031638219117073
        ,
          2797840991874596234
        ,
         17121084499438328619
        ,
          4373540390356655020
        ,
         18312025190947524124
        ,
         11123513779052556976
        ,
          1712548024984304797
        ,
          1448966626644148855
        ,
         17367645762600376490
        ,
          7932699449579249902
        ,
          7865403168444276628
        ,
         14494908727152594953
        ,
         10946642521477953067
        ,
          5740418459987660151
        ,
          3910347746976295989
        ,
         14535196456382641896
        ,
         14163551088941984882
        ,
         18112709370107324862
        ,
         13928172529005088760
        ,
          4167405284126998752
        ,
         10372752136224632868
        ,
         13480920295362273351
        ,
          9312930506824927476
        ,
         15892109888632629097
        ,
         10859998884351292460
        ,
          5174625467132566184
        ,
         16784937943168171668
        ,
          6037535114760638111
        ,
          2638886380230767780
        ,
         12315482224242814952
        ,
         13916964771722788812
        ,
         10592200065614333534
        ,
         13893142310375380975
        ,
         14064368170973105781
        ,
         16750939030047482317
        ,
         10772895640557089436
        ,
          9764080371225167663
        ,
         16949858468525134857
        ,
          4592608032460181867
        ,
         18406689683428594865
        ,
          3294752992789157148
        ,
         14255834492386703793
        ,
          3805114237856518710
        ,
          3854262267948535658
        ,
          1225011063855105653
        ,
          6625838973723501328
        ,
         14622993335306995026
        ,
          5551642886146890603
        ,
          2499457011524938642
        ,
         15616208856498372384
        ,
         14671592307714693784
        ,
         14210128038124617549
        ,
           198582564198568426
        ,
          8549969985611772564
        ,
          9873134121949806534
        ,
         13634843918861849532
        ,
         13433788287152197197
        ,
          8980763513181297322
        ,
          8775713910150441535
        ,
          1171326931937929018
        ,
         13725950924040654253
        ,
          9863232693093838543
        ,
          1605374221943738154
        ,
          7935507763398630159
        ,
         10056515389375490008
        ,
          3053469238517118921
        ,
         13490864586791281671
        ,
         10080599717367424426
        ,
          6421832807039024400
        ,
         15637259813407686338
        ,
         15502537538490705250
        ,
         11865864966713876555
        ,
          1146766048007115532
        ,
          4064869759741609706
        ,
         18127339747681231887
        ,
          6719655495658402534
        ,
         17526660754412972470
        ,
         12686062255678534282
        ,
         13875749993340150225
        ,
          3164471842518796774
        ,
          7137533811878741535
        ,
         13967095885226634229
        ,
          3012254800518606678
        ,
          8784990751864317792
        ,
         13582194451643049759
        ,
          7339486078626263086
        ,
         13060950793458657360
        ,
          1364952918294765861
        ,
         10551079797287024698
        ,
          9209461205613169426
        ,
          4823732425238337321
        ,
          4505815107780408106
        ,
          7189861356125752304
        ,
          6435269021240119617
        ,
         13933384720007003522
        ,
          9433895115987887009
        ,
           866508630607016091
        ,
          7465386461012036224
        ,
         10863810974241479274
        ,
          3259386571597659805
        ,
          8546720850349060665
        ,
         16346826842699436164
        ,
         11615507801620603866
        ,
         13165986844458356746
        ,
         14237218577750731489
        ,
         11186265224627544703
        ,
          6485394181042758605
        ,
         12147917955736547212
        ,
          2269880124622275326
        ,
         16595953769128778340
        ,
          4322307501732119111
        ,
         10538565223042916886
        ,
          3622381756024060537
        ,
          7786059131836575899
        ,
         12320260104106980418
        ,
          9338299062658401865
        ,
         16165323359620492160
        ,
          4431624882370918087
        ,
          3192992299668975718
        ,
         14680708219105353575
        ,
         10339537605721421303
        ,
         13074937725104674974
        ,
          9866669591966107167
        ,
         14141828438626704816
        ,
         17968973090703035237
        ,
          5670597439799190339
        ,
          2223257555376524558
        ,
          1888442613218400001
        ,
         13499560954369253155
        ,
          7736092056942715995
        ,
          4939825603949318794
        ,
         14712770156739398201
        ,
         11079566258508416554
        ,
         17005398665653061232
        ,
          1619773083746353300
        ,
         17525078059860507067
        ,
         16665678537700126727
        ,
          9189351048016419781
        ,
          5760818127805295855
        ,
          5177472942596649872
        ,
         12774114271347078947
        ,
         15744654171892770229
        ,
         14546238959717993071
        ,
           192185467355367591
        ,
          3847814588785124598
        ,
          7749900697749843754
        ,
         14563919365170179669
        ,
          8349364528377253528
        ,
          3870143941224407618
        ,
          6702737047325064426
        ,
          8361140880464088089
        ,
          9179537785961975558
        ,
          9848295261035721008
        ,
          8453361253497413411
        ,
          1163857869940655714
        ,
          4034987707868065428
        ,
          6073799104239646916
        ,
          8796616777152251269
        ,
          6551090036624304767
        ,
          1586134799468595739
        ,
          9608778259511608195
        ,
          3687336499711393560
        ,
         12640904793448289360
        ,
          5950485614894015769
        ,
         16594772284609861465
        ,
          5323420983887109671
        ,
          7745708027685326199
        ,
         10722751848558868172
        ,
         16424567804940164743
        ,
         15721297398076084199
        ,
          9324789849986519560
        ,
         10880018890140269665
        ,
         12658328768540054401
        ,
          7949702033796708659
        ,
         18239740237294131451
        ,
          1031093361034041182
        ,
          6853738641481917831
        ,
          8176964519163878970
        ,
         10779244381852514825
        ,
          4272384940929095836
        ,
           121419440478665054
        ,
         14026067469267073076
        ,
          6927946140432483955
        ,
         10657031352810680350
        ,
         14579887559107457263
        ,
         13855216472918202419
        ,
          3656356229118440324
        ,
          8838796884797523320
        ,
         10594113154698817103
        ,
         10262694810612737642
        ,
         11932602033809281321
        ,
         15006280687636966342
        ,
         13812631369261804134
        ,
          4340129196450900339
        ,
          4720583759609565189
        ,
          8860400081746125467
        ,
         14277949063647519217
        ,
          7785713018043022968
        ,
         16851383201990870027
        ,
          9184316084228148265
        ,
          1127703496429263706
        ,
          8599445521250576365
        ,
         15313913572321885409
        ,
          3619257043488334172
        ,
          5278965250983501250
        ,
         16028260711066744146
        ,
         17787885706994543558
        ,
         11519575497367830164
        ,
         10686032745948510218
        ,
          4938159445577741484
        ,
         11822143879597220440
        ,
         18070321914448057356
        ,
           525512046650997215
        ,
          9772777286220470607
        ,
         10083512896334719283
        ,
          9272194861764302666
        ,
         17714512248498774889
        ,
          2923795349850985497
        ,
          1043726688792156981
        ,
         14895394273980335182
        ,
           784177378505685014
        ,
         16545276302725679021
        ,
          6729729425034482091
        ,
         10315996721756946380
        ,
         13941734581753498960
        ,
         13439569863439230655
        ,
          5880484523971808247
        ,
          6770289830254552025
        ,
          9969311600947323268
        ,
         13005522933640256016
        ,
         13030114176073171635
        ,
         12746593183676921879
        ,
          8750140497552320236
        ,
         10371317440169189905
        ,
         12184848077559100179
        ,
         16045784248012847570
        ,
          6562288898382443300
        ,
         17819396403062759870
        ,
         13939378459902065120
        ,
          9763566940814967463
        ,
         12617950381674896938
        ,
          7680888876419994607
        ,
          9481348792086183681
        ,
          1721743591481482057
        ,
          2371923690506404826
        ,
          5732915757184479224
        ,
          6990825371814316774
        ,
          8454994193695665608
        ,
         12363660763724107506
        ,
          6103238983185627013
        ,
         16274499173712986134
        ,
          8846013326081350976
        ,
         15564328895474535936
        ,
          7783733883078820635
        ,
          2600385237537258878
        ,
          4754571868466811070
        ,
          9424986022408199782
        ,
         11273985910887665823
        ,
         11476897301149102998
        ,
         18299978954624298818
        ,
         16589965174880180298
        ,
          6001067013538415619
        ,
         11122610152749552662
        ,
          8536899016503964986
        ,
          2535158011392890821
        ,
         13116752324695486591
        ,
          1305884909966450112
        ,
         11598000844457912254
        ,
         12509381417098744470
        ,
         16914891014772376381
        ,
         14132298473712965757
        ,
           635097142007720220
        ,
           419531556812365919
        ,
          3943835127312625401
        ,
          4312726505380646559
        ,
          9887507024882189646
        ,
          4941248286355985811
        ,
         12734133300518754259
        ,
          5149564033587641801
        ,
          3306241896983041111
        ,
          4903015998517567738
        ,
         14345851310763611453
        ,
          9542827377185555756
        ,
          5070678398713884001
        ,
         17183238706958753992
        ,
         16725656707473395594
        ,
         12738556701364840351
        ,
           572935115055318056
        ,
          6923048788321750271
        ,
         16095514998830503304
        ,
          9062308601751628393
        ,
          2394632429829176975
        ,
         13785809644907828894
        ,
         18207468648355873268
        ,
          3088779334852734070
        ,
          9658728622862829399
        ,
          8536679608353919740
        ,
          9089687459063174296
        ,
         11340228936115196659
        ,
          3720451460815338002
        ,
          3265556843602700193
        ,
          3431984470337425219
        ,
         10462131227339985973
        ,
          3280410064351034012
        ,
         10138740843906598136
        ,
          1786709675861900456
        ,
          5097754772184263945
        ,
         11909352416094175482
        ,
         15915030304466540672
        ,
         14476267102548784444
        ,
          9960621885414756445
        ,
         15196144216421545268
        ,
          2802972090107471134
        ,
          8174163671825746283
        ,
         11296031382273730546
        ,
         12513958205300792740
        ,
         16828154205167553079
        ,
         11853717740373254184
        ,
         15730308779203155365
        ,
          8910608989143341950
        ,
         16249524368453720980
        ,
          9065077803221763297
        ,
           241963234453051901
        ,
          7719062198725569867
        ,
          8403530365586778688
        ,
         15380564064215898295
        ,
         14958135729345343850
        ,
          5205080626193456490
        ,
          6345154603843446877
        ,
         11518646974342249801
        ,
         17647151513551338867
        ,
         15080236966786239719
        ,
         15908417799293803674
        ,
          4576648280555328362
        ,
         16241563950225027187
        ,
          6963333745311114312
        ,
          3778277039837416706
        ,
          2719717366513216202
        ,
         10722833851128698601
        ,
          5988443131768156187
        ,
          8831137696105748729
        ,
           766465789898503510
        ,
         14725154497919861789
        ,
          9150988774943855217
        ,
          6074963825853892736
        ,
         15729002902403104618
        ,
         12882467219207382848
        ,
         12754984354815090381
        ,
         11225242242047734504
        ,
         10798424168465647499
        ,
          7179316360652288436
        ,
          5778995630135570127
        ,
          8261092122275175015
        ,
          9542647396718852340
        ,
          6536623487761799396
        ,
         17051379792908306123
        ,
         11258247424733768612
        ,
            40759487734163912
        ,
         17240960705831497080
        ,
          6713328022970733117
        ,
          4553017813549910058
        ,
           919381004769777886
        ,
           961101082650987593
        ,
         10481558092286132554
        ,
          3779543509132877038
        ,
          1998871208666825843
        ,
           887404282035865791
        ,
         15142124729522688099
        ,
         18404364356002455406
        ,
         11433766988601147872
        ,
          9813621635363244550
        ,
         11576221149646588552
        ,
          6213230326557960980
        ,
         17841725988449936441
        ,
          5395446966934773780
        ,
           936321878533814972
        ,
          4334539794096888568
        ,
          5118627140471972748
        ,
         15900705387574170983
        ,
         16560603607120437989
        ,
          4417230530374133106
        ,
         16679733636129838940
        ,
          1949249503966654947
        ,
           421823474851375455
        ,
          3494336731282215678
        ,
           824130188654753024
        ,
          1156015840766958044
        ,
         15971562473187196549
        ,
         12596038422066662129
        ,
         10080350098781426316
        ,
          3481917570074552313
        ,
         14424906134524608347
        ,
         16025218179391157257
        ,
          2922063876841085894
        ,
         15867592155646871589
        ,
          7585326556788407770
        ,
         17354195348332167576
        ,
           482350690970371200
        ,
         15354212443750069700
        ,
         11683601846982823874
        ,
         16359887575899166026
        ,
           101578678905966328
        ,
          7317848076220395947
        ,
          6239097861955336812
        ,
         18021534336241172596
        ,
         16634180499133426203
        ,
          6005248877468772796
        ,
          7845428683878919550
        ,
          4509997884841930588
        ,
         17051128294370864912
        ,
          9813137551192922542
        ,
          5001126810155299113
        ,
          4763467318870812924
        ,
          3782136560868152748
        ,
         12192835982646788012
        ,
         16676298636680462149
        ,
         14724822773168358659
        ,
          2966276694722859223
        ,
         11222921347647975887
        ,
         14255541727682141791
        ,
         17503817252251763924
        ,
          3543695413723837226
        ,
          5871722333633274707
        ,
         12811855587809536027
        ,
            63423600273550177
        ,
         17535579500939108241
        ,
         12300449764360184310
        ,
         13885456699150823184
        ,
          6899060872085163919
        ,
         11543130859529734679
        ,
         17197464563346328689
        ,
          4874233152240144063
        ,
         14903111289535335915
        ,
          4440817171214251959
        ,
         10803770948666299735
        ,
          2236822366282851830
        ,
         16954530129623033592
        ,
          6975591640514292312
        ,
         13505191299940290049
        ,
           369796624065194135
        ,
         11050129644310425333
        ,
         13965000125886476162
        ,
          2374856914617254140
        ,
         10192647403205094217
        ,
         17249968129174993549
        ,
          6140000047801359517
        ,
          8558151783383414565
        ,
         16743921767871786545
        ,
          6333024221144946550
        ,
         13526592813254820246
        ,
         15237899509979061066
        ,
          5494892616778898326
        ,
          2891926767615218669
        ,
         11459610723367753166
        ,
         10348086339918927048
        ,
          9706807972425555468
        ,
         13828833078241526315
        ,
         16791206608829985445
        ,
         16774603868861056846
        ,
         10322787205842293324
        ,
          6267489663763486948
        ,
          1577730327244271627
        ,
          9510387962467142848
        ,
          4261353725292810170
        ,
         16041574770015853445
        ,
         17976684551069395276
        ,
          1517393044226029230
        ,
         14185917116340718878
        ,
         12464703785897941504
        ,
         13247078881508080544
        ,
         15652251703567741138
        ,
          3707023323783146127
        ,
         17528985980693523386
        ,
          4282932027682607238
        ,
         14779150576209283896
        ,
          3154648617371533130
        ,
         10675844759191345568
        ,
          7869691538452461899
        ,
         12377236332384736146
        ,
          2587784838847211820
        ,
          5066481245365401016
        ,
         11202123424644509790
        ,
          6986982115617445718
        ,
          6187405547519715471
        ,
         11117142314439944029
        ,
          4416927482560428713
        ,
          3367277986813490706
        ,
         16918688208981298959
        ,
         12736318764591298567
        ,
         16723024937524996286
        ,
         15212771632627518319
        ,
           134978276846083080
        ,
          8100828621114999667
        ,
          6533571095563022832
        ,
         16051461283381926757
        ,
         12264377884211476819
        ,
         16571868317233614377
        ,
         17003759738569944411
        ,
          7838973840932183545
        ,
          8725287762066356820
        ,
          3476006771347015236
        ,
          6750501208613132511
        ,
          4027516157367644620
        ,
          6281396839376611507
        ,
          4402616740721738721
        ,
          7719283118483718243
        ,
         10727148627683660021
        ,
          3292353224028254569
        ,
         10875883481600790540
        ,
         18257265380575945949
        ,
          9935279446831069748
        ,
         14348624058937806681
        ,
         13644871148262921637
        ,
         12758989531352982174
        ,
           178257688768267034
        ,
         10204668022759410557
        ,
          1058648063972185411
        ,
         17815897878659056792
        ,
          6607875058716390849
        ,
         14938487539436297540
        ,
          9157919681378202243
        ,
         12433091447434404604
        ,
         17317565602433554255
        ,
           561138643599324607
        ,
         10765045546217851897
        ,
         17377775477116187185
        ,
          8354164200532101711
        ,
          1539959649834169278
        ,
          9334131278298772126
        ,
         11715028291282437228
        ,
          2880406064610416100
        ,
          7070690022278313809
        ,
          1565133298667768622
        ,
         11280859053654324889
        ,
          5232431131577324120
        ,
          9163083602435077330
        ,
          8424927974053866044
        ,
         12679555061350347997
        ,
          1066550091231385933
        ,
         12569495054357661558
        ,
          3887739294702362029
        ,
          5649261317907847247
        ,
          5284007334148636919
        ,
          8731529134648854143
        ,
         16754979079001670828
        ,
         15558094259961343043
        ,
          7908116439382313521
        ,
          8343684263684860082
        ,
         11734876602617213336
        ,
          4445164268924458854
        ,
          9131563823709292633
        ,
         11515310067833604313
        ,
         13106360428667918073
        ,
          4065162567747851850
        ,
         16596767208758056332
        ,
         10217717174601847916
        ,
          1944773395032869173
        ,
         10165238163955931411
        ,
           153288546488430690
        ,
          2293983963784714754
        ,
         17877207187762042041
        ,
          1478642589166526275
        ,
         11791728277383546901
        ,
          1579422263062740838
        ,
          8241155147745863207
        ,
          7210682922403573899
        ,
         13630521446797758405
        ,
          7982600325562251469
        ,
         12702748051019610148
        ,
         15645929099304194741
        ,
          8541579400466825678
        ,
         18142014863741745868
        ,
          3284533668749651137
        ,
         13422711813276571341
        ,
          8327069727492555115
        ,
          1200984330574927056
        ,
         15406468097211014336
        ,
         11277561825722802074
        ,
         16186957528944577110
        ,
         16940923007243273701
        ,
         15794998561265364930
        ,
          9232434545250379207
        ,
           469264559397078756
        ,
         11781307585683805683
        ,
          1866536430905564027
        ,
         11371912638266431590
        ,
            65074931508771692
        ,
         11751908890246894780
        ,
         13988050972229787512
        ,
          4399321221746518700
        ,
         11753994198509506687
        ,
         14221085282380974996
        ,
          6672001604548722664
        ,
         12251765961408000641
        ,
          9222619522653919998
        ,
         16198468902194891427
        ,
          6164947588768032208
        ,
           435086123619588625
        ,
          8702592894871301143
        ,
         14883244699712087192
        ,
          5728510668047693202
        ,
          3972430926407978224
        ,
         14145439452340475639
        ,
         14610109667246184191
        ,
         12685607957293417279
        ,
          6754428385661767224
        ,
          6440375492708859693
        ,
          3230505540058267469
        ,
         15421956740431438957
        ,
         12400046994265477260
        ,
           315795028084660134
        ,
         17754322762205499956
        ,
         11933270368865922013
        ,
         16098146358435163545
        ,
          6425924207813843780
        ,
         11487255063788259443
        ,
         12599312155966469704
        ,
          4374277702265889260
        ,
          6276490797992204815
        ,
         14823029822410713873
        ,
          6971435619198480099
        ,
         14578315071387396537
        ,
         11165302210529219422
        ,
         16066386812222620772
        ,
          5483536419057606633
        ,
         17218550619232764336
        ,
          9274521906884663910
        ,
         14416673157875156583
        ,
         11250318800869991050
        ,
         17103159444197391544
        ,
          7471683200698575257
        ,
         16904023256919478659
        ,
         16942252706517280860
        ,
         15467373381370073753
        ,
         15833428143841494536
        ,
         11946620416108879652
        ,
         10127891772755777142
        ,
         14791134668707286496
        ,
         11485857858739532012
        ,
          4346216580966914123
        ,
         15846359202675235193
        ,
         16548084963925639085
        ,
         14974476156048052735
        ,
         17263187425110934022
        ,
         15582848715801103235
        ,
          9454662461767225262
        ,
          3456310729367994851
        ,
          1006066198205792868
        ,
          9785800680231189184
        ,
          1367846178596738684
        ,
         13893221313163917755
        ,
          1574457287967149365
        ,
          8210306891656417517
        ,
          4215711670323407114
        ,
           535564830273400584
        ,
         16228343553638163461
        ,
          2773671178224685271
        ,
         15107985374949245535
        ,
           130785623936754747
        ,
         13913023824907270098
        ,
          5489037057941765048
        ,
          1517302815903833090
        ,
          1476585635436678338
        ,
         18236910389268825279
        ,
         13675455377930781183
        ,
          4800987676431260720
        ,
         17557108208231389167
        ,
          5971229257863290714
        ,
         15698592894801652850
        ,
          1547579094414551765
        ,
           217434042633593815
        ,
         14170811561201432556
        ,
         13364132703307421176
        ,
         13086425902844974533
        ,
          7853643550717688762
        ,
         10832576215839351672
        ,
         14032074809027201448
        ,
          9847754822490345332
        ,
          9644757105356845998
        ,
          5121452941839006498
        ,
         10670399059193263079
        ,
         10344520197119980960
        ,
         17899926875892839050
        ,
          2567081019800060573
        ,
          5245407563912519375
        ,
         17030954097420858872
        ,
          7369854973666206691
        ,
         16136704598405241684
        ,
         10785591716995462230
        ,
          6821574599260317510
        ,
          4294804995520304105
        ,
          1965742154240588895
        ,
          9175414609168049887
        ,
          7130954873825358040
        ,
         12984166182318334519
        ,
          2364123056204651303
        ,
         14731778211917535504
        ,
         15138224500619286599
        ,
          7233047869731923902
        ,
          7921693120710418057
        ,
         16162115510204247700
        ,
           680761949124131483
        ,
         18079512011639451810
        ,
         14405621530093597304
        ,
          3249289194903038783
        ,
          4127024938581966507
        ,
          9459880391712029125
        ,
          9572720167065470341
        ,
          2438825538341020807
        ,
           131742902811919053
        ,
         10577100246679504874
        ,
           341853720515213214
        ,
         10865906632962988543
        ,
          1074218830271879702
        ,
          9665097806874466010
        ,
         10316889974734845183
        ,
          8765353080032952506
        ,
          1235568568714517784
        ,
         10926111604153620816
        ,
         13239235136827383255
        ,
          6832994703458854441
        ,
           515930456009164992
        ,
          6597608529566869397
        ,
          2956641824439925552
        ,
          1692446425874315106
        ,
         18139311805193686859
        ,
          9300366275837554950
        ,
         11420874931773149060
        ,
         11780704889872111894
        ,
         13242693001768948871
        ,
         17890435802088438991
        ,
          8528296920431752811
        ,
         11752284824788959729
        ,
          3324480817573421957
        ,
          5122522373546305000
        ,
          1075936923979774244
        ,
          2511552229700981876
        ,
         12978513175656437113
        ,
          1221606756185616820
        ,
         14455436367105515746
        ,
         13642333289363845622
        ,
         13259564119563840011
        ,
          1392406722237294824
        ,
         16358677829328545761
        ,
           388327937069720323
        ,
          2476737397391830356
        ,
           905803241437273283
        ,
        };
    // We also will simply define blocks and threadsPerBlock here rather than
    // take them as arguments. It wil be less confusing in the CPU code to
    // not require them.
    // Never raise above 1024, or we won't have enough seeds and the code will fail
    int blocks = 512;
    int local_size = 200;

    // Now that we've initialized all of the seeds, we can follow the same
    // format as most GPU code.

    // Rather than use CUDA's timing code, I am using the standard c++
    // timer code so that there is consistency between the CPU and GPU
    // timers. 

    // Allocate host memory
    unsigned long int *input = seeds;
    // We could use fewer seeds if we are generating less than 1024 numbers.
    // However, that is not the intended use-case of this PRNG. Generating 1024 random
    // integers can be done by the slowest PRNG in the CPU demo in 0.001657 seconds.
    // With that in mind, trying to use less than 1024 values will just end up using
    // the seeds themselves which come from /dev/urandom, making them acceptable.
    unsigned long int *output = new unsigned long int[targetBytes];

    // First we will check that the tests are working correctly by generating
    // a non-random set of digits.
    output_file.open("constant_rng_GPU.txt");
    cout << "Writing " << targetBytes << " equal floats to constant_rng_GPU.txt" << endl;
    output_file << "# generator constant seed = n/a\ntype: d\ncount: "<<targetBytes <<
        "\nnumbit: 64" << endl;

    begin = clock();

    // Allocate device memory
    unsigned long int *d_input;
    unsigned long int *d_output;
    cudaMalloc(&d_input, 1024 * sizeof(unsigned long int));
    cudaMalloc(&d_output, targetBytes * sizeof(unsigned long int));

    // Copy input to GPU
    cudaMemcpy(d_input, input, 1024 * sizeof(unsigned long int), 
        cudaMemcpyHostToDevice);
    // Runs Kernel
    cudaCallConstantKernel(blocks, local_size, d_input, d_output, targetBytes, 5);
    // Copies data back.
    cudaMemcpy(output, d_output, targetBytes * sizeof(unsigned long int), 
                cudaMemcpyDeviceToHost);

    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << output[i]<< "\n";
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Note. We could keep the seeds on the GPU to speed things up. However,
    // part of the slowdowns of using a GPU is the memcopying. It would be unfair
    // to the first generator if copying the data was part of the time, and only it
    // had to do the copying. In addition, saving the data to a file should be part
    // of how long it takest to run. Therefore, so should the copying of data.
    // Free GPU memory.
    cudaFree(d_input);
    cudaFree(d_output);

    // BUILTIN
    // Now, we will use cuRAND to test against the builtin random genertor.
    output_file.open("builtin_rng_GPU.txt");
    cout << "Writing " << targetBytes << " equal floats to builtin_rng_GPU.txt" << endl;
    output_file << "# generator constant seed = n/a\ntype: d\ncount: "<<targetBytes <<
        "\nnumbit: 64" << endl;

    begin = clock();
    unsigned long long int *d_output_prime;

    // Allocate device memory
    cudaMalloc(&d_output_prime, targetBytes * sizeof(unsigned long int));

    // Runs Kernel
    // We will just use cuRand API calls instead for this one
    curandGenerator_t gen;
    /* Create pseudo-random number generator */
    curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);
    
    /* Set seed */
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    /* Generate n floats on device */
    curandGenerateLongLong(gen, d_output_prime, targetBytes);

    /* Copy device memory to host */
    cudaMemcpy(output, d_output_prime, targetBytes * sizeof(unsigned long int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << output[i]<< "\n";
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Free GPU memory.
    curandDestroyGenerator(gen);
    cudaFree(d_output);

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
    
    begin = clock();

    // Allocate device memory
    cudaMalloc(&d_input, 1024 * sizeof(unsigned long int));
    cudaMalloc(&d_output, targetBytes * sizeof(unsigned long int));

    // Copy input to GPU
    cudaMemcpy(d_input, input, 1024 * sizeof(unsigned long int), 
        cudaMemcpyHostToDevice);
    // Runs Kernel
    cudaCallLCGKernel(blocks, local_size, d_input, d_output, targetBytes, a, c, m);
    // Copies data back.
    cudaMemcpy(output, d_output, targetBytes * sizeof(unsigned long int), 
                cudaMemcpyDeviceToHost);

    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << output[i]<< "\n";
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Free GPU memory.
    cudaFree(d_input);
    cudaFree(d_output);

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

    begin = clock();

    // Allocate device memory
    cudaMalloc(&d_input, 1024 * sizeof(unsigned long int));
    cudaMalloc(&d_output, targetBytes * sizeof(unsigned long int));

    // Copy input to GPU
    cudaMemcpy(d_input, input, 1024 * sizeof(unsigned long int), 
        cudaMemcpyHostToDevice);
    // Runs Kernel
    cudaCallSCGKernel(blocks, local_size, d_input, d_output, targetBytes, k);
    // Copies data back.
    cudaMemcpy(output, d_output, targetBytes * sizeof(unsigned long int), 
                cudaMemcpyDeviceToHost);

    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << output[i]<< "\n";
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Free GPU memory.
    cudaFree(d_input);
    cudaFree(d_output);

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

    begin = clock();

    // Allocate device memory
    cudaMalloc(&d_input, 1024 * sizeof(unsigned long int));
    cudaMalloc(&d_output, targetBytes * sizeof(unsigned long int));

    // Copy input to GPU
    cudaMemcpy(d_input, input, 1024 * sizeof(unsigned long int), 
        cudaMemcpyHostToDevice);
    // Runs Kernel
    cudaCallXORKernel(blocks, local_size, d_input, d_output, targetBytes, 2685821657736338717, 13, 30, 19);
    // Copies data back.
    cudaMemcpy(output, d_output, targetBytes * sizeof(unsigned long int), 
                cudaMemcpyDeviceToHost);

    for (int i = 0; i < targetBytes; ++i)
    {
        output_file << output[i]<< "\n";
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    output_file.close();
    cout << "That took " << elapsed_secs << "seconds." <<endl;

    // Free GPU memory.
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
