Analysis you should perform
--------------------------------------------------------------------------------
Write all analysis in README.txt.

How much does IO and parsing dominate your runtime? Compute this by commenting
out all of the CUDA code and just running the loop over all of the data. You
might have to use the data somehow (like writing it to a buffer with
readLSAReview) to prevent the compiler from optimizing out your loop. Once you
have program runtime with logistic regression and program runtime just from
IO and parsing, you can compute the fraction of the time spent doing IO and
parsing.

It takes 16.12 seconds to run the code normally. 
With the kernel commented out (the call to cudaClassify), the program
takes 15.68 seconds. (16.12-15.68)/(16.12) = .027
The fraction doing IO/parsing is 97%
However, note that these times flucuate by about .4 seconds, so it might
not be very accurate of an estimate.

What is the latency and throughput of your kernel for different batch sizes?
For batch sizes 1, 32, 1024, 2048, 16384, and 65536 record the amount of time
it takes your kernel to run (the latency) and compute the throughput
(batch_size / latency). You should only measure the latency of the kernel
itself, not the copying of data onto and off of the device.

We know that the IO/parsing takes 15.68 seconds. Thus, if we subtract 
from that we get the kernel time.
1    : time limit enforced
32   :16.27s
1024 :15.49s
2048 :16.12s
16384:15.44s
65536: segfault (Due to how shared memory works, this is too large to handle)

Note that the IO/parsing sohuld be almost exactly the same timewise depending
on the batchsize, so we can just subtract. The fact that some values are negative
indiciates that the kernel is so fast that the fluctuations in speed from heat
and other processes dominate the results.

The file has 1,569,264 rows, so that means that we can make another chart
to determine the time per batch.
batchsize: total time: num_batches : kernel time : time per batch
1    :n/a     :  n/a : n/a  : n/a
32   :16.27s  :49039 : .58s : 1.1e-5s
1024 :15.49s  : 1532 :-.18s : -.00017s
2048 :16.12s  :  766 : .44s : .0005s
16384:15.44s  :   95 :-.24s :-.002s  
65536:n/a     :   23 : n/a  : n/a

Obviously some of this data is wrong because of inacuracy in the measurements.