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

What is the latency and throughput of your kernel for different batch sizes?
For batch sizes 1, 32, 1024, 2048, 16384, and 65536 record the amount of time
it takes your kernel to run (the latency) and compute the throughput
(batch_size / latency). You should only measure the latency of the kernel
itself, not the copying of data onto and off of the device.