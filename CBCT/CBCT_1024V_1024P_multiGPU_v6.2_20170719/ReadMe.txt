average time cost per iteration (after 20 iters) and the forward timing consuming

aristotle     host2P v1        P2P

single      62.5 (17.1)       61.6 (17.0)
1 GPU       83.1 (17.4)       62.2 (17.0)
2 GPU       53.3 (9.2)        61.6 (8.93)
3 GPU       45.9 (7.3)
4 GPU       42.0 (5.6)     


radon         host2P v1        P2P

single      59.9 (12.0)     59.5 (11.9)
1GPU        69.7 (12.3)     60.0 (12.2)
2GPU        41.7 (6.39)     44.5 (6.34)
3GPU        35.4 (5.48)


=====================================


aristotle     host2P v1.0         v1.1                 v2.1                 v 3.0                   v5.1              v5.2 (cc3.5+fastMath)

single      61.6 (17.1+41.2)    62.0 (17.1+41.2)     61.9 (17.1+41.2)      61.9 (17.1+41.2)     53.18 (16.8+34.1)      54.8 (17.3+35.4)
1 GPU       81.6 (17.4+61.2)    82.4 (17.4+61.0)     65.3 (17.4+42.2)      64.7 (17.4+42.3)         N/A                     N/A
2 GPU       53.9 (8.97+41.3)    53.7 (8.97+40.9)     44.5 (8.97+30.7)      34.7 (8.97+22.4)     29.74 (9.01+18.9)      30.8 (9.03+20.2)
3 GPU       47.6 (7.08+36.6)*   47.8 (7.08+36.5)*    42.1 (7.08+31.3)*     26.3 (7.08+16.2)     22.85 (7.09+15.1)      23.8 (7.10+15.3)
4 GPU       42.2 (5.93+32.7)*   43.3 (5.92+32.8)*    39.0 (5.93+29.3)*     21.2 (5.41+13.1)     18.27 (5.45+12.0)      18.9 (5.48+12.1)


radon         host2P v1.0         v1.1                  v2.1                v 3.0                   v5.1              v5.2 (cc3.5+fastMath)

single      61.2 (12.2+46.5)     61.7 (12.3+46.5)     61.3 (12.2+46.4)     64.1 (13.7+48.3)     59.85 (12.2+47.0)      37.6 (12.2+24.9)
1 GPU       70.6 (12.3+55.6)     70.4 (12.3+55.5)     61.8 (12.3+46.5)         N/A                 N/A                     N/A
2 GPU       42.3 (6.39+33.2)     42.4 (6.36+33.3)     39.0 (6.37+29.8)     35.1 (7.06+25.4)     30.67 (6.39+23.5)      20.3 (6.31+13.4)
3 GPU       34.8 (5.12+26.6)     35.7 (5.04+26.6)     32.6 (5.04+24.6)     25.2 (5.46+17.7)     22.23 (5.07+16.6)      15.5 (5.01+9.70)
4 GPU               N/A                 N/A                 N/A                N/A                 N/A                     N/A

host2P

v1.0
1.copy complete h_volume to each d_volume[i] on GPUs; (note: all d_volume[i] are the same)
2.parallelly generate partial projection d_proj_partial[i] based on d_volume[i];
3.transfer d_proj_partial[i] to host and stack them into h_proj;
4.copy complete h_proj to each d_proj[i] on GPUs;
5.parallel backprojection for each part of h_volume from d_proj[i] (note: all d_proj[i] are the same)
6.repeat 1

The bottle neck is the step 1. Transfer from host to device with a complete volume is time comsuming.

v1.1
change all the function interfaces of v1.0 to (page-locked) host memory pass rather than device memory pass

v2.0
1.set up d_volume[i], each of them are the whole size but only has a portion of non-zeros values;
2.parallelly generate porjection d_proj[i] based on d_volume[i]; 
3.transfer d_proj[i] to host and add them into h_proj; (not really a simple add)
4.coply h_proj into device d_proj[i] on GPUs;
5.parallel backprojection for the non-zeros part of each d_volume[i] from d_proj[i] (note: all d_proj[i] are the same)
6.repeat 2

The bottle neck is the step 2. It requires the forward process for a complete projection view from a complete volume. It is not parallelized at all.

v2.1
change all the function interfaces of v2.0 to (page-locked) host memory pass rather than device memory pass

v3.0
1.set up d_volume[i], each of them are the partial size;
2.parallelly generate porjection d_proj[i] based on d_volume[i]; 
3.transfer d_proj[i] to host and add them into h_proj; (not really a simple add)
4.coply h_proj into device d_proj[i] on GPUs;
5.parallel backprojection for the non-zeros part of each d_volume[i] from d_proj[i] (note: all d_proj[i] are the same)
6.repeat 2

The step 2 and step 5 are both parallelized, and there is not big volume transfer between host and device.

v4.0
The CBCT/FBCT/PBCT coherently share the same forward / backprojection code in multi-GPU computing.

v5.0
The CBCT/FBCT/PBCT coherently share the same forward / backprojection code in single-GPU computing.

v5.1
OpenMP is applied to concurrently do the file saving and cost function calculations.

v5.2
1.update pixel-driven backprojection code
2.refine all the double variables to float (single precision)
3.enable fast math for cc2.0 and above

v6.0
1.add FDK reconstruction
2.free unused GPUs when running
3.optimize the execution configuration for pixel-driven backprojection

v6.1
1.udpate the constant definition and "include"
2.bug fixed: should cudaMemset zero for d_proj_sumLen_addr[i] and d_proj_weightedLen_addr
3.bug fixed: TV norm calculation now corrected for 2048V

v6.2
1.bug fixed: FDK rotation orientation
2.experimental dataset tested