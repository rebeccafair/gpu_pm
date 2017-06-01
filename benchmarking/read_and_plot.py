import re
import matplotlib.pyplot as plt
import numpy as np

# Param arrays organised as p[dirs][files][blocks][threads]

dirs = ['no_bitarray','before_shared_bitarray','before_lyrs','match_by_layer','less_shared_mem','equal_ngroups']
files = ['single_muon_timings','pileup_46_timings','pileup_80_timings','pileup_200_timings']
pileups = [0,46,80,200]
blocks = [0,500,1000,2500,5000,10000]
threads = [64,128,192,256]

def create_nested_list():
    lst = [[[[] for k in range(len(blocks))] for j in range(len(files))] for i in range(len(dirs))]
    return lst

def get_bitarr_time(data):
    regex = 'Average bit array creation time is (\d+\.\d+) ms'
    matches = re.findall(regex,data)
    if len(matches) < len(blocks)*len(threads):
        for i in range(len(blocks)*len(threads) - len(matches)):
            matches.append('0')
    return matches

def get_kernel_time_ms(data):
    regex_times = '\s+\d+\.\d{2}%\s+\d+\.\d+\ws\s+1000?\s+(\d+\.\d+)\ws\s+\d+\.\d+\ws\s+\d+\.\d+\ws\s+matchByBlock'
    regex_units = '\s+\d+\.\d{2}%\s+\d+\.\d+\ws\s+1000?\s+\d+\.\d+((?:u|m)?s)\s+\d+\.\d+\ws\s+\d+\.\d+\ws\s+matchByBlock'
    times = re.findall(regex_times,data)
    units = re.findall(regex_units,data)
    for u, unit in enumerate(units):
        if unit == 'us':
            times[u] = float(times[u])/1000 
        elif unit == 's':
            times[u] = float(times[u])*1000
    return times

def read_append_data(d,f,param,param_method,data):
    matches = param_method(data);
    match_index = 0;
    for b, block in enumerate(blocks):
        for t, thread in enumerate(threads):
            param[d][f][b].append(float(matches[match_index]))
            match_index += 1

kernel_time = create_nested_list()
bitarr_create_time = create_nested_list()

for d, directory in enumerate(dirs):
    for f, filename in enumerate(files):
        with open(directory + '/' + filename, 'r') as fl:
            print 'Reading file ' + directory + '/' + filename
            data = fl.read()
            read_append_data(d,f,kernel_time,get_kernel_time_ms,data)
            read_append_data(d,f,bitarr_create_time,get_bitarr_time,data)

#print kernel_time
kt_per_pileup_nobit_64 = []
kt_per_pileup_nobit_128 = []
kt_per_pileup_nobit_192 = []
kt_per_pileup_nobit_256 = []
kt_per_pileup_bit_64 = []
kt_per_pileup_bit_128 = []
kt_per_pileup_bit_192 = []
kt_per_pileup_bit_256 = []
bitarr_per_pileup_128 = []
kt_per_pileup_shared_64 = []
kt_per_pileup_shared_128 = []
kt_per_pileup_shared_192 = []
kt_per_pileup_shared_256 = []
kt_per_pileup_lyr_64 = []
kt_per_pileup_lyr_128 = []
kt_per_pileup_lyr_192 = []
kt_per_pileup_lyr_256 = []
kt_per_pileup_lyr_500_64 = []
kt_per_pileup_lyr_500_128 = []
kt_per_pileup_lyr_500_192 = []
kt_per_pileup_lyr_500_256 = []
kt_per_pileup_lyr_1000_64 = []
kt_per_pileup_lyr_1000_128 = []
kt_per_pileup_lyr_1000_192 = []
kt_per_pileup_lyr_1000_256 = []
kt_per_pileup_lyr_2500_64 = []
kt_per_pileup_lyr_2500_128 = []
kt_per_pileup_lyr_2500_192 = []
kt_per_pileup_lyr_2500_256 = []
kt_per_pileup_lyr_5000_64 = []
kt_per_pileup_lyr_5000_128 = []
kt_per_pileup_lyr_5000_192 = []
kt_per_pileup_lyr_5000_256 = []
kt_per_pileup_lyr_10000_64 = []
kt_per_pileup_lyr_10000_128 = []
kt_per_pileup_lyr_10000_192 = []
kt_per_pileup_lyr_10000_256 = []
kt_per_pileup_smem_2500_64 = []
kt_per_pileup_smem_2500_128 = []
kt_per_pileup_smem_2500_192 = []
kt_per_pileup_smem_2500_256 = []
kt_per_pileup_eqgrp_2500_64 = []
kt_per_pileup_eqgrp_2500_128 = []
kt_per_pileup_eqgrp_2500_192 = []
kt_per_pileup_eqgrp_2500_256 = []
for f, filename in enumerate(files):
    kt_per_pileup_nobit_64.append(kernel_time[0][f][0][0])
    kt_per_pileup_nobit_128.append(kernel_time[0][f][0][1])
    kt_per_pileup_nobit_192.append(kernel_time[0][f][0][2])
    kt_per_pileup_nobit_256.append(kernel_time[0][f][0][3])
    kt_per_pileup_bit_64.append(kernel_time[1][f][0][0])
    kt_per_pileup_bit_128.append(kernel_time[1][f][0][1])
    kt_per_pileup_bit_192.append(kernel_time[1][f][0][2])
    kt_per_pileup_bit_256.append(kernel_time[1][f][0][3])
    bitarr_per_pileup_128.append(bitarr_create_time[1][f][0][1])
    kt_per_pileup_shared_64.append(kernel_time[2][f][0][0])
    kt_per_pileup_shared_128.append(kernel_time[2][f][0][1])
    kt_per_pileup_shared_192.append(kernel_time[2][f][0][2])
    kt_per_pileup_shared_256.append(kernel_time[2][f][0][3])
    kt_per_pileup_lyr_64.append(kernel_time[3][f][0][0])
    kt_per_pileup_lyr_128.append(kernel_time[3][f][0][1])
    kt_per_pileup_lyr_192.append(kernel_time[3][f][0][2])
    kt_per_pileup_lyr_256.append(kernel_time[3][f][0][3])
    kt_per_pileup_lyr_500_64.append(kernel_time[3][f][1][0])
    kt_per_pileup_lyr_500_128.append(kernel_time[3][f][1][1])
    kt_per_pileup_lyr_500_192.append(kernel_time[3][f][1][2])
    kt_per_pileup_lyr_500_256.append(kernel_time[3][f][1][3])
    kt_per_pileup_lyr_1000_64.append(kernel_time[3][f][2][0])
    kt_per_pileup_lyr_1000_128.append(kernel_time[3][f][2][1])
    kt_per_pileup_lyr_1000_192.append(kernel_time[3][f][2][2])
    kt_per_pileup_lyr_1000_256.append(kernel_time[3][f][2][3])
    kt_per_pileup_lyr_2500_64.append(kernel_time[3][f][3][0])
    kt_per_pileup_lyr_2500_128.append(kernel_time[3][f][3][1])
    kt_per_pileup_lyr_2500_192.append(kernel_time[3][f][3][2])
    kt_per_pileup_lyr_2500_256.append(kernel_time[3][f][3][3])
    kt_per_pileup_lyr_5000_64.append(kernel_time[3][f][4][0])
    kt_per_pileup_lyr_5000_128.append(kernel_time[3][f][4][1])
    kt_per_pileup_lyr_5000_192.append(kernel_time[3][f][4][2])
    kt_per_pileup_lyr_5000_256.append(kernel_time[3][f][4][3])
    kt_per_pileup_lyr_10000_64.append(kernel_time[3][f][5][0])
    kt_per_pileup_lyr_10000_128.append(kernel_time[3][f][5][1])
    kt_per_pileup_lyr_10000_192.append(kernel_time[3][f][5][2])
    kt_per_pileup_lyr_10000_256.append(kernel_time[3][f][5][3])
    kt_per_pileup_smem_2500_64.append(kernel_time[4][f][3][0])
    kt_per_pileup_smem_2500_128.append(kernel_time[4][f][3][1])
    kt_per_pileup_smem_2500_192.append(kernel_time[4][f][3][2])
    kt_per_pileup_smem_2500_256.append(kernel_time[4][f][3][3])
    kt_per_pileup_eqgrp_2500_64.append(kernel_time[5][f][3][0])
    kt_per_pileup_eqgrp_2500_128.append(kernel_time[5][f][3][1])
    kt_per_pileup_eqgrp_2500_192.append(kernel_time[5][f][3][2])
    kt_per_pileup_eqgrp_2500_256.append(kernel_time[5][f][3][3])

fig0 = plt.figure()
plt.plot(pileups,1/np.array(kt_per_pileup_nobit_64),label='64 threads',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_nobit_128),label='128 threads',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_nobit_192),label='192 threads',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_nobit_256),label='256 threads',marker='s')
plt.legend(loc=1,prop={'size':10})
plt0 = fig0.add_subplot(111)
plt0.set_xlabel('Pileup')
plt0.set_ylabel('Average Kernel Rate (kHz)')
plt0.set_yscale('log')
plt0.set_title('Initial Average Event Rate For Different Thread Numbers',{'verticalalignment':'bottom'})
fig0.savefig("eventrate_nobitarray_allt.png")

fig1 = plt.figure()
plt.plot(pileups,kt_per_pileup_nobit_64,label='64 threads',marker='s')
plt.plot(pileups,kt_per_pileup_nobit_128,label='128 threads',marker='s')
plt.plot(pileups,kt_per_pileup_nobit_192,label='192 threads',marker='s')
plt.plot(pileups,kt_per_pileup_nobit_256,label='256 threads',marker='s')
plt.legend(loc=2,prop={'size':10})
plt1 = fig1.add_subplot(111)
plt1.set_xlabel('Pileup')
plt1.set_ylabel('Average Kernel Time (ms)')
plt1.set_title('Initial Average Matching Kernel Time For Different Thread Numbers',{'verticalalignment':'bottom'})
#plt1.set_yscale('log')
fig1.savefig("kernel_time_nobitarray_allt.png")


fig2 = plt.figure()
plt.plot(pileups,kt_per_pileup_nobit_64,label='64 threads',marker='s')
plt.plot(pileups,kt_per_pileup_bit_64,label='64 threads with bit array',marker='s')
plt.plot(pileups,kt_per_pileup_bit_128,label='128 threads with bit array',marker='s')
plt.plot(pileups,kt_per_pileup_bit_192,label='192 threads with bit array',marker='s')
plt.plot(pileups,kt_per_pileup_bit_256,label='256 threads with bit array',marker='s')
plt.legend(loc=2,prop={'size':10})
plt2 = fig2.add_subplot(111)
plt2.set_xlabel('Pileup')
plt2.set_ylabel('Average Kernel Time (ms)')
plt2.set_yscale('log')
plt2.set_title('Comparison of Average Matching Kernel Time With and Without Bit Arrays',{'verticalalignment':'bottom'})
fig2.savefig("kt_compare_bitarray.png")

fig5 = plt.figure()
plt.plot(pileups,1/np.array(kt_per_pileup_nobit_64),label='64 threads no bit array',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_bit_64),label='64 threads with bit array',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_bit_128),label='128 threads with bit array',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_bit_192),label='192 threads with bit array',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_bit_256),label='256 threads with bit array',marker='s')
plt.legend(loc=1,prop={'size':10})
plt5 = fig5.add_subplot(111)
plt5.set_xlabel('Pileup')
plt5.set_ylabel('Average Kernel Rate (kHz)')
plt5.set_yscale('log')
plt5.set_title('Comparison of Average Kernel Rate With and Without Bit Arrays',{'verticalalignment':'bottom'})
fig5.savefig("eventrate_compare_bitarray.png")

fig3 = plt.figure()
plt.plot(pileups,bitarr_per_pileup_128,label='CPU bit array creation',marker='s')
plt.plot(pileups,np.array(bitarr_per_pileup_128) + np.array(kt_per_pileup_bit_128),label='CPU bit array + GPU matching 128 threads',marker='s')
plt.legend(loc=2,prop={'size':10})
plt3 = fig3.add_subplot(111)
plt3.set_xlabel('Pileup')
plt3.set_ylabel('Average Time (ms)')
plt3.set_title('Stacked Bit Array Creation and Matching Time',{'verticalalignment':'bottom'})
fig3.savefig("stacked_bitarray.png")

fig4 = plt.figure()
plt.plot(pileups,bitarr_per_pileup_128,label='CPU bit array creation',marker='s')
plt.plot(pileups,np.array(bitarr_per_pileup_128) + np.array(kt_per_pileup_bit_128),label='CPU bit array + GPU matching 128 threads',marker='s')
plt.legend(loc=2,prop={'size':10})
plt3 = fig3.add_subplot(111)
plt3.set_xlabel('Pileup')
plt3.set_ylabel('Average Time (ms)')
plt3.set_title('Stacked Bit Array Creation and Matching Time',{'verticalalignment':'bottom'})
fig3.savefig("stacked_bitarray.png")

fig6 = plt.figure()
plt.plot(pileups,1/np.array(kt_per_pileup_bit_128),label='128 threads global bit array',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_shared_64),label='64 threads shared bit array',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_shared_128),label='128 threads shared bit array',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_shared_192),label='192 threads shared bit array',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_shared_256),label='256 threads shared bit array',marker='s')
plt.legend(loc=1,prop={'size':10})
plt6 = fig6.add_subplot(111)
plt6.set_xlabel('Pileup')
plt6.set_ylabel('Average Kernel Rate (kHz)')
#plt5.set_yscale('log')
plt6.set_title('Comparison of Average Kernel Rate With Global and Shared Bit Arrays',{'verticalalignment':'bottom'})
fig6.savefig("eventrate_compare_shared.png")

fig7 = plt.figure()
plt.plot(pileups,1/np.array(kt_per_pileup_shared_128),label='128 threads shared bit array',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_64),label='64 threads 1 th/patt',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_128),label='128 threads 1 th/patt',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_192),label='192 threads 1 th/patt',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_256),label='256 threads 1 th/patt',marker='s')
plt.legend(loc=1,prop={'size':10})
plt7 = fig7.add_subplot(111)
plt7.set_xlabel('Pileup')
plt7.set_ylabel('Average Kernel Rate (kHz)')
#plt5.set_yscale('log')
plt7.set_title('Comparison of Average Kernel Rate With 1 or 8 threads per pattern',{'verticalalignment':'bottom'})
fig7.savefig("eventrate_compare_lyr.png")

fig8 = plt.figure()
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_64),label='64 threads',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_128),label='128 threads',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_192),label='192 threads',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_256),label='256 threads',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_500_64),label='64 threads, 500 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_500_128),label='128 threads, 500 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_500_192),label='192 threads, 500 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_500_256),label='256 threads, 500 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_1000_64),label='64 threads, 1000 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_1000_128),label='128 threads, 1000 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_1000_192),label='192 threads, 1000 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_1000_256),label='256 threads, 1000 blocks',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_2500_64),label='64 threads, 2500 blocks',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_2500_128),label='128 threads, 2500 blocks',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_2500_192),label='192 threads, 2500 blocks',marker='s')
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_2500_256),label='256 threads, 2500 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_5000_64),label='64 threads, 5000 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_5000_128),label='128 threads, 5000 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_5000_192),label='192 threads, 5000 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_5000_256),label='256 threads, 5000 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_10000_64),label='64 threads, 10000 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_10000_128),label='128 threads, 10000 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_10000_192),label='192 threads, 10000 blocks',marker='s')
#plt.plot(pileups,1/np.array(kt_per_pileup_lyr_10000_256),label='256 threads, 10000 blocks',marker='s')
plt.legend(loc=1,prop={'size':10})
plt8 = fig8.add_subplot(111)
plt8.set_xlabel('Pileup')
plt8.set_ylabel('Average Kernel Rate (kHz)')
#plt5.set_yscale('log')
plt8.set_title('Comparison of Average Kernel Rate For Different Threads/Blocks',{'verticalalignment':'bottom'})
fig8.savefig("eventrate_lyr_thblock.png")

fig9 = plt.figure()
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_64),label='64 threads, 1 grp/block',marker='o')
plt.plot(pileups,1/np.array(kt_per_pileup_lyr_2500_128),label='128 threads, 2500 blocks, high smem',marker='s',color='r')
plt.plot(pileups,1/np.array(kt_per_pileup_smem_2500_64),label='64 threads, 2500 blocks low smem',marker='o',color='g')
plt.plot(pileups,1/np.array(kt_per_pileup_smem_2500_128),label='128 threads, 2500 blocks low smem',marker='s',color='g')
plt.legend(loc=1,prop={'size':10})
plt9 = fig9.add_subplot(111)
plt9.set_xlabel('Pileup')
plt9.set_ylabel('Average Kernel Rate (kHz)')
#plt5.set_yscale('log')
plt9.set_title('Comparison of Average Kernel Rate For Different Threads/Blocks',{'verticalalignment':'bottom'})
fig9.savefig("eventrate_smem_thblock.png")

fig10 = plt.figure()
plt.plot(pileups,1/np.array(kt_per_pileup_smem_2500_64),label='64 threads, 2500 blocks',marker='o',color='g')
plt.plot(pileups,1/np.array(kt_per_pileup_smem_2500_128),label='128 threads, 2500 blocks',marker='s',color='g')
plt.plot(pileups,1/np.array(kt_per_pileup_eqgrp_2500_64),label='64 threads, 2500 blocks equal groups',marker='o',color='c')
plt.plot(pileups,1/np.array(kt_per_pileup_eqgrp_2500_128),label='128 threads, 2500 blocks equal groups',marker='s',color='c')
plt.legend(loc=1,prop={'size':10})
plt10 = fig10.add_subplot(111)
plt10.set_xlabel('Pileup')
plt10.set_ylabel('Average Kernel Rate (kHz)')
#plt5.set_yscale('log')
plt10.set_title('Comparison of Average Kernel Rate For Different Threads/Blocks',{'verticalalignment':'bottom'})
fig10.savefig("eventrate_eqgrp_thblock.png")

