import re
import matplotlib.pyplot as plt
import numpy as np

# Param arrays organised as p[dirs][files][blocks][threads]

dirs = ['less_shared_mem_blocks','equal_ngroups_blocks']
files = ['single_muon_timings','pileup_46_timings','pileup_80_timings','pileup_200_timings']
pileups = [0,46,80,200]
blocks = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,7000,8000,9000,10000]
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
kt_per_block_smem_64_p1 = []
kt_per_block_smem_64_p46 = []
kt_per_block_smem_64_p80 = []
kt_per_block_smem_64_p200 = []
kt_per_block_smem_128_p1 = []
kt_per_block_smem_128_p46 = []
kt_per_block_smem_128_p80 = []
kt_per_block_smem_128_p200 = []
kt_per_block_smem_192_p1 = []
kt_per_block_smem_192_p46 = []
kt_per_block_smem_192_p80 = []
kt_per_block_smem_192_p200 = []
kt_per_block_eqgrp_64_p1 = []
kt_per_block_eqgrp_64_p46 = []
kt_per_block_eqgrp_64_p80 = []
kt_per_block_eqgrp_64_p200 = []
kt_per_block_eqgrp_128_p1 = []
kt_per_block_eqgrp_128_p46 = []
kt_per_block_eqgrp_128_p80 = []
kt_per_block_eqgrp_128_p200 = []
kt_per_block_eqgrp_192_p1 = []
kt_per_block_eqgrp_192_p46 = []
kt_per_block_eqgrp_192_p80 = []
kt_per_block_eqgrp_192_p200 = []
for b, block in enumerate(blocks):
    kt_per_block_smem_64_p1.append(kernel_time[0][0][b][0])
    kt_per_block_smem_64_p46.append(kernel_time[0][1][b][0])
    kt_per_block_smem_64_p80.append(kernel_time[0][2][b][0])
    kt_per_block_smem_64_p200.append(kernel_time[0][3][b][0])
    kt_per_block_smem_128_p1.append(kernel_time[0][0][b][1])
    kt_per_block_smem_128_p46.append(kernel_time[0][1][b][1])
    kt_per_block_smem_128_p80.append(kernel_time[0][2][b][1])
    kt_per_block_smem_128_p200.append(kernel_time[0][3][b][1])
    kt_per_block_smem_192_p1.append(kernel_time[0][0][b][2])
    kt_per_block_smem_192_p46.append(kernel_time[0][1][b][2])
    kt_per_block_smem_192_p80.append(kernel_time[0][2][b][2])
    kt_per_block_smem_192_p200.append(kernel_time[0][3][b][2])
    kt_per_block_eqgrp_64_p1.append(kernel_time[1][0][b][0])
    kt_per_block_eqgrp_64_p46.append(kernel_time[1][1][b][0])
    kt_per_block_eqgrp_64_p80.append(kernel_time[1][2][b][0])
    kt_per_block_eqgrp_64_p200.append(kernel_time[1][3][b][0])
    kt_per_block_eqgrp_128_p1.append(kernel_time[1][0][b][1])
    kt_per_block_eqgrp_128_p46.append(kernel_time[1][1][b][1])
    kt_per_block_eqgrp_128_p80.append(kernel_time[1][2][b][1])
    kt_per_block_eqgrp_128_p200.append(kernel_time[1][3][b][1])
    kt_per_block_eqgrp_192_p1.append(kernel_time[1][0][b][2])
    kt_per_block_eqgrp_192_p46.append(kernel_time[1][1][b][2])
    kt_per_block_eqgrp_192_p80.append(kernel_time[1][2][b][2])
    kt_per_block_eqgrp_192_p200.append(kernel_time[1][3][b][2])

fig0 = plt.figure()
plt.plot(blocks,1/np.array(kt_per_block_smem_64_p1),label='Single muon 64 threads',marker='o',color='b')
plt.plot(blocks,1/np.array(kt_per_block_smem_128_p1),label='Single muon 128 threads',marker='s',color='b')
#plt.plot(blocks,1/np.array(kt_per_block_smem_192_p1),label='Single muon 192 threads',marker='o')
plt.plot(blocks,1/np.array(kt_per_block_smem_64_p46),label='Pileup 46 64 threads',marker='o',color='g')
plt.plot(blocks,1/np.array(kt_per_block_smem_128_p46),label='Pileup 46 128 threads',marker='s',color='g')
#plt.plot(blocks,1/np.array(kt_per_block_smem_192_p46),label='Pileup 46 192 threads',marker='s')
plt.plot(blocks,1/np.array(kt_per_block_smem_64_p80),label='Pileup 80 64 threads',marker='o',color='r')
plt.plot(blocks,1/np.array(kt_per_block_smem_128_p80),label='Pileup 80 128 threads',marker='s',color='r')
#plt.plot(blocks,1/np.array(kt_per_block_smem_192_p80),label='Pileup 80 192 threads',marker='^')
plt.plot(blocks,1/np.array(kt_per_block_smem_64_p200),label='Pileup 200 64 threads',marker='o',color='c')
plt.plot(blocks,1/np.array(kt_per_block_smem_128_p200),label='Pileup 200 128 threads',marker='s',color='c')
#plt.plot(blocks,1/np.array(kt_per_block_smem_192_p200),label='Pileup 200 192 threads',marker='x')
#plt.plot((0,10000),(1/kernel_time[0][0][b]
#plt.legend(loc=1,prop={'size':10})
plt0 = fig0.add_subplot(111)
plt0.set_xlabel('Blocks')
plt0.set_ylabel('Average Kernel Rate (kHz)')
plt0.set_title('Average Kernel Rate For Different Numbers of CUDA Blocks',{'verticalalignment':'bottom'})
lgd = plt0.legend(loc='center left', bbox_to_anchor=(1,0.5))
fig0.savefig("eventrate_cudablocks.pdf",format='pdf',bbox_extra_artists=[lgd], bbox_inches='tight')

fig2 = plt.figure()
plt.plot(blocks,1/np.array(kt_per_block_eqgrp_64_p1),label='Single muon 64 threads',marker='o',color='b')
plt.plot(blocks,1/np.array(kt_per_block_eqgrp_128_p1),label='Single muon 128 threads',marker='s',color='b')
#plt.plot(blocks,1/np.array(kt_per_block_eqgrp_192_p1),label='Single muon 192 threads',marker='o')
plt.plot(blocks,1/np.array(kt_per_block_eqgrp_64_p46),label='Pileup 46 64 threads',marker='o',color='g')
plt.plot(blocks,1/np.array(kt_per_block_eqgrp_128_p46),label='Pileup 46 128 threads',marker='s',color='g')
#plt.plot(blocks,1/np.array(kt_per_block_eqgrp_192_p46),label='Pileup 46 192 threads',marker='s')
plt.plot(blocks,1/np.array(kt_per_block_eqgrp_64_p80),label='Pileup 80 64 threads',marker='o',color='r')
plt.plot(blocks,1/np.array(kt_per_block_eqgrp_128_p80),label='Pileup 80 128 threads',marker='s',color='r')
#plt.plot(blocks,1/np.array(kt_per_block_eqgrp_192_p80),label='Pileup 80 192 threads',marker='^')
plt.plot(blocks,1/np.array(kt_per_block_eqgrp_64_p200),label='Pileup 200 64 threads',marker='o',color='c')
plt.plot(blocks,1/np.array(kt_per_block_eqgrp_128_p200),label='Pileup 200 128 threads',marker='s',color='c')
#plt.plot(blocks,1/np.array(kt_per_block_eqgrp_192_p200),label='Pileup 200 192 threads',marker='x')
#plt.plot((0,10000),(1/kernel_time[0][0][b]
#plt.legend(loc=1,prop={'size':10})
plt2 = fig2.add_subplot(111)
plt2.set_xlabel('Blocks')
plt2.set_ylabel('Average Kernel Rate (kHz)')
plt2.set_title('Average Kernel Rate For Different Numbers of CUDA Blocks',{'verticalalignment':'bottom'})
lgd = plt2.legend(loc='center left', bbox_to_anchor=(1,0.5))
fig2.savefig("eventrate_cudablocks_eqgrps.pdf",format='pdf',bbox_extra_artists=[lgd], bbox_inches='tight')


fig1 = plt.figure()
plt.plot(blocks[1:],1/np.array(kt_per_block_smem_64_p1[1:]),label='Single muon 64 threads',marker='o',color='b')
plt.plot(blocks[1:],1/np.array(kt_per_block_smem_128_p1[1:]),label='Single muon 128 threads',marker='s',color='b',linestyle='dashed')
plt.plot((500,10000),(1/kt_per_block_smem_64_p1[0],1/kt_per_block_smem_64_p1[0]),'b')
plt.plot((500,10000),(1/kt_per_block_smem_128_p1[0],1/kt_per_block_smem_128_p1[0]),'b',linestyle='dashed')
#plt.plot(blocks,1/np.array(kt_per_block_smem_192_p1),label='Single muon 192 threads',marker='o')
plt.plot(blocks[1:],1/np.array(kt_per_block_smem_64_p46[1:]),label='Pileup 46 64 threads',marker='o',color='g')
plt.plot(blocks[1:],1/np.array(kt_per_block_smem_128_p46[1:]),label='Pileup 46 128 threads',marker='s',color='g',linestyle='dashed')
plt.plot((500,10000),(1/kt_per_block_smem_64_p46[0],1/kt_per_block_smem_64_p46[0]),'g')
plt.plot((500,10000),(1/kt_per_block_smem_128_p46[0],1/kt_per_block_smem_128_p46[0]),'g',linestyle='dashed')
#plt.plot(blocks,1/np.array(kt_per_block_smem_192_p46),label='Pileup 46 192 threads',marker='s')
plt.plot(blocks[1:],1/np.array(kt_per_block_smem_64_p80[1:]),label='Pileup 80 64 threads',marker='o',color='r')
plt.plot(blocks[1:],1/np.array(kt_per_block_smem_128_p80[1:]),label='Pileup 80 128 threads',marker='s',color='r',linestyle='dashed')
plt.plot((500,10000),(1/kt_per_block_smem_64_p80[0],1/kt_per_block_smem_64_p80[0]),'r')
plt.plot((500,10000),(1/kt_per_block_smem_128_p80[0],1/kt_per_block_smem_128_p80[0]),'r',linestyle='dashed')
#plt.plot(blocks,1/np.array(kt_per_block_smem_192_p80),label='Pileup 80 192 threads',marker='^')
plt.plot(blocks[1:],1/np.array(kt_per_block_smem_64_p200[1:]),label='Pileup 200 64 threads',marker='o',color='c')
plt.plot(blocks[1:],1/np.array(kt_per_block_smem_128_p200[1:]),label='Pileup 200 128 threads',marker='s',color='c',linestyle='dashed')
plt.plot((500,10000),(1/kt_per_block_smem_64_p200[0],1/kt_per_block_smem_64_p200[0]),'c')
plt.plot((500,10000),(1/kt_per_block_smem_128_p200[0],1/kt_per_block_smem_128_p200[0]),'c',linestyle='dashed')
#plt.plot(blocks,1/np.array(kt_per_block_smem_192_p200),label='Pileup 200 192 threads',marker='x')
#plt.plot((0,10000),(1/kernel_time[0][0][b]
#plt.legend(loc='center left',prop={'size':10},bbox_to_anchor=(1,0.5))
plt1 = fig1.add_subplot(111)
plt1.set_xlabel('Blocks')
plt1.set_xlim([500,10000])
plt1.set_ylabel('Average Kernel Rate (kHz)')
plt1.set_title('Average Kernel Rate For Different Numbers of CUDA Blocks',{'verticalalignment':'bottom'})
lgd = plt1.legend(loc='center left', bbox_to_anchor=(1,0.5))
#fig1.savefig("eventrate_cudablocks_lines.png",bbox_extra_artists(lgd,), bbox_inches='tight')
fig1.savefig("eventrate_cudablocks_lines.pdf",format='pdf',bbox_extra_artists=[lgd], bbox_inches='tight')

