import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

blocks = [2500]

patts = [[] for i in range(len(blocks))]
grps = [[] for i in range(len(blocks))]

for b, block in enumerate(blocks):
    with open('eqgrp_patts_' + str(block)) as patt_f:
        patts[b] = [int(x.strip('\n')) for x in patt_f.readlines()]
    with open('eqgrp_grps_' + str(block)) as grp_f:
        grps[b] = [int(x.strip('\n')) for x in grp_f.readlines()]

#print patts
#print grps
for b, block in enumerate(blocks):

    fig0 = plt.figure()
    plt.hist(patts[b],bins=100)
    plt.xlabel('Patterns')
    plt.ylabel('N Blocks')
    plt.title('Distribution of Patterns to ' + str(block) + ' CUDA Blocks')
    fig0.savefig('patts_' + str(block) + '.png')

    fig1 = plt.figure()
    #plt.hist(grps[b],bins=np.arange(0,max(grps[b])+2,1))
    plt.hist(grps[b],bins=range(0,max(grps[b])+2,1))
    plt.xlabel('Groups')
    plt.xlim(xmin=1,xmax=max(grps[b])+1)
    plt.ylabel('N Blocks')
    plt.title('Distribution of Groups to ' + str(block) + ' CUDA Blocks')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(min_n_ticks=0,integer=True))
    fig1.savefig('grps_' + str(block) + '.png')

