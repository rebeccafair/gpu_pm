import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

blocks = [500,1000,2500,5000,10000]

patts = [[] for i in range(len(blocks))]
grps = [[] for i in range(len(blocks))]

for b, block in enumerate(blocks):
    with open('patts_' + str(block)) as patt_f:
        patts[b] = [int(x.strip('\n')) for x in patt_f.readlines()]
    with open('grps_' + str(block)) as grp_f:
        grps[b] = [int(x.strip('\n')) for x in grp_f.readlines()]

patts_in_grp = []
with open('patt_in_grp') as f:
    patts_in_grp = [int(x.strip('\n')) for x in f.readlines()]

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

fig2 = plt.figure()
plt.hist(patts_in_grp,bins=max(patts_in_grp))
#plt.hist(patts_in_grp,bins=100)
#plt.hist(patts_in_grp,bins=[0,1,2,3,4,5,10,50,100,250,500,1000,1500,2000])
plt.xlabel('Patterns In Group')
plt.ylabel('N Groups')
plt.xlim(0,100)
#plt.yscale('log', nonposy='clip')
plt.title('Distribution of Patterns to Groups')
fig2.savefig('patts_in_grps.png')
