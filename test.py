import pickle
import numpy as np
import matplotlib.pyplot as plt

dirpath = '/Users/gongqili/recordings/all/loop100_2020-03-02_10:40:10'

loop = 15

# file4 = open(dirpath + '/ULVR_ACtest_res.txt', 'rb')
# file5 = open(dirpath + '/RR_ACtest_res.txt', 'rb')
# file6 = open(dirpath + '/RDSa_ACtest_res.txt', 'rb')
# ULVR_AC_res = pickle.load(file4, encoding='iso-8859-1')
# RR_AC_res = pickle.load(file5, encoding='iso-8859-1')
# RDSa_AC_res = pickle.load(file6, encoding='iso-8859-1')
# for i in range(len(ULVR_AC_res)):
#     ULVR_AC_sum.append(np.sum(ULVR_AC_res[:i]))

tickers = ['ULVR', 'RR', 'RDSa']
for loop in np.arange(23, 24):
    res = []
    dones = []
    for ticker in tickers:
        file = open(dirpath + '/loop{}_{}_test_res.txt'.format(loop, ticker), 'rb')
        re = pickle.load(file, encoding='iso-8859-1')
        res.append(re)
        file = open(dirpath + '/loop{}_{}_test_dones.txt'.format(loop, ticker), 'rb')
        done = pickle.load(file, encoding='iso-8859-1')
        dones.append(done)

    ep_res = []
    for i in range(len(tickers)):
        for indx in range(len(dones[i])):
            if indx == 0:
                ep_re = res[i][ : dones[i][indx] + 1]
                while len(ep_re) < 24:
                    ep_re.append(0)
                ep_res.append(ep_re)
            else:
                ep_re = res[i][dones[i][indx - 1] + 1 : dones[i][indx]]
                while len(ep_re) < 24:
                    ep_re.append(0)
                ep_res.append(ep_re)

    ep_res = list(np.array(ep_res).flatten())
    accumulated_res = []
    for i in range(len(ep_res)):
        accumulated_res.append(np.sum(ep_res[:i]))
    plt.plot(accumulated_res)
    plt.title('Accumulated Profit over Hothead Agent')
    plt.xlabel('Trading Steps')
    plt.ylabel('Profit')
    plt.show()