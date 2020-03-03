import pickle
import numpy as np
import matplotlib.pyplot as plt

dirpath = '/Users/gongqili/recordings/all/loop100_2020-03-02_10:40:10'
AC_dirpath = '/Users/gongqili/recordings/all/loop100_2020-03-03_00:50:12'

loop = 15

# file4 = open(dirpath + '/ULVR_ACtest_res.txt', 'rb')
# file5 = open(dirpath + '/RR_ACtest_res.txt', 'rb')
# file6 = open(dirpath + '/RDSa_ACtest_res.txt', 'rb')
# ULVR_AC_res = pickle.load(file4, encoding='iso-8859-1')
# RR_AC_res = pickle.load(file5, encoding='iso-8859-1')
# RDSa_AC_res = pickle.load(file6, encoding='iso-8859-1')
# for i in range(len(ULVR_AC_res)):
#     ULVR_AC_sum.append(np.sum(ULVR_AC_res[:i]))
kappas = [0.2, 0.3, 0.4, 0.5]
tickers = ['ULVR', 'RR', 'RDSa']

for ticker in tickers:
    AC_res = {}
    for kappa in kappas:
        AC_res[kappa] = []

    file = open(dirpath + '/loop{}_{}_test_res.txt'.format(loop, ticker), 'rb')
    res = pickle.load(file, encoding='iso-8859-1')
    file = open(dirpath + '/loop{}_{}_test_dones.txt'.format(loop, ticker), 'rb')
    dones = pickle.load(file, encoding='iso-8859-1')
    for kappa in kappas:
        file = open(AC_dirpath + '/{}_ACtest_res_kappa{}.txt'.format(ticker, kappa), 'rb')
        AC_res[kappa] = pickle.load(file, encoding='iso-8859-1')

    ep_res = []
    for indx in range(len(dones)):
        if indx == 0:
            ep_re = res[ : dones[indx] + 1]
            while len(ep_re) < 24:
                ep_re.append(0)
            ep_res += ep_re
        else:
            ep_re = res[dones[indx - 1] + 1 : dones[indx]]
            while len(ep_re) < 24:
                ep_re.append(0)
            ep_res += ep_re

    accumulated_res = []
    for i in range(len(ep_res)):
        accumulated_res.append(np.sum(ep_res[:i]))

    accumulated_AC_res = {}
    for kappa in kappas:
        accumulated_AC_res[kappa] = []
        for i in range(len(AC_res[kappa])):
            accumulated_AC_res[kappa].append(np.sum(AC_res[kappa][:i]))

    plt.plot(accumulated_res, label='RL Model')
    for kappa in kappas:
        plt.plot(accumulated_AC_res[kappa], label='AC Model, k = {}'.format(kappa))
    plt.title('Accumulated Profit over Hothead Agent, {}'.format(ticker))
    plt.xlabel('Trading Steps')
    plt.ylabel('Profit')
    plt.legend()
    plt.show()

    print('RL', np.average(ep_res), np.std(ep_res), np.average(ep_res)/np.std(ep_res))
    for kappa in kappas:
        print('AC kappa = {}'.format(kappa), np.average(AC_res[kappa]),
              np.std(AC_res[kappa]), np.average(AC_res[kappa])/np.std(AC_res[kappa]))