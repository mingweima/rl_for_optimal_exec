import pickle
import numpy as np
import matplotlib.pyplot as plt

dirpath = '/Users/gongqili/recordings/all/loop100_2020-02-24_09:49:09'

loop = 15

file1 = open(dirpath + '/loop{}_ULVR_test_res.txt'.format(loop), 'rb')
file2 = open(dirpath + '/loop{}_RR_test_res.txt'.format(loop), 'rb')
file3 = open(dirpath + '/loop{}_RDSa_test_res.txt'.format(loop), 'rb')
file4 = open(dirpath + '/ULVR_ACtest_res.txt', 'rb')
file5 = open(dirpath + '/RR_ACtest_res.txt', 'rb')
file6 = open(dirpath + '/RDSa_ACtest_res.txt', 'rb')
ULVR_res = pickle.load(file1, encoding='iso-8859-1')
RR_res = pickle.load(file2, encoding='iso-8859-1')
RDSa_res = pickle.load(file3, encoding='iso-8859-1')
ULVR_AC_res = pickle.load(file4, encoding='iso-8859-1')
RR_AC_res = pickle.load(file5, encoding='iso-8859-1')
RDSa_AC_res = pickle.load(file6, encoding='iso-8859-1')
ULVR_sum = []
ULVR_AC_sum = []
for i in range(len(ULVR_res)):
    ULVR_sum.append(np.sum(ULVR_res[:i]))
for i in range(len(ULVR_AC_res)):
    ULVR_AC_sum.append(np.sum(ULVR_AC_res[:i]))

plt.plot(ULVR_sum)
plt.plot(ULVR_AC_sum)
plt.show()
print('ULVR: ', np.average(ULVR_res), np.std(ULVR_res), np.average(ULVR_res)/np.std(ULVR_res))
print('RR: ', np.average(RR_res), np.std(RDSa_res), np.average(RR_res)/np.std(RDSa_res))
print('RDSa: ', np.average(RDSa_res), np.std(RDSa_res), np.average(RDSa_res)/np.std(RDSa_res))
print('ULVRAC: ', np.average(ULVR_AC_res), np.std(ULVR_AC_res), np.average(ULVR_AC_res)/np.std(ULVR_AC_res))
print('RRAC: ', np.average(RR_AC_res), np.std(RR_AC_res), np.average(RR_AC_res)/np.std(RR_AC_res))
print('RDaAC: ', np.average(RDSa_AC_res), np.std(RDSa_AC_res), np.average(RDSa_AC_res)/np.std(RDSa_AC_res))