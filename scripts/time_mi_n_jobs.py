import numpy as np
import multiprocessing, time

from info_measures.numpy import kraskov_stoegbauer_grassberger as ksg


rng = np.random.RandomState(20180518)
n_repeat = 10
n_samples = np.logspace(3, 7, 10, dtype=int)
#n_samples = np.logspace(2, 4, 5, dtype=int)
n_cores = multiprocessing.cpu_count()

timings = np.zeros((2, n_samples.size, n_cores, n_repeat))

for ii, ns in enumerate(n_samples):
    for jj in range(n_repeat):
        d = rng.randn(ns, 2)
        x = d[:, [0]]
        y = d[:, [1]]
        ksg_e = ksg.MutualInformation(x, y)
        for n_jobs in range(1, n_cores+1):
            for kind in [1, 2]:
                t0 = time.time()
                mi = ksg_e.mutual_information(kind=kind, n_jobs=n_jobs)
                t1 = time.time()
                timings[kind-1,ii,n_jobs-1,jj] = t1-t0
    print('Finished {} samples.'.format(n_samples))
print(2, 'n_samples', 'cores')
print(np.median(timings, axis=-1))
np.savez('time.npz', **{'timings': timings, 'n_samples': n_samples})
