import h5py
with h5py.File('data/spikes/run_pop_001.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    for k in f.keys():
        print(k, f[k].shape, f[k].dtype)
