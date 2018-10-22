import argparse, h5py, pickle, os
import numpy as np


from info_measures.numpy import kraskov_stoegbauer_grassberger as ksg
from info_measures.dataset_generators import (VectorSpaceGenerator,
                                              ImageGenerator,
                                              MultiChannelTimeseriesGenerator)

parser = argparse.ArgumentParser(description='Calculate PI for a dataset.')
parser.add_argument('data_file', type=str,
                    help='HDF5 file with an "X variable.')
parser.add_argument('dataset_type', type=str, choices=['v', 'i', 't'],
                    help='VectorSpace (v), Image (i), or Timeseries (t).')
parser.add_argument('--n_samples', '-n', type=int, required=True,
                    help='Number of samples.')
parser.add_argument('--dim', '-d', type=int, required=True,
                    help='Dimension for each variable.')
parser.add_argument('--seed', type=int, required=True,
                    help='Random seed.')
parser.add_argument('--save_folder', type=str, required=True,
                    help='Base location to save results.')
parser.add_argument('--grow_axis', '-g', type=int,
                    help='Which image axis to use as PI growth axis.')

args = parser.parse_args()
data_file = args.data_file
dim = args.dim
dataset_type = args.dataset_type
n_samples = args.n_samples
seed = args.seed
save_folder = args.save_folder
grow_axis = args.grow_axis
args = vars(args)

with h5py.File(data_file) as f:
    raw_data = f['X'].value

ndim = raw_data.ndim
if dataset_type == 'v':
    if ndim != 3:
        raise ValueError
    ds = VectorSpaceGenerator(raw_data)
    samples = ds.sample_data(dim, n_samples=n_samples)
elif dataset_type == 'i':
    if ndim != 4:
        raise ValueError
    ds = ImageGenerator(raw_data, grow_axis=grow_axis)
    samples = ds.sample_data(dim, perp_dim=1, n_samples=n_samples)
elif dataset_type == 't':
    if ndim != 4:
        raise ValueError
    ds = MultiChannelTimeseriesGenerator(raw_data, seed=seed)
    samples = ds.sample_data(dim, n_channels=1, n_samples=n_samples)
else:
    raise NotImplementedError
rng = ds.rng

X = samples[:, :dim]
X = X.reshape(n_samples, -1)
Y = samples[:, dim:]
Y = Y.reshape(n_samples, -1)

mi_e = ksg.MutualInformation(X, Y)
args['mi'] = mi_e.mutual_information(n_jobs=4)

rng.shuffle(Y)
mi_e = ksg.MutualInformation(X, Y)
args['mi_shuffle'] = mi_e.mutual_information(n_jobs=4)

file_name = '{}d_{}s.pkl'.format(dim, seed)
save_path = os.path.join(save_folder,
                         os.path.splitext(os.path.basename(data_file))[0],
                         dataset_type,
                         '{}_samples'.format(str(n_samples)))
try:
    os.makedirs(save_path)
except FileExistsError:
    pass

save_path = os.path.join(save_path, file_name)
print(save_path)
with open(save_path, 'wb') as f:
    pickle.dump(args, f)
