import os
import scipy.io
import numpy as np

script_dir = os.path.dirname(__file__)
src = os.path.join(script_dir, 'dataSorted_allOrientations.mat')
if not os.path.exists(src):
    raise FileNotFoundError(src)

mat = scipy.io.loadmat(src)
if 'out' not in mat:
    raise KeyError("'out' key not found in mat file")

out = mat['out']
# Duplicate last dimension to create left/right versions
new_out = np.concatenate([out, out], axis=3)
# Round to 4 decimals like import_data does
new_out = np.round(new_out, 4)

dst = os.path.join(script_dir, 'dataSorted_leftAndRight.mat')
scipy.io.savemat(dst, {'out': new_out})
print(f'Wrote {dst} with shape {new_out.shape}')
