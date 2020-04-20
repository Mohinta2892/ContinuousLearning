import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

allo = np.load("samia_binary/allo_e02_l001.npy", allow_pickle=True)
ego = np.load("samia_binary/ego_e02_l001.npy", allow_pickle=True)

print(allo.tolist())
print(ego.tolist())