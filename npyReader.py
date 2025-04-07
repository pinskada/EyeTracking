import numpy as np

# Replace 'filename.npy' with the path to your file
filePath = 'pupil_parameters.npy'


data = np.load(filePath, allow_pickle=True)

print(data)