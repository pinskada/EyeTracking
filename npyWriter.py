import numpy as np

# Define the same dictionary
data = {
    'pupil': [41.0, (53, 53)],
    'cr1': [171.7, (3, 3)],
    'cr2': [171.7, (3, 3)]
}

# Save it as a .npy file with pickle enabled
np.save('pupil_parameters.npy', data, allow_pickle=True)
