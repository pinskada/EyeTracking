import numpy as np
import eyeloop.config as config
import os

def save_pupil_parameters(side):

    # Define the same dictionary
    data = {
        'pupilThr': config.graphical_user_interface.pupil_processor.binarythreshold,
        'pupilBlur': config.graphical_user_interface.pupil_processor.blur,
        'srchStep': config.arguments.search_step,
        'minR': config.graphical_user_interface.pupil_processor.min_radius,
        'maxR': config.graphical_user_interface.pupil_processor.min_radius
    }

    # Save it as a .npy file with pickle enabled
    fileName = f'pupil_parameters_{side}.npy'
    np.save(fileName, data, allow_pickle=True)
    print(f"[INFO] Importer {side}: Saved parameters to {fileName}.")