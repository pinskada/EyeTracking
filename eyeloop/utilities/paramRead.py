import numpy as np
import eyeloop.config as config
import os

def read_pupil_parameters(side):
    fileName = f'pupil_parameters_{side}.npy'
    if os.path.exists(fileName):
        data = np.load(fileName, allow_pickle=True).item()


        config.graphical_user_interface.pupil_processor.binarythreshold = data.get('pupilThr', 50)
        config.graphical_user_interface.pupil_processor.blur = data.get('pupilBlur', (31,31))
        config.arguments.search_step = data.get('srchStep', 10)
        config.graphical_user_interface.pupil_processor.min_radius = data.get('minR', 5)
        config.graphical_user_interface.pupil_processor.max_radius = data.get('maxR', 20)
        print(f"[INFO] Importer {side}: Loaded saved parameters: {data}")
    else:
        config.graphical_user_interface.pupil_processor.binarythreshold = 50
        config.graphical_user_interface.pupil_processor.blur = (31, 31)
        config.arguments.search_step = 10
        config.graphical_user_interface.pupil_processor.min_radius = 5
        config.graphical_user_interface.pupil_processor.max_radius = 20
        print(f"[INFO] Importer {side}: No saved parameters found. Using default values.")
