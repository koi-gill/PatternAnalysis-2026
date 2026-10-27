"""
Containing the source code for training, validating, testing, and saving your model.
The model should be imported from modules.py and the data loader should be imported from
dataset.py. Make sure to plot the losses and metrics during training.
"""

import modules
import dataset
from pathlib import Path

train_path = Path("HipMRI_Study_open/keras_slices_data/keras_slices_train")
file_paths = [str(file) for file in train_path.iterdir() if file.is_file()]

images = dataset.load_data_2D(file_paths, normImage=True, categorical=False, early_stop=True)

print(images)