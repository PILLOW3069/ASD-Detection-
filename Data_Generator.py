import tensorflow as tf
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import rotate, shift, zoom
from tensorflow.keras.utils import Sequence

def load_nifti_image(file_path):
    """Loads a NIfTI image and expands its dimensions."""
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return np.expand_dims(img_data, axis=-1)

def augment_data(image):
    """Applies 3D data augmentation."""
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10, 10)
        image = rotate(image, angle, reshape=False, mode='nearest')
    if np.random.rand() > 0.5:
        shift_val = np.random.uniform(-5, 5, 3)
        image = shift(image, shift_val, mode='nearest')
    return image

class NiftiDataGenerator(Sequence):
    """Generates multi-modal data for Keras."""
    def __init__(self, file_paths, labels, phenotypic_data, batch_size=5, dim=(182, 218, 182), n_channels=1, augment=True):
        self.file_paths = file_paths
        self.labels = labels
        self.phenotypic_data = phenotypic_data
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.file_paths))
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data."""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.file_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        batch_phenotypic = self.phenotypic_data.iloc[indexes]

        X, y = self.__data_generation(batch_paths, batch_labels, batch_phenotypic)
        return X, y

    def __data_generation(self, batch_paths, batch_labels, batch_phenotypic):
        """Generates data containing batch_size samples."""
        X_cnn = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_dense = np.empty((self.batch_size, self.phenotypic_data.shape[1]))
        y = np.empty((self.batch_size), dtype=int)

        for i, (file_path, label, pheno_row) in enumerate(zip(batch_paths, batch_labels, batch_phenotypic.iterrows())):
            img_data = load_nifti_image(file_path)
            if self.augment:
                img_data = augment_data(img_data)
            X_cnn[i,] = img_data
            X_dense[i,] = pheno_row[1].values
            y[i] = label

        return [X_cnn, X_dense], y
