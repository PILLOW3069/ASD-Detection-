import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
import os

def create_3d_cnn(input_shape=(182, 218, 182, 1)):
    """Creates the 3D CNN branch for MRI data."""
    inputs = Input(input_shape)

    x = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(inputs)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)

    return Model(inputs, x)

def create_dense_branch(input_shape=(6,)):
    """Creates the dense branch for phenotypic data."""
    inputs = Input(input_shape)
    x = Dense(32, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    return Model(inputs, x)

def create_multimodal_model(cnn_input_shape=(182, 218, 182, 1), dense_input_shape=(6,)):
    """Creates the multi-modal model by combining the 3D CNN and dense branches."""
    cnn_branch = create_3d_cnn(cnn_input_shape)
    dense_branch = create_dense_branch(dense_input_shape)

    combined_input = tf.keras.layers.concatenate([cnn_branch.output, dense_branch.output])

    x = Dense(256, activation='relu')(combined_input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[cnn_branch.input, dense_branch.input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define input shapes
cnn_input_shape = (182, 218, 182, 1)
dense_input_shape = (6,) # Number of phenotypic features

# Create the multi-modal model
asd_detector = create_multimodal_model(cnn_input_shape, dense_input_shape)

# Checkpoint callback
checkpoint_path = "/kaggle/working/latest.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch')
