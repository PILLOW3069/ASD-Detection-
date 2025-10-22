from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Model import asd_detector
from Data_Feeder import train_generator, val_generator

# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath='best_model.h5',
    save_best_only=True,
    monitor='val_loss'
)

# Train the multi-modal model
history = asd_detector.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,  # Increased epochs with early stopping
    callbacks=[early_stopping, model_checkpoint],
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Save the trained model
asd_detector.save('multimodal_asd_predictor.h5')
