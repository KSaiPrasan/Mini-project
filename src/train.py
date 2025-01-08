from .model import create_model, compile_model
from .data_preprocessing import preprocess_data  # Corrected import
import tensorflow as tf
import os

def main():
    # Hyperparameters
    input_shape = (224, 224, 3)
    batch_size = 32
    epochs = 10

    # Directories
    train_dir = 'data/train'
    val_dir = 'data/val'
    model_dir = 'models'

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    train_generator, val_generator = preprocess_data(train_dir, val_dir, img_size=input_shape[:2], batch_size=batch_size)

    # Build and compile model
    model = create_model(input_shape)
    compile_model(model)

    # Define callbacks (e.g., model checkpointing)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), save_best_only=True)

    # Train model
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[checkpoint_cb])

    # Evaluate the model after training
    test_loss, test_acc = model.evaluate(val_generator)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")

    # Save final model
    model.save(os.path.join(model_dir, 'final_model.h5'))

if __name__ == '__main__':
    main()