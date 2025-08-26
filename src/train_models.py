import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import load_datasets

# ====================
# Model Definitions
# ====================
def build_zfnet(image_size=(224,224), num_classes=10):
    return Sequential([
        Conv2D(96, (7, 7), strides=2, activation='relu', input_shape=(*image_size, 3)),
        MaxPooling2D((3, 3), strides=2),
        Conv2D(256, (5, 5), activation='relu'),
        MaxPooling2D((3, 3), strides=2),
        Conv2D(384, (3, 3), activation='relu'),
        Conv2D(384, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((3, 3), strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

def build_vgg16(image_size=(224,224), num_classes=10):
    base = VGG16(include_top=False, weights="imagenet", input_shape=(*image_size, 3))
    for layer in base.layers:
        layer.trainable = False
    x = Flatten()(base.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base.input, outputs=output)

def build_googlenet(image_size=(224,224), num_classes=10):
    base = InceptionV3(include_top=False, weights="imagenet", input_shape=(*image_size, 3))
    for layer in base.layers:
        layer.trainable = False
    x = AveragePooling2D(pool_size=(5, 5))(base.output)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base.input, outputs=output)

# ====================
# Training Function
# ====================
def compile_and_train(model, train_dataset, val_dataset, name, epochs=10):
    model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=3)]
    )
    end = time.time()
    print(f"{name} training completed in {end - start:.2f} seconds")
    return history

def plot_metrics(history, name):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{name} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{name} - Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ====================
# Main Execution
# ====================
if __name__ == "__main__":
    train_dataset, val_dataset, class_names = load_datasets()
    
    # Train GoogLeNet (best performing model)
    model = build_googlenet(num_classes=len(class_names))
    history = compile_and_train(model, train_dataset, val_dataset, "GoogLeNet")
    plot_metrics(history, "GoogLeNet")
    
    # Save the model
    model.save("models/googlenet_animals10.h5")
