import tensorflow as tf

def load_datasets(dataset_path="dataset/raw-img", image_size=(224, 224), batch_size=32, validation_split=0.2, seed=42):
    """
    Loads and preprocesses the Animals-10 dataset.
    Returns train_dataset, val_dataset, class_names
    """

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    # Normalize pixel values (0–255 → 0–1)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    class_names = train_dataset.class_names
    return train_dataset, val_dataset, class_names
