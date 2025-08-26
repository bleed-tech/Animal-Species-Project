import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predict_species(model_path, img_path, image_size=(224,224), class_names=None):
    model = tf.keras.models.load_model(model_path)

    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)] if class_names else np.argmax(predictions)
    confidence = np.max(predictions)

    print(f"Predicted Species: {predicted_class} ({confidence*100:.2f}% confidence)")

if __name__ == "__main__":
    # Example usage
    class_names = ['cat','dog','horse','elephant','butterfly','chicken','spider','cow','sheep','squirrel']
    predict_species("models/googlenet_animals10.h5", "test_image.jpg", class_names=class_names)
