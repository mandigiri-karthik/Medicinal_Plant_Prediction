import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Paths to the dataset
train_dir = 'D:\Downloads\IEEE_internship\Medicinal_plant_dataset_Train'
test_dir = 'D:\Downloads\IEEE_internship\Medicinal_plant_test'

# Medicinal uses dictionary (Example: you need to fill this with actual data)
medicinal_uses = {
    'plant_1': 'Medicinal use of plant 1.',
    'plant_2': 'Medicinal use of plant 2.',
    'plant_3': 'Medicinal use of plant 1.',
    'plant_4': 'Medicinal use of plant 2.',
    'plant_5': 'Medicinal use of plant 1.',
    'plant_6': 'Medicinal use of plant 2.',
    'plant_7': 'Medicinal use of plant 1.',
    'plant_8': 'Medicinal use of plant 2.',
    'plant_9': 'Medicinal use of plant 1.',
    'plant_10': 'Medicinal use of plant 2.',
    # Add all plant names and their medicinal uses here
}

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Building the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Save the model
model.save('plant_classifier_model.h5')

# Function to predict plant and get medicinal uses
def get_medicinal_use(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    # Mapping the predicted class index to the plant name
    plant_name = list(train_generator.class_indices.keys())[predicted_class_index]
    
    # Get medicinal uses
    return plant_name, medicinal_uses.get(plant_name, "No medicinal use information available.")

# Example usage
plant_name, uses = get_medicinal_use('path_to_image.jpg')
print(f"Plant: {plant_name}\nMedicinal Uses: {uses}")
