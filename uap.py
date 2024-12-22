# -*- coding: utf-8 -*-
"""UAP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1N3_ZboS9y_1qiCkJjCr_B6VvfNN_yLZt

# Preparing Dataset

## Mengunduh Dataset
"""

!gdown --id 13-14hjG-XmYMzFy8SzHX9WoWBVihKqM3

!unzip Hardware.zip

"""## Library"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2
import os
import seaborn as sns
import random
import shutil
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
from collections import defaultdict
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

"""## EDA"""

# Path Folder Dataset
hardware_folder_path = 'Hardware'

# Get class names and image counts
class_names = os.listdir(hardware_folder_path)
image_counts = [len(os.listdir(os.path.join(hardware_folder_path, cls))) for cls in class_names]

# Create a Pandas Series for class distribution
class_distribution = pd.Series(image_counts, index=class_names)

"""### Distribusi Citra per Kelas

"""

import matplotlib.pyplot as plt

# Plot class distribution
plt.figure(figsize=(12, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of Images per Class')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""### Ukuran rata-rata dan dimensi citra"""

from PIL import Image
import os
import pandas as pd

# Collect image size data
image_data = []
for root, _, files in os.walk(hardware_folder_path):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            file_path = os.path.join(root, file)
            with Image.open(file_path) as img:
                image_data.append({
                    "Width": img.width,
                    "Height": img.height,
                    "Size (KB)": os.path.getsize(file_path) / 1024
                })

# Create a DataFrame
image_df = pd.DataFrame(image_data)

# Calculate average dimensions and file size
avg_width = image_df['Width'].mean()
avg_height = image_df['Height'].mean()
avg_size_kb = image_df['Size (KB)'].mean()

# Display results
print(f"Average Width: {avg_width:.2f} pixels")
print(f"Average Height: {avg_height:.2f} pixels")
print(f"Average File Size: {avg_size_kb:.2f} KB")

# Display summary statistics for dimensions
print("\nSummary Statistics:")
print(image_df.describe())

"""### Contoh Citra

"""

import matplotlib.pyplot as plt
import os
from PIL import Image

# Get class names
classes = sorted(os.listdir(hardware_folder_path))

# Plot one image from each class
plt.figure(figsize=(15, 10))

for idx, cls in enumerate(classes):
    class_folder = os.path.join(hardware_folder_path, cls)
    if os.path.isdir(class_folder):
        # Get the first image in the class folder
        first_image = os.listdir(class_folder)[0]
        img_path = os.path.join(class_folder, first_image)

        # Load and plot the image
        with Image.open(img_path) as img:
            plt.subplot(4, 4, idx + 1)
            plt.imshow(img)
            plt.title(cls)
            plt.axis('off')

plt.tight_layout()
plt.show()

"""### Jumlah citra per kelas

"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Count images in each class
class_counts = {}
for cls in os.listdir(hardware_folder_path):
    class_folder = os.path.join(hardware_folder_path, cls)
    if os.path.isdir(class_folder):
        class_counts[cls] = len([file for file in os.listdir(class_folder) if file.lower().endswith(('png', 'jpg', 'jpeg'))])

# Convert to DataFrame for better display
class_counts_df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Image Count']).sort_values(by='Image Count', ascending=False)

# Display class counts
print("Number of images per class:")
print(class_counts_df)

"""## Augmentasi"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import glob
import os
import shutil

# Path Folder Dataset
hardware_folder_path = 'Hardware/'  # Folder asli
target_per_class = 800  # Target per kelas

# Augmented Folder
augmented_data_path = 'Augmented_Hardware/'
os.makedirs(augmented_data_path, exist_ok=True)

# Get class names and file paths
class_names = [f for f in os.listdir(hardware_folder_path) if os.path.isdir(os.path.join(hardware_folder_path, f))]
class_file_paths = {cls: glob.glob(os.path.join(hardware_folder_path, cls, '*')) for cls in class_names}

# Count current images and calculate additional needed per class
additional_images_needed = {cls: max(0, target_per_class - len(files)) for cls, files in class_file_paths.items()}

# Augmentation configuration
augment_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Augmentation process
for cls, additional_needed in additional_images_needed.items():
    cls_path = os.path.join(hardware_folder_path, cls)
    aug_cls_path = os.path.join(augmented_data_path, cls)
    os.makedirs(aug_cls_path, exist_ok=True)

    # Copy existing images to augmented folder without changing file names
    for img_path in glob.glob(os.path.join(cls_path, '*')):
        img_name = os.path.basename(img_path)
        target_path = os.path.join(aug_cls_path, img_name)
        if not os.path.exists(target_path):
            shutil.copy(img_path, target_path)

    # Perform augmentation if additional images are needed
    if additional_needed > 0:
        cls_images = glob.glob(os.path.join(cls_path, "*"))
        count = 0
        for img_path in cls_images:
            if count >= additional_needed:
                break
            try:
                img = load_img(img_path)
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # Generate augmented images
                for batch in augment_datagen.flow(
                    img_array,
                    batch_size=1,
                    save_to_dir=aug_cls_path,
                    save_prefix='aug',
                    save_format='jpg'
                ):
                    count += 1
                    if count >= additional_needed:
                        break
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# Verify augmentation results
augmented_counts = {cls: len(os.listdir(os.path.join(augmented_data_path, cls))) for cls in class_names}
print("Augmentation completed. Images per class:", augmented_counts)

"""### Jumlah citra setelah di augmentasi"""

import os

# Path to dataset
hardware_folder_path_augment = 'Augmented_Hardware'

# Count images in each class
class_counts = {}
for cls in os.listdir(hardware_folder_path_augment):
    class_folder = os.path.join(hardware_folder_path_augment, cls)
    if os.path.isdir(class_folder):
        class_counts[cls] = len([file for file in os.listdir(class_folder) if file.lower().endswith(('png', 'jpg', 'jpeg'))])

# Total number of images
total_images = sum(class_counts.values())
print(f"\nTotal images in dataset: {total_images}")

"""### Distribusi citra setelah di augmentasi"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to augmented dataset
#augmented_dataset_path = 'path/to/your/augmented/dataset'

# Count images in each class
augmented_class_counts = {}
for cls in os.listdir(hardware_folder_path_augment):
    class_folder = os.path.join(hardware_folder_path_augment, cls)
    if os.path.isdir(class_folder):
        augmented_class_counts[cls] = len([file for file in os.listdir(class_folder) if file.lower().endswith(('png', 'jpg', 'jpeg'))])

# Convert to DataFrame for visualization
augmented_class_counts_df = pd.DataFrame(list(augmented_class_counts.items()), columns=['Class', 'Image Count']).sort_values(by='Image Count', ascending=False)

# Display augmented class counts
print("Number of images per class after augmentation:")
print(augmented_class_counts_df)

# Plot augmented class distribution
plt.figure(figsize=(12, 6))
augmented_class_counts_df.set_index('Class')['Image Count'].plot(kind='bar', color='lightgreen')
plt.title('Distribution of Images per Class After Augmentation')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""## Splitting Data Citra"""

import os
import shutil
from sklearn.model_selection import train_test_split

original_dataset_path = 'Augmented_Hardware/'
split_base_path = 'Split_Dataset/'
os.makedirs(split_base_path, exist_ok=True)

train_path = os.path.join(split_base_path, 'train')
val_path = os.path.join(split_base_path, 'val')
test_path = os.path.join(split_base_path, 'test')

# Function to create directories
def create_directories(base_path, class_names):
    for cls in class_names:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

# Get class names and file paths
class_names = [f for f in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, f))]
class_file_paths = {
    cls: [os.path.join(original_dataset_path, cls, file)
          for file in os.listdir(os.path.join(original_dataset_path, cls))
          if file.lower().endswith(('png', 'jpg', 'jpeg'))]
    for cls in class_names
}

# Create directories for splits
create_directories(train_path, class_names)
create_directories(val_path, class_names)
create_directories(test_path, class_names)

# Split files into train, validation, and test (70-20-10)
for cls, files in class_file_paths.items():
    train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=1/3, random_state=42)

    # Copy files to respective directories
    for file in train_files:
        shutil.copy(file, os.path.join(train_path, cls))
    for file in val_files:
        shutil.copy(file, os.path.join(val_path, cls))
    for file in test_files:
        shutil.copy(file, os.path.join(test_path, cls))

print("Dataset successfully split into train (70%), validation (20%), and test (10%) sets!")

"""## Prepare data for model"""

# Parameter
img_height, img_width = 125, 125
batch_size = 32
epochs = 50

# Image data generator
train_datagen = ImageDataGenerator(rescale=1./256)
val_datagen = ImageDataGenerator(rescale=1./256)
test_datagen = ImageDataGenerator(rescale=1./256)

"""### Inisialisai object diirectory

"""

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),  # Match input shape of your model
    batch_size=batch_size,  # Adjust batch size as needed
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

"""### Amount of Data after split

"""

# prompt: menampilkan jumlah data train, val, dan tes

import os

split_base_path = 'Split_Dataset/'
train_path = os.path.join(split_base_path, 'train')
val_path = os.path.join(split_base_path, 'val')
test_path = os.path.join(split_base_path, 'test')

# Function to count files in a directory
def count_files(directory):
  return sum([len(files) for r, d, files in os.walk(directory)])

train_count = count_files(train_path)
val_count = count_files(val_path)
test_count = count_files(test_path)

print(f"Jumlah data train: {train_count}")
print(f"Jumlah data val: {val_count}")
print(f"Jumlah data tes: {test_count}")

"""# Creating a Model CNN"""

num_classes = 14
input_shape = (125, 125, 3)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the feature maps
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization

    # Fully Connected Layer 2 (Output Layer)
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax activation

    return model

# Create the model
cnn_model = create_cnn_model(input_shape, num_classes)

# Compile the model
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Display the model summary
cnn_model.summary()

"""## Training Model"""

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

history = cnn_model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[early_stopping],
    verbose=1
)

"""## Save Model CNN"""

from google.colab import files
cnn_model.save('cnn_model.h5')
files.download('cnn_model.h5')

"""## Plot Accuracy and Loss"""

import matplotlib.pyplot as plt

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(train_loss) + 1)

# Plot loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""## Classification Report CNN on Validation Data"""

# Predict on validation data
y_pred = cnn_model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

# Print classification report
print(classification_report(y_true, y_pred_classes, target_names=val_generator.class_indices.keys()))

"""## Confussion Matrix"""

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = cnn_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(test_generator.class_indices.keys()),
            yticklabels=list(test_generator.class_indices.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

"""## Evaluating Model on Data test"""

# Evaluate the model on the test set
test_loss, test_accuracy = cnn_model.evaluate(test_generator, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

"""## Classification Report on Test Data"""

# Predict on test data
y_pred = cnn_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Print classification report
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

"""# Pre-Trained MobileNet

"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load MobileNet with pretrained weights
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(125, 125, 3))

# Freeze the base model layers to retain pretrained weights
base_model.trainable = False

num_classes = 14

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    verbose=1
)

from google.colab import files
model.save('model.h5')
files.download('model.h5')

import matplotlib.pyplot as plt

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(train_loss) + 1)

# Plot loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Predict on validation data
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

# Print classification report
print(classification_report(y_true, y_pred_classes, target_names=val_generator.class_indices.keys()))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(test_generator.class_indices.keys()),
            yticklabels=list(test_generator.class_indices.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on test data
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Print classification report
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))