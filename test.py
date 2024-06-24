import tensorflow as tf
import albumentations as albu
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
from ModelArchitecture.DiceLoss import dice_metric_loss
from ModelArchitecture import DUCK_Net

import glob
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
import os

# Checking the number of GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Setting the model parameters
img_size = 128 # Change as required
dataset_type = 'kvasir' # Options: kvasir/cvc-clinicdb/cvc-colondb/etis-laribpolypdb
learning_rate = 1e-4
seed_value = 1000
filters = 17 # Number of filters
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
model_type = "DuckNet"

progress_path = 'ProgressFull.csv'
progressfull_path = 'ProgressFullpath.txt'
plot_path = 'ProgressFullplot.png'
model_path = 'ModelSaveTensorFlow/Ducknet.h5'

EPOCHS = 3
min_loss_for_saving = 0.8

folder_path = "Kvasir-SEG"  # Add the path to your data directory
IMAGES_PATH = os.path.join(folder_path, 'images')
MASKS_PATH = os.path.join(folder_path, 'masks')

def load_data(img_height, img_width, images_to_be_loaded, dataset):
    if dataset == 'kvasir':
        train_ids = glob.glob(os.path.join(IMAGES_PATH, "*.jpg"))
    elif dataset == 'cvc-clinicdb':
        train_ids = glob.glob(os.path.join(IMAGES_PATH, "*.tif"))
    elif dataset == 'cvc-colondb' or dataset == 'etis-laribpolypdb':
        train_ids = glob.glob(os.path.join(IMAGES_PATH, "*.png"))

    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)  # Set the number of images to be loaded

    X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

    print(f'Resizing {images_to_be_loaded} training images and masks...')
    for n, id_ in tqdm(enumerate(train_ids), total=images_to_be_loaded):
        if n == images_to_be_loaded:
            break

        image_path = id_
        mask_path = image_path.replace("images", "masks")

        image = imread(image_path)
        mask_ = imread(mask_path)

        # Resize image
        pillow_image = Image.fromarray(image)
        pillow_image = pillow_image.resize((img_width, img_height), resample=Image.LANCZOS)
        image = np.array(pillow_image)

        # Normalize image
        X_train[n] = image / 255.0

        # Resize mask
        pillow_mask = Image.fromarray(mask_)
        pillow_mask = pillow_mask.resize((img_width, img_height), resample=Image.LANCZOS)
        mask_ = np.array(pillow_mask)

        # Convert mask to binary (Ensure single channel)
        mask = (mask_ >= 127).astype(np.uint8)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0] # Select first channel if it has 3

        Y_train[n] = mask

    Y_train = np.expand_dims(Y_train, axis=-1)  # Add channel dimension for consistency

    # Debugging output
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Unique values in Y_train: {np.unique(Y_train)}")

    return X_train, Y_train

# Load the data
X, Y = load_data(img_size, img_size, -1, dataset_type)

# Splitting the data, seed for reproducibility
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=seed_value)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.111, shuffle=True, random_state=seed_value)

# Defining the augmentations
aug_train = albu.Compose([
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
    albu.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22), always_apply=True),
])

def augment_images():
    x_train_out = []
    y_train_out = []

    for i in range(len(x_train)):
        ug = aug_train(image=x_train[i], mask=y_train[i])
        x_train_out.append(ug['image'])
        y_train_out.append(ug['mask'])

    return np.array(x_train_out), np.array(y_train_out)

# Creating the model
model = DUCK_Net.create_model(img_height=img_size, img_width=img_size, input_chanels=3, out_classes=1, starting_filters=filters)

# Compiling the model
model.compile(optimizer=optimizer, loss=dice_metric_loss)

# Training the model
step = 0

for epoch in range(EPOCHS):
    print(f'Training, epoch {epoch}')
    print('Learning Rate: ' + str(learning_rate))

    step += 1

    image_augmented, mask_augmented = augment_images()

    csv_logger = CSVLogger(progress_path, append=True, separator=';')

    model.fit(x=image_augmented, y=mask_augmented, epochs=1, batch_size=4, validation_data=(x_valid, y_valid), verbose=1, callbacks=[csv_logger])

    prediction_valid = model.predict(x_valid, verbose=0)
    loss_valid = dice_metric_loss(y_valid, prediction_valid)

    loss_valid = loss_valid.numpy()
    print("Loss Validation: " + str(loss_valid))

    prediction_test = model.predict(x_test, verbose=0)
    loss_test = dice_metric_loss(y_test, prediction_test)
    loss_test = loss_test.numpy()
    print("Loss Test: " + str(loss_test))

    with open(progressfull_path, 'a') as f:
        f.write(f'epoch: {epoch}\nval_loss: {loss_valid}\ntest_loss: {loss_test}\n\n\n')

    if min_loss_for_saving > loss_valid:
        min_loss_for_saving = loss_valid
        print("Saved model with val_loss: ", loss_valid)
        model.save(model_path)

    del image_augmented, mask_augmented
    gc.collect()

# Computing the metrics and saving the results
print("Loading the model")

if os.path.isfile(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'dice_metric_loss': dice_metric_loss})

    prediction_train = model.predict(x_train, batch_size=4)
    prediction_valid = model.predict(x_valid, batch_size=4)
    prediction_test = model.predict(x_test, batch_size=4)

    print("Predictions done")

    dice_train = f1_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
    dice_test = f1_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
    dice_valid = f1_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))

    print("Dice finished")

    miou_train = jaccard_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
    miou_test = jaccard_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
    miou_valid = jaccard_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))

    print("Miou finished")

precision_train = precision_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                                  np.ndarray.flatten(prediction_train > 0.5))
precision_test = precision_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                                 np.ndarray.flatten(prediction_test > 0.5))
precision_valid = precision_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                                  np.ndarray.flatten(prediction_valid > 0.5))

print("Precision finished")

recall_train = recall_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                            np.ndarray.flatten(prediction_train > 0.5))
recall_test = recall_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                           np.ndarray.flatten(prediction_test > 0.5))
recall_valid = recall_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                            np.ndarray.flatten(prediction_valid > 0.5))

print("Recall finished")

accuracy_train = accuracy_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                                np.ndarray.flatten(prediction_train > 0.5))
accuracy_test = accuracy_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                               np.ndarray.flatten(prediction_test > 0.5))
accuracy_valid = accuracy_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                                np.ndarray.flatten(prediction_valid > 0.5))

print("Accuracy finished")

final_file = f'results_{model_type}_{filters}_{dataset_type}.txt'
print(final_file)

with open(final_file, 'a') as f:
    f.write(f'{dataset_type}\n\n')
    f.write(f'dice_train: {dice_train} dice_valid: {dice_valid} dice_test: {dice_test}\n\n')
    f.write(f'miou_train: {miou_train} miou_valid: {miou_valid} miou_test: {miou_test}\n\n')
    f.write(f'precision_train: {precision_train} precision_valid: {precision_valid} precision_test: {precision_test}\n\n')
    f.write(f'recall_train: {recall_train} recall_valid: {recall_valid} recall_test: {recall_test}\n\n')
    f.write(f'accuracy_train: {accuracy_train} accuracy_valid: {accuracy_valid} accuracy_test: {accuracy_test}\n\n\n\n')

print('File done')
