#!/usr/bin/env python3

import argparse
import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Build a Keras model based on specified parameters
def create_model(captcha_length, num_symbols, input_shape, model_depth=5, module_size=2):
    input_tensor = keras.Input(shape=input_shape)
    x = input_tensor
    for i in range(model_depth):
        for _ in range(module_size):
            x = keras.layers.Conv2D(32 * (2 ** min(i, 3)), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
    
    x = keras.layers.Flatten()(x)
    outputs = [keras.layers.Dense(num_symbols, activation='softmax', name=f'char_{i+1}')(x) for i in range(captcha_length)]
    
    return keras.Model(inputs=input_tensor, outputs=outputs)

# Dataset generator for Keras
class ImageSequence(keras.utils.Sequence):
    def __init__(self, dataset_dir, batch_size, captcha_length, symbols, width, height):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.symbols = symbols
        self.width = width
        self.height = height
        self.file_list = os.listdir(dataset_dir)
        self.files = {file.split('.')[0]: file for file in self.file_list}
        self.count = len(self.file_list)
        
    def __len__(self):
        return int(np.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, len(self.symbols)), dtype=np.uint8) for _ in range(self.captcha_length)]
        
        batch_files = random.sample(self.files.keys(), self.batch_size)
        
        for i, file_key in enumerate(batch_files):
            file_name = self.files[file_key]
            img_path = os.path.join(self.dataset_dir, file_name)
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
            X[i] = image
            
            label_text = file_key.replace(',', '|').replace(';', '\\').split('_')[0]
            for j, char in enumerate(label_text):
                y[j][i, self.symbols.find(char)] = 1

        return X, tuple(y)

def main():
    parser = argparse.ArgumentParser(description="Train a captcha recognition model.")
    parser.add_argument('--width', type=int, required=True, help='Width of captcha image')
    parser.add_argument('--height', type=int, required=True, help='Height of captcha image')
    parser.add_argument('--length', type=int, required=True, help='Length of captcha in characters')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--train-dataset', type=str, required=True, help='Directory for training dataset')
    parser.add_argument('--validate-dataset', type=str, required=True, help='Directory for validation dataset')
    parser.add_argument('--output-model-name', type=str, required=True, help='Output path for saving model')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs for training')
    parser.add_argument('--symbols', type=str, required=True, help='File with symbols used in captchas')
    args = parser.parse_args()

    with open(args.symbols) as f:
        captcha_symbols = f.readline().strip()

    model = create_model(args.length, len(captcha_symbols), (args.height, args.width, 3))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                  metrics=['accuracy'] * args.length)
    model.summary()

    train_data = ImageSequence(args.train_dataset, args.batch_size, args.length, captcha_symbols, args.width, args.height)
    val_data = ImageSequence(args.validate_dataset, args.batch_size, args.length, captcha_symbols, args.width, args.height)

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(args.output_model_name + '.keras', save_best_only=True)
    ]

    with open(args.output_model_name + ".json", "w") as json_file:
        json_file.write(model.to_json())

    try:
        model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks)
    except KeyboardInterrupt:
        print(f"Training interrupted. Saving weights as {args.output_model_name}_resume.h5")
        model.save_weights(args.output_model_name + '_resume.h5')

if __name__ == '__main__':
    main()

