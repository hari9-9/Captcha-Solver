#!/bin/bash

# Define parameters
WIDTH=192
HEIGHT=96
BATCH_SIZE=64
EPOCHS=20
SYMBOLS_FILE="symbols.txt"
OUTPUT_BASE_DIR="char"

for i in {1..6}; do
    echo "Running iteration for character length = $i"
    
    # Define directory paths
    OUTPUT_DIR="${OUTPUT_BASE_DIR}${i}"
    TRAIN_DIR="$OUTPUT_DIR/testing/train"
    VALIDATE_DIR="$OUTPUT_DIR/testing/validate"
    MODEL_DIR="$OUTPUT_DIR/single_char${i}_model"

    # Generate training and validation sets
    python3 generate_data.py --width $WIDTH --height $HEIGHT --min-length $i --max-length $i --count 64000 --output-dir "$OUTPUT_DIR/testing" --symbols $SYMBOLS_FILE

    # Train the model with the dynamically generated model directory
    python3 train_gpu_filtered.py --width $WIDTH --height $HEIGHT --length $i --batch-size $BATCH_SIZE --train-dataset $TRAIN_DIR --validate-dataset $VALIDATE_DIR --output-model-name $MODEL_DIR --epochs $EPOCHS --symbols $SYMBOLS_FILE

    echo "Completed iteration for character length = $i"
    echo "------------------------------------"
done

