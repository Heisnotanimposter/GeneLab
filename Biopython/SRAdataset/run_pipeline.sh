#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Step 1: Preprocess the data
echo "Starting preprocessing..."
python preprocess.py
echo "Preprocessing completed."

# Step 2: Train the model
echo "Starting model training..."
python train_model.py
echo "Model training completed."

# Step 3: Test the model
echo "Starting model testing..."
python test_model.py
echo "Model testing completed."

echo "All steps completed successfully."
