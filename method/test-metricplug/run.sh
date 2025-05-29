#!/bin/bash

# Define the parameters for grid search
DATAS=("Cora" "CiteSeer")
SIMILARITIES=("cosine" "tropical" "euclidean" "manhattan" "hamming" )
LRS=(0.001 0.005 0.01 0.05)
HID=(64 128 256)
P_DIM=(64 128 256)

# Create a CSV file to store the results
OUTPUT_FILE="grid_search_results.csv"
echo "dataset,similarity,lr,hid_dim,proj_dim,val_acc,test_acc" > $OUTPUT_FILE

# Loop over all combinations of similarities and learning rates
for D in "${DATAS[@]}"; do
for SIM in "${SIMILARITIES[@]}"; do
    for LR in "${LRS[@]}"; do
        for H in "${HID[@]}"; do
            for P in "${P_DIM[@]}"; do
                echo "Running with similarity=$SIM and lr=$LR..."
                
                # Run the Python script with the current hyperparameters
                python main.py --similarity $SIM --lr $LR --hidden_dim $H --proj_dim $P --dataset $D> result.log
                
                # Extract the validation and test accuracy from the result.log
                VAL_ACC=$(grep "Best validation accuracy" result.log | awk '{print $5}')
                TEST_ACC=$(grep "Final test accuracy" result.log | awk '{print $5}')
                
                
            done
        done
    done
done
done

