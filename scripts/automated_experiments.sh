#!/bin/bash

# Get the project root directory (src)


ROOT_DIR=$(ls)/../configs/hyperparam/wandb/

optimizers=$(ls $ROOT_DIR)

for opti in $optimizers: 

    configs=$(ls $ROOT_DIR/$opti)
    for file in $configs; do 
        echo "reating sweep for $file"
        sweep_id=$(wandb sweep $file)

        if ["$SWEEP_ID"-z ]; then
            echo "Failed to create sweep!"
            exit 1

        echo "running agent for $sweep_id"
        wandb agent $sweep_id
    done
    
    for file in *.yml; do
        echo "reating sweep for $file"
        sweep_id=$(wandb sweep $file)

        if ["$SWEEP_ID"-z ]; then
            echo "Failed to create sweep!"
            exit 1

        echo "running agent for $sweep_id"
        wandb agent $sweep_id
    done





wait 
echo "Finished"
