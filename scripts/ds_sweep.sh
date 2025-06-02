#!/bin/bash

ROOT_DIR=$(pwd)/../configs/hyperparam/wandb   
optimizer=$1
dataset=$2   
num_agents=$3

sweep_config="$ROOT_DIR/$optimizer/$dataset.yaml"  

run_agent() {
    local agent=$1   
    echo "Creating sweep for $sweep_config"
    
    sweep_id=$(wandb sweep "$sweep_config")
    
  
    echo $sweep_id
  
    
    echo "Running agent $agent for $sweep_id"
    wandb agent "$sweep_id"  
    echo $?
    if [$? -eq 1]; then
        echo "Failed running agent" $?
    fi
} 

 for i in $(seq 1 "$num_agents"); do 
    run_agent "$i" &
done 

wait   
echo "All completed"


