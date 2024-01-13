#!/bin/bash
export WANDB_PROJECT=gnn_llm

# opt-350m
export WANDB_NAME=exp050_opt2.7b
deepspeed train.py --config_file configs/training_config_$WANDB_NAME.yml
