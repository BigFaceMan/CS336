#!/bin/bash

cd /home/spsong/Code/cs336/assignment1-basics/cs336_basics

CONFIG_PATH="/home/spsong/Code/cs336/assignment1-basics/config/train_small_story.yaml"
python train.py --config $CONFIG_PATH