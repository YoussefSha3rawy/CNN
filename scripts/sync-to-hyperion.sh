#!/bin/bash

# Get the name of the current directory
DIR_NAME=$(basename "$PWD")

# Run the rsync command using the directory name
rsync -uvaP --exclude=wandb --exclude=__pycache__ --exclude=*.zip --exclude=weights ./* adfx751@localhost:"$DIR_NAME"