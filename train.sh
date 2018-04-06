#!/usr/bin/env bash

file="nohup.out"
if [ -e "$file" ]
then
    echo "Removeing old '$file' file."
    rm "$file"
fi

echo "Starting training..."
nohup python3 python/s_train.py &

