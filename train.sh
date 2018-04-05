#!/usr/bin/env bash

file="nohup.out"
if [ ! -f "$file"  ]
then
    echo "Removeing old '$file.'"
    rm "$file"
fi

echo "Starting training..."
nohup python3 python/s_train.py &
