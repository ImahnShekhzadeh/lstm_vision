#!/bin/sh
isort /app/scripts/*.py
black /app/scripts/*.py

python3 -B /app/scripts/run.py --batch_size 8192 --bidirectional \
                               --dropout_rate 0.2 --learning_rate 1e-3 \
                               --num_epochs 15 --seed_number 0 \
                               --use_amp --freq_output__train 1 \
                               --freq_output__val 1 --num_workers 4 \
                               --pin_memory