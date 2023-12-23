#!/bin/sh
isort /app/scripts/*.py
black /app/scripts/*.py

python3 -B /app/scripts/run.py --bidirectional --seed_number 0 --use_amp \
                               --freq_output__train 10 --freq_output__val 2