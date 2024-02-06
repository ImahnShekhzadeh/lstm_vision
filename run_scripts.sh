#!/bin/sh
isort /app/lstm_vision/*.py
black /app/lstm_vision/*.py

python3 -B /app/lstm_vision/run.py "$@"
