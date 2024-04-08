#!/usr/bin/env bash

echo "Creating Python virtual environment"
python -m venv .venv

echo "Activating Python virtual environment"
source .venv/bin/activate

echo "Installing Python dependencies"
pip install -r requirements.txt
