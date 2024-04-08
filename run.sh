#!/usr/bin/env bash

if [ ! -f .env ]; then
    cp .envsample .env
    echo "Warning: .env file not found. Copied .envsample as .env; please update with your values."
    exit 1
fi

source .env

source .venv/bin/activate

streamlit run app.py