#!/usr/bin/env bash

# requires for ollama module
if [ ! -d ~/scratch/.ollama ]; then
  mkdir ~/scratch/.ollama
  ln -s ~/scratch/.ollama ~/.ollama
fi
module load ollama/0.9.0 python

rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install jupyterlab

echo 'run `jupyter lab` next'
