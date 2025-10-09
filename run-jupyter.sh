#!/usr/bin/env bash

module purge
module load ollama/0.9.0 python
source .venv/bin/activate

ollama serve &> ollama.log &
jupyter lab &> jupyter.log &
