#!/usr/bin/env bash

ollama serve &> ollama.log &
jupyter lab &> jupyter.log &
