#!/usr/bin/env bash

python -c "import ollama; ollama.pull('$1')"
