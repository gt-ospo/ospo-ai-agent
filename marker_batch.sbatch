#!/bin/bash
 
#SBATCH -J ospo-ai-agent-marker-pdf
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH -t 06:00:00
#SBATCH -G H100:2
 
set -euxo pipefail

hostname

module purge
module load ollama/0.9.0 python

source .venv/bin/activate

ollama serve &

sleep 3

for file in docs/*.pdf; do
  marker_single "$file" \
    --use_llm \
    --llm_service=marker.services.ollama.OllamaService \
    --ollama_base_url http://localhost:11434 \
    --ollama_model llama3.2:3b \
    --output_format markdown \
    --output_dir marker_output
done
