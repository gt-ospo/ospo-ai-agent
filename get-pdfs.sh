#!/usr/bin/env bash

set -xeuo pipefail

OUT_DIR=$(pwd)/docs

if [ ! -d "$OUT_DIR" ]; then
  mkdir $OUT_DIR
  cd $(mktemp -d)
  wget https://github.com/docugami/KG-RAG-datasets/archive/547f132d3c2bcc2b9976ca1d8413b99d8ba45aa1.zip -O datasets.zip
  unzip datasets.zip
  cp ./KG-RAG-datasets-*/sec-10-q/data/v1/docs/* $OUT_DIR
  cp ./KG-RAG-datasets-*/sec-10-q/data/v1/qna_data*.csv $OUT_DIR
fi

if [ ! -d ~/.cache/datalab ]; then
  DATALAB_CACHE="$HOME/scratch/$(date +%s)"
  mkdir "$DATALAB_CACHE"
  ln -s "$DATALAB_CACHE" ~/.cache/datalab
fi

if [ ! -d text ]; then
  mkdir text
  for file in docs/*.pdf
  do
    OUTDIR="$(mktemp -d)"
    base="$(basename "$file")"
    echo "$file"
    .venv/bin/marker_single "$file" --use_llm --ollama_model "llama3.2:3b" --ollama_base_url "http://$OLLAMA_HOST" --llm_service=marker.services.ollama.OllamaService --output_dir "$OUTDIR"
    mv "$OUTDIR/${base%.*}/${base%.*}.md" "text"
    rm -rf "$OUTDIR"
  done
fi