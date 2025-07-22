#!/usr/bin/env bash
set -euo pipefail

# Ensure docs directory exists
if [ ! -d docs ]; then
  ./get-pdfs.sh
fi

# Install marker-pdf if not already installed
echo "Installing marker-pdf package..."
pip install marker-pdf

if [ ! -d ~/.cache/datalab ]; then
  DATALAB_CACHE="/scratch/$(date +%s)"
  mkdir "$DATALAB_CACHE"
  ln -s "$DATALAB_CACHE" ~/.cache/datalab
fi

mkdir -p text
for file in docs/*.pdf
do
  base="$(basename "$file")"
  pdftotext "$file" "text/${base%.*}.txt"
done

mkdir -p embed2
for file in text/*.txt
do
  base="$(basename "$file")"
  if [ ! -f "embed2/$base.marker" ]; then
    echo "$file"
    python embed2.py "$file" embed2
    touch "embed2/$base.marker"
  fi
done
