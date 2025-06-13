#!/usr/bin/env bash
set -euo pipefail
if [ ! -d docs ]; then
  ./get-pdfs.sh
fi

mkdir -p text
for file in docs/*.pdf
do
  base="$(basename "$file")"
  if [ ! -f "text/${base%.*}.txt" ]; then
    echo make "text/${base%.*}.txt"
    pdftotext "$file" "text/${base%.*}.txt"
  fi
done

mkdir -p summaries
for file in text/*.txt
do
  base="$(basename "$file")"
  if [ ! -f "summaries/$base" ]; then
    echo make "summaries/$base"
    python summarize.py "$file" "summaries/$base"
  fi
done

mkdir -p embeddings
for file in summaries/*.txt
do
  base="$(basename "$file")"
  if [ ! -f "embeddings/${base%.*}.json" ]; then
    echo make "embeddings/${base%.*}.json"
    python embed.py "$file" "embeddings/${base%.*}.json"
  fi
done