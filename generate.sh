#!/usr/bin/env bash
set -euo pipefail

# Ensure docs directory exists
if [ ! -d docs ]; then
  ./get-pdfs.sh
fi

# Install marker-pdf if not already installed
echo "Installing marker-pdf package..."
pip install marker-pdf

mkdir -p text
for file in docs/*.pdf
do
  base="$(basename "$file")"
  if [ ! -f "text/${base%.*}.txt" ]; then
    echo "Converting $file to text/${base%.*}.txt using marker-pdf CLI with Ollama"
    # Use marker-pdf CLI with --use_llm option for Ollama integration
    # Set default model if GENERATE_MODEL is not set
    MODEL=${GENERATE_MODEL:-"llama3.3"}
    python -m marker_pdf "$file" "text/${base%.*}.md" --use_llm --llm_model "$MODEL" --base_url "http://localhost:11434" --batch_multiplier 2
    
    # Convert the markdown output to plain text
    if [ -f "text/${base%.*}.md" ]; then
      # Extract text content from markdown (remove markdown formatting)
      sed -e 's/^#*\s*//' -e 's/\*\*\(.*\)\*\*/\1/g' -e 's/\*\(.*\)\*/\1/g' -e '/^$/d' "text/${base%.*}.md" > "text/${base%.*}.txt"
      rm "text/${base%.*}.md"  # Clean up intermediate markdown file
      echo "Successfully converted $file to text/${base%.*}.txt"
    else
      echo "Error: marker-pdf failed to create output file"
      # Fallback to pdftotext if marker fails
      echo "Falling back to pdftotext..."
      if command -v pdftotext >/dev/null 2>&1; then
        pdftotext "$file" "text/${base%.*}.txt"
        echo "Fallback: Used pdftotext for $file"
      else
        echo "Error: Both marker-pdf and pdftotext unavailable for $file"
      fi
    fi
  fi
done

## flow v1
# mkdir -p summaries
# for file in text/*.txt
# do
#   base="$(basename "$file")"
#   if [ ! -f "summaries/$base" ]; then
#     echo make "summaries/$base"
#     python summarize.py "$file" "summaries/$base"
#   fi
# done

# mkdir -p embeddings
# for file in summaries/*.txt
# do
#   base="$(basename "$file")"
#   if [ ! -f "embeddings/${base%.*}.json" ]; then
#     echo make "embeddings/${base%.*}.json"
#     python embed.py "$file" "embeddings/${base%.*}.json"
#   fi
# done

## flow v2
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
