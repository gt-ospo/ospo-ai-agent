import itertools
import json
import os
import sys
import hashlib

import ollama

import chunk


TEXT_PATH = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'nomic-embed-text')

def sha256(text: str):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

with open(TEXT_PATH) as f:
    text = f.read()
chunks = chunk.semantic_chunk(text, embed_model=EMBED_MODEL)
for i, sentences in enumerate(chunks):
    chunk_text = f"From {TEXT_PATH}:" + ' '.join(sentences)
    h = sha256(chunk_text)
    output_path = os.path.join(OUTPUT_DIR, h)
    if os.path.isfile(output_path):
        print(f"{TEXT_PATH} chunk {i} already generated; skipping.")
        continue
    e = ollama.embed(EMBED_MODEL, chunk_text)['embeddings']
    data = dict(
        text_path=TEXT_PATH,
        chunk_index=i,
        chunk_text=chunk_text,
        vector=e,
    )
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"{TEXT_PATH} chunk {i} done.")