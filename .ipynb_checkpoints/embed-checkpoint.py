import itertools
import json
import sys

import ollama


text_path = sys.argv[1]
embedding_path = sys.argv[2]
with open(text_path) as f:
    text = f.read()
resp = ollama.embed('nomic-embed-text', text)
with open(embedding_path, "w") as f:
    json.dump(resp["embeddings"], f)
print(f"embedded {text_path} to {embedding_path}.", file=sys.stderr)
