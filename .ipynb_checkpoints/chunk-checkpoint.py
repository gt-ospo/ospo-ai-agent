import sys

import ollama
import nltk
import numpy as np


#assert nltk.download('punkt_tab')


def compare_vec(a, b):
    dot_product = np.dot(a, b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    return dot_product / mag_a / mag_b

def semantic_chunk(text: str, embed_model='nomic-embed-text', percentile=0.95):
    sentences = nltk.sent_tokenize(text)
    print(f"embedding {len(sentences)} sentences...", file=sys.stderr)
    embeddings = [ollama.embed(embed_model, sentence)['embeddings'][0] for sentence in sentences]
    if len(embeddings) < 2:
        return [sentences]
    print(f"comparing {len(embeddings)} embeddings...", file=sys.stderr)
    deltas = [compare_vec(*x) for x in zip(embeddings, embeddings[1:])]
    deltas_sorted = sorted(deltas)
    threshold = deltas_sorted[int(len(deltas_sorted)*percentile)]
    print(f"threshold ({percentile*100:.2f} percentile) is {threshold}", file=sys.stderr)
    chunks = []
    last_split = 0
    for i, delta in enumerate(deltas):
        if delta >= threshold:
            chunks.append(sentences[last_split:i])
            last_split = i
            print(f"split on {i}")
    if last_chunk := sentences[last_split:]:
        chunks.append(last_chunk)
    print(f"split into {len(chunks)} chunks.", file=sys.stderr)
    return chunks

if __name__ == '__main__':
    text = sys.stdin.read()
    chunks = semantic_chunk(text)
    for chunk in result:
        print(chunk)
        print()