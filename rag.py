import json
import subprocess
import sys
import os
import os.path

import ollama
import chromadb


def load_text(*, embed_model: str, embeddings_path: str, text_path: str):
    client = chromadb.Client()
    collection = client.create_collection(name="docs")
    n_loaded = 0
    files = os.listdir(text_path)
    for name in files:
        basename = os.path.splitext(name)[0]
        if (n_loaded/len(files)*100) % 10 == 0:
            print(f"loaded {n_loaded}/{len(files)}.", file=sys.stderr)
        with open(os.path.join(text_path, basename+".txt")) as f:
            text = f.read()
        # result = subprocess.run(["make", json_path], capture_output=True)
        # if result.returncode != 0:
        #     print(result.stdout.decode('utf-8'))
        #     print(result.stderr.decode('utf-8'))
        #     raise RuntimeError("failed to make embedding")
        with open(os.path.join(embeddings_path, basename + ".json")) as f:
            embedding = json.load(f)
            collection.add(
                ids=[name],
                embeddings=embedding,
                documents=text,
            )
        n_loaded += 1
    return collection


def repl(collection, *, embed_model: str, generate_model: str):
    print("> ", end="")
    query = input()
    print("embedding your query...", file=sys.stderr)
    resp = ollama.embed(embed_model, query)
    print("querying text...", file=sys.stderr)
    results = collection.query(
        query_embeddings=resp["embeddings"],
        n_results=1,
    )
    if len(results['ids'][0]) == 0:
        print("no documents found; aborted.", file=sys.stderr)
        return
    # print(results)
    text = results['documents'][0][0]
    docs_list = "\n- ".join(ids[0] for ids in results["ids"])
    print(f"using these documents:\n- {docs_list}", file=sys.stderr)
    print("generating response...", file=sys.stderr)
    output = ollama.generate(
        generate_model,
        prompt=f"Respond to the query at the end using this data: {text}. {query}",
    )
    print(output["response"])
    print("done.", file=sys.stderr)

if __name__ == "__main__":
    generate_model = "llama3.2"
    embed_model = "nomic-embed-text"
    print("loading documents...", file=sys.stderr)
    collection = load_text(
        embed_model=embed_model,
        embeddings_path="embeddings",
        text_path="summaries",
    )
    print("loaded documents.", file=sys.stderr)
    while True:
        repl(collection, embed_model=embed_model, generate_model=generate_model)
