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
    files = [name for name in files if name[0] != '.']
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


def ask(query: str, collection, *, embed_model: str, generate_model: str):
    retrieval_query = ollama.generate(
        generate_model,
        prompt=f"You are a search agent. Below is a search query from a user.\n> {query}\nWhat keywords are the best for finding information for the query? Return ONLY the queries, and nothing else - no explanation.",
    )["response"]
    print(f"retrieval_query={retrieval_query}")
    resp = ollama.embed(embed_model, retrieval_query)
    results = collection.query(
        query_embeddings=resp["embeddings"],
        n_results=8,
    )
    if len(results['ids'][0]) == 0:
        raise RuntimeError("no documents retrieved")
    docs_list = "\n- ".join(ids for ids in results["ids"][0])
    print(f"using these documents:\n- {docs_list}", file=sys.stderr)
    texts = results['documents'][0]
    print("using these documents (filtered):")
    texts2 = []
    for i, text in enumerate(texts):
        id = results["ids"][0][i]
        text_ = "> " + text.replace("\n", "\n> ")
        helpful_output = ollama.generate(
            generate_model,
            prompt=f"As a search agent, you are helping someone answer their question. \n\nQuery: {query}\n\nName of Text:\n{id}\n\nDoes the text significantly contribute towards the answer? If no, write NO. If yes, write YES."
        )["response"]
        if "NO" in helpful_output:
            continue
        # prompt = f"As a search agent, you are helping someone answer their question. \n\nQuery: {query}\n\nText:\n{text_}\n\nDoes the text significantly contribute towards the answer? If no, write NO. If yes, then how?"
        # helpful_output = ollama.generate(
        #     generate_model,
        #     prompt,
        # )["response"]
        # #print(prompt)
        # #print(helpful_output)
        # #print("-"*30)
        # if "NO" in helpful_output:
        #     continue
        print(f"- {id}")
        texts2.append(helpful_output)
    print("answering...")
    output = ollama.generate(
        generate_model,
        prompt=f"Respond to the query using this data: {query}\n {'\n'.join(texts2)}. Respond to this query: {query}",
    )
    return output["response"]


def repl(collection, *, embed_model: str, generate_model: str):
    print("> ", end="")
    query = input()
    print(ask(query, collection, embed_model=embed_model, generate_model=generate_model))

if __name__ == "__main__":
    generate_model = os.environ.get("MODEL", "llama3.3")
    embed_model = "nomic-embed-text"
    print("loading documents...", file=sys.stderr)
    collection = load_text(
        embed_model=embed_model,
        embeddings_path="embeddings",
        text_path="summaries",
    )
    print("loaded documents.", file=sys.stderr)
    while True:
        try:
            repl(collection, embed_model=embed_model, generate_model=generate_model)
        except EOFError:
            print("bye.")
            break
