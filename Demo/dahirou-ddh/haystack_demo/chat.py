"""
Query the Neo4j RAG pipeline:

    python chat.py "insert prompt here"

or run `python chat.py` for an interactive REPL.
"""
import sys, textwrap
from neo4j import GraphDatabase
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.dataclasses import Document

import config as C, llm

driver   = GraphDatabase.driver(C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))

embedder = SentenceTransformersDocumentEmbedder(model=C.EMBED_MODEL)
embedder.warm_up() 

def retrieve(query: str, k=C.TOP_K):
    docs = [Document(content=query)]
    emb_out = embedder.run(documents=docs)
    vec = emb_out["documents"][0].embedding

    cypher = (
        "CALL db.index.vector.queryNodes($index,$k,$vec) "
        "YIELD node, score "
        "RETURN node.id AS id, node.content AS content "
        "ORDER BY score DESC"
    )
    with driver.session() as s:
        return list(s.run(cypher, index=C.INDEX_NAME, k=k, vec=vec))


def build_prompt(passages, question):
    ctx_lines = []
    for r in passages:
        snippet = r["content"].strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        ctx_lines.append(f"[{r['id']}] {snippet}")
    context = "\n".join(ctx_lines)
    return textwrap.dedent(f"""{C.SYSTEM_PROMPT}
    Context:
    {context}
    User: {question}
    Assistant:""")


def ask(question: str):
    hits = retrieve(question)
    if not hits:
        print("No context found.")
        return

    prompt = build_prompt(hits, question)
    answer = llm.llm_gen(prompt).strip()
    print("\nASSISTANT: ", answer)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        print("RAG chat: ")
        while True:
            q = input("YOU: ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                ask(q)
