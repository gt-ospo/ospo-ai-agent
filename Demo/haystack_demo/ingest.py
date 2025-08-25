"""
Usage:
    python ingest.py path/to/my.pdf
Creates/updates nodes (:Document {id, content, embedding}) 
and the native vector index in Neo4j.
"""
import sys, uuid, logging
from pathlib import Path
from neo4j import GraphDatabase
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.dataclasses import Document
import config as C

logging.basicConfig(level=logging.INFO)

def main(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    conv  = PyPDFToDocument()
    split = DocumentSplitter(split_by="sentence", split_length=8, split_overlap=2)
    split.warm_up()
    docs  = split.run(documents=conv.run([str(pdf_path)])["documents"])["documents"]
    
    for d in docs:
        d.id = uuid.uuid5(uuid.NAMESPACE_URL, d.content).hex

    logging.info("Created %d chunks", len(docs))

    embedder = SentenceTransformersDocumentEmbedder(model=C.EMBED_MODEL)
    embedder.warm_up()
    docs = embedder.run(documents=docs)["documents"]

    driver = GraphDatabase.driver(C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
    with driver.session() as s:
        s.run(f"""
            CREATE VECTOR INDEX {C.INDEX_NAME} IF NOT EXISTS
            FOR (d:Document)
            ON (d.embedding)
            OPTIONS {{
                indexConfig: {{
                `vector.dimensions`: {C.EMBED_DIM},
                `vector.similarity_function`: 'cosine'
                }}
            }}
        """)
        tx = s.begin_transaction()
        for d in docs:
            tx.run(
                """
                MERGE (doc:Document {id: $id})
                SET doc.content = $content,
                    doc.embedding = $emb
                """,
                id=d.id,
                content=d.content,
                emb=d.embedding
            )
        tx.commit()
    driver.close()
    logging.info("Ingest complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py my.pdf"); sys.exit(1)
    main(Path(sys.argv[1]))
