from pathlib import Path
import textwrap

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "foreclosedender"
INDEX_NAME = "documentEmbedding"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  
EMBED_DIM = 768                                        

GGUF_PATH = Path("/home/hice1/dharden7/deepsearch/models/llama3-8b"
                       "/meta-llama-3-8b-instruct.Q4_K_M.gguf")
N_CTX = 4096
N_THREADS = 16
N_GPU_LAYERS = -1  

TOP_K = 8
MAX_TOKENS = 256

SYSTEM_PROMPT  = textwrap.dedent("""\
    You are a research assistant. Answer the user’s question **only** from the
    text in “Context”. Quote the exact phrase when you’re certain it appears.
    Keep the reply concise and cite the chunk id in brackets.
""")
