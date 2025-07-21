# RAG-Based Chatbot Tutorial: Complete Code Implementation

This comprehensive tutorial demonstrates how to build a RAG (Retrieval-Augmented Generation) chatbot using the KG-RAG dataset and PACE ICE's compute resources. Unlike the conceptual tutorial, this version provides complete, executable code examples for every step.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Text Processing and Chunking](#text-processing-and-chunking)
4. [Embedding Generation](#embedding-generation)
5. [RAG Implementation](#rag-implementation)
6. [Model Management](#model-management)
7. [Evaluation System](#evaluation-system)
8. [Complete Workflow](#complete-workflow)

## Environment Setup

### Prerequisites
- Python: basic knowledge (for loops, method calls, generators, etc)
- Bash: basic knowledge (for loops, variable substitution, etc)
- Access to [PACE ICE's instance of Open OnDemand](https://ondemand-ice.pace.gatech.edu)
  - If off-campus, use the GT VPN (or in-browser VPN) to connect

### Step 1: Get Started with ICE

We are going to be running LLMs (large language models) on ICE. The easiest way to get started with LLMs on ICE is to go to Open OnDemand, click "Interactive Apps" on the top navbar, and then "Ollama + Jupyter (Beta)."

![Interactive Apps dropdown showing Ollama option](https://github.com/user-attachments/assets/8d4b9deb-90f4-48f6-98b6-c1e6b163cbd0)

#### Configure Ollama Settings

1. **Ollama Models Directory**: Select "Temporary directory" as your Ollama models directory. This directory is where the LLMs will be downloaded to. Since "PACE shared models" does not allow downloading additional models, we will use "Temporary directory" and download the models ourselves.

<img src='https://github.com/user-attachments/assets/1397c880-1892-4987-b718-eb6d1c1eeb48' alt='dropdown for choosing Ollama models directory' width='512'/>

2. **Node Type**: For the node type, select "NVIDIA GPU (first avail)." This will ensure you get a GPU of some sort, while making sure you won't wait too long for a specific GPU to free up.

3. **Other Settings**: The default values can be used for everything else.

#### Launch Your Environment

Once you click submit, you should see a card like the following:

![Card showing queued job status](https://github.com/user-attachments/assets/9503a326-2f65-4dea-b04f-317a125c4d01)

This card will become green once the environment is ready:

![Card showing running job status with Connect to Jupyter button](https://github.com/user-attachments/assets/79c02363-b076-402c-851f-aa118504e6ce)

Click "Connect to Jupyter" to open a new tab with your new environment.

### Step 2: Setting up your Python Environment

```python
import os
import subprocess
```

```
#!/usr/bin/env bash

# run outside of a notebook

python -m venv .venv
.venv/bin/pip install ipykernel
.venv/bin/ipykernel install --user --name=ospo-ai-agent
```

```python
!pip install ollama
!pip install nltk
!pip install numpy
!pip install chromadb
```

### Package Details

Each dependency enables specific capabilities in our RAG system:

#### Ollama
Interface with local LLM models running on ICE infrastructure. This enables running powerful language models (like Llama, Mistral) locally without API costs or internet dependency. Essential for both text generation and embeddings in our RAG system.

#### NLTK (Natural Language Toolkit)
Sentence tokenization and text preprocessing. Breaks documents into meaningful sentences for semantic chunking. Proper sentence boundaries are crucial for maintaining context when splitting text - better than naive character-based splitting that might break mid-sentence.

#### NumPy
Mathematical operations on embedding vectors. Enables efficient cosine similarity calculations between embeddings. Critical for semantic chunking (comparing sentence similarities) and vector database operations. Much faster than pure Python for numerical computations.

#### ChromaDB
Vector database for storing and retrieving document embeddings. Provides fast semantic search capabilities - instead of keyword matching, finds documents that are conceptually similar to queries. Handles the complex vector similarity math automatically and scales to large document collections.


```python
import nltk
import ollama

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('punkt_tab')

# Verify installations
try:
    import chromadb
    import numpy as np
    print("✅ All imports successful!")
    print(f"NumPy version: {np.__version__}")
    print("✅ Environment setup complete!")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

## Data Preparation

### Downloading KG-RAG Dataset

The dataset contains SEC 10-Q filings with corresponding questions and answers for evaluation.

```python
# Download KG-RAG dataset using IPython shell commands
!bash get-pdfs.sh
```

## Text Summarization

### Why Summarization Comes First

Before chunking text, we often need to summarize it because:

1. **Remove Template Text**: Documents often contain headers, legal notices, and formatting that don't add value
2. **Prevent Hallucination**: Based on anecdotal evidence, LLMs are more likely to hallucinate with unnecessary information

### Intelligent Document Summarization

```python
# Document summarization with relevance filtering
import itertools
import json
import ollama

def summarize_document(text_path: str, summary_path: str, 
                      chunk_size=8000, model="llama3.2:3b"):
    """
    Summarize a document by processing it in chunks and filtering relevant content.
    
    Args:
        text_path: Path to input text file
        summary_path: Path to save summary
        chunk_size: Size of text chunks for processing
        model: Model to use for summarization
    """
    with open(text_path, 'r') as f:
        text = f.read()
    
    # Split into manageable chunks
    chunks = list(itertools.batched(text, chunk_size))
    full_summary = ""
    
    print(f"Processing {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        chunk_text = ''.join(chunk)
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        
        # Generate summary
        summary_response = ollama.generate(
            model=model,
            prompt=f"""Make a concise, informative summary of the following text from {text_path}:

{chunk_text}

Requirements:
- Use Markdown formatting
- Focus on key facts and figures
- Be self-contained and clear
- If there's no important information, return an empty summary

Summary:"""
        )
        
        summary = summary_response["response"]
        print(f"Generated summary: {summary[:100]}...")
        
        # Check if summary contains useful information
        relevance_response = ollama.generate(
            model=model,
            prompt=f"""Does the following summary contain useful, specific information (not just formatting or generic statements)?

Summary: {summary}

Respond with JSON format: {{"relevant": true}} or {{"relevant": false}}"""
        )
        
        relevance_text = relevance_response["response"]
        print(f"Relevance check: {relevance_text}")
        
        # Parse relevance (simple heuristic)
        is_relevant = ("true" in relevance_text.lower() and 
                      "false" not in relevance_text.lower())
        
        if is_relevant:
            full_summary += summary + "\n\n"
            print("✓ Summary added to final document")
        else:
            print("✗ Summary skipped (not relevant)")
    
    # Save final summary
    with open(summary_path, 'w') as f:
        f.write(full_summary.strip())
    
    print(f"Complete summary saved to {summary_path}")
    return full_summary

# Example usage
def demo_summarization():
    # Create sample document
    sample_doc = """
    FINANCIAL PERFORMANCE OVERVIEW
    
    Revenue for Q3 2024 reached $89.5 billion, representing a 6% increase compared to Q3 2023.
    iPhone revenue was $46.2 billion, up from $43.8 billion in the prior year quarter.
    Services revenue grew to $22.3 billion, a 12% increase year-over-year.
    
    OPERATING EXPENSES
    
    Research and development expenses were $7.8 billion for the quarter.
    Sales and marketing expenses totaled $1.2 billion.
    General and administrative expenses were $0.6 billion.
    
    GEOGRAPHIC BREAKDOWN
    
    Americas revenue: $37.2 billion
    Europe revenue: $22.5 billion  
    Greater China revenue: $15.1 billion
    Japan revenue: $5.9 billion
    Rest of Asia Pacific revenue: $8.8 billion
    """
    
    with open('sample_financial.txt', 'w') as f:
        f.write(sample_doc)
    
    # Generate summary
    summary = summarize_document('sample_financial.txt', 'sample_summary.txt')
    print("\n--- FINAL SUMMARY ---")
    print(summary)

# Uncomment to run demo
# demo_summarization()
```

## Text Processing and Chunking

### Why Text Chunking is Important

Text chunking is a critical step in RAG systems because:

1. **LLM Context Limits**: Large language models have maximum context lengths (token limits). Even with large context windows, performance often degrades with very long inputs.

2. **Retrieval Precision**: When you search a vector database, you want to retrieve the most relevant pieces of information, not entire documents that may contain irrelevant sections.

3. **Quality Over Quantity**: Based on anecdotal evidence, an LLM is more likely to hallucinate if given a *lot* of unnecessary information. Therefore, to make a good RAG chatbot, we want to give the LLM the minimum amount of information that still results in a good answer.

### How Chunking Works

There are several approaches to chunking text:

- **Fixed-length chunking**: Split text every N characters or words (simple but breaks context)
- **Sentence-based chunking**: Split at sentence boundaries (better context preservation)
- **Semantic chunking**: Split based on topic changes using embedding similarity (optimal context)

### Semantic Chunking Implementation

Semantic chunking uses embedding similarity to determine natural breakpoints in text, creating more coherent chunks than simple length-based splitting. This approach identifies topic changes by measuring the similarity between consecutive sentences.

**Note**: This chunking process is typically applied to the summarized text from the previous step, ensuring we chunk clean, relevant content rather than raw documents with formatting artifacts.

```python
# Semantic chunking implementation
import sys
import ollama
import nltk
import numpy as np

def compare_vec(a, b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(a, b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    return dot_product / mag_a / mag_b

def semantic_chunk(text: str, embed_model='nomic-embed-text', percentile=0.95):
    """
    Split text into semantically coherent chunks using embedding similarity.
    
    Args:
        text: Input text to chunk
        embed_model: Model to use for embeddings
        percentile: Similarity threshold percentile for splitting
    
    Returns:
        List of text chunks (each chunk is a list of sentences)
    """
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    print(f"Processing {len(sentences)} sentences...", file=sys.stderr)
    
    # Generate embeddings for each sentence
    embeddings = []
    for i, sentence in enumerate(sentences):
        if i % 10 == 0:
            print(f"Embedding sentence {i+1}/{len(sentences)}", file=sys.stderr)
        embedding = ollama.embed(embed_model, sentence)['embeddings'][0]
        embeddings.append(embedding)
    
    if len(embeddings) < 2:
        return [sentences]
    
    # Calculate similarity between consecutive sentences
    print(f"Comparing {len(embeddings)} embeddings...", file=sys.stderr)
    deltas = [compare_vec(*x) for x in zip(embeddings, embeddings[1:])]
    
    # Determine splitting threshold
    deltas_sorted = sorted(deltas)
    threshold = deltas_sorted[int(len(deltas_sorted) * percentile)]
    print(f"Similarity threshold ({percentile*100:.2f} percentile): {threshold:.4f}", file=sys.stderr)
    
    # Split at low similarity points
    chunks = []
    last_split = 0
    for i, delta in enumerate(deltas):
        if delta <= threshold:  # Low similarity = good split point
            chunks.append(sentences[last_split:i+1])
            last_split = i + 1
            print(f"Split after sentence {i+1} (similarity: {delta:.4f})")
    
    # Add remaining sentences
    if last_split < len(sentences):
        chunks.append(sentences[last_split:])
    
    print(f"Created {len(chunks)} semantic chunks", file=sys.stderr)
    return chunks

# Example usage
def demo_semantic_chunking():
    sample_text = """
    Apple Inc. reported strong quarterly results. Revenue increased by 15% year-over-year.
    The iPhone segment saw particularly strong growth. Sales in China recovered significantly.
    
    In other news, the company announced a new data center initiative. 
    This represents a major shift in infrastructure strategy.
    The project will require substantial capital investment over the next three years.
    
    Looking ahead, management remains optimistic about growth prospects.
    However, they cautioned about potential supply chain disruptions.
    """
    
    chunks = semantic_chunk(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(' '.join(chunk))

# Uncomment to run demo
# demo_semantic_chunking()
```

## Embedding Generation

### What are Embeddings and Why Use Them?

Embeddings are numerical vector representations of text that capture semantic meaning. In a RAG system, embeddings enable:

1. **Semantic Search**: Instead of exact keyword matching, you can find documents that are conceptually related to your query
2. **Vector Database Storage**: Embeddings allow fast similarity searches using mathematical operations like cosine similarity
3. **Context Understanding**: Similar concepts have similar embeddings, even if they use different words

### How Vector Databases Work

A vector database stores and retrieves text by its embedding vector, generated by an embedding model like `nomic-embed-text`. The simplified workflow (in pseudocode) is:

```
embed(text: str) -> List[float]                         # Convert text to numbers
vector_db_store(id: str, vector: List[float])           # Store with identifier
vector_db_query(vector: List[float]) -> List[Document]  # Find similar
```

For example, storing information about pies (in pseudocode)
```
vector_db_store('pumpkin_pie', embed('pumpkin pie'))
vector_db_store('apple_pie', embed('apple pie'))
vector_db_store('elderberry_pie', embed('elderberry pie'))
```

When a user asks about "pumpkins", running `vector_db_query(embed('pumpkins'))` would return `'pumpkin_pie'` as the top result because the embeddings are mathematically similar.

### Simple Embedding Approach

```python
# Basic embedding generation
import json
import ollama

def generate_embedding(text_path: str, embedding_path: str, model='nomic-embed-text'):
    """
    Generate embeddings for a text file.
    
    Args:
        text_path: Path to input text file
        embedding_path: Path to save embedding JSON
        model: Embedding model to use
    """
    try:
        with open(text_path, 'r') as f:
            text = f.read()
        
        print(f"Generating embedding for {text_path}...")
        resp = ollama.embed(model, text)
        
        with open(embedding_path, 'w') as f:
            json.dump(resp["embeddings"], f)
        
        print(f"Embedded {text_path} -> {embedding_path}")
        return True
        
    except Exception as e:
        print(f"Error embedding {text_path}: {e}")
        return False

# Example usage
def demo_embedding():
    # Create sample text file
    sample_text = "Apple Inc. is a technology company that designs consumer electronics."
    with open('sample.txt', 'w') as f:
        f.write(sample_text)
    
    # Generate embedding
    success = generate_embedding('sample.txt', 'sample_embedding.json')
    
    if success:
        # Load and examine embedding
        with open('sample_embedding.json', 'r') as f:
            embedding = json.load(f)
        print(f"Embedding dimensions: {len(embedding[0])}")
        print(f"First 5 values: {embedding[0][:5]}")

# Uncomment to run demo
# demo_embedding()
```

## RAG Implementation

### Simple RAG System

Using global variables and functions for clarity in notebook environments:

```python
# Simple RAG implementation with global variables
import json
import os
import time
import ollama
import chromadb

# Global configuration
EMBED_MODEL = "nomic-embed-text"
GENERATE_MODEL = "llama3.3"

# Global database connection
client = chromadb.Client()
collection = None

def load_documents_from_embeddings(embeddings_path: str):
    """Load documents from pre-computed embeddings directory"""
    global collection
    collection = client.create_collection(name="docs")
    
    files = os.listdir(embeddings_path)
    files = [name for name in files if name[0] != '.' and not name.endswith(".marker")]
    
    n_loaded = 0
    last_print = time.time()
    
    for name in files:
        now = time.time()
        if now - last_print > 1:
            print(f"Loaded {n_loaded}/{len(files)} documents")
            last_print = now
        
        try:
            with open(os.path.join(embeddings_path, name), 'r') as f:
                data = json.load(f)
                collection.add(
                    ids=[name],
                    embeddings=data['vector'][0],
                    documents=data['chunk_text'],
                )
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
        
        n_loaded += 1
    
    print(f"Successfully loaded {n_loaded} documents")
    return collection

def load_documents_from_text(embeddings_path: str, text_path: str):
    """Load documents from text files with corresponding embeddings"""
    global collection
    collection = client.create_collection(name="docs")
    
    files = os.listdir(text_path)
    files = [name for name in files if name[0] != '.']
    
    n_loaded = 0
    for name in files:
        basename = os.path.splitext(name)[0]
        
        if (n_loaded / len(files) * 100) % 10 == 0:
            print(f"Loaded {n_loaded}/{len(files)} documents")
        
        try:
            # Load text
            with open(os.path.join(text_path, basename + ".txt"), 'r') as f:
                text = f.read()
            
            # Load corresponding embedding
            with open(os.path.join(embeddings_path, basename + ".json"), 'r') as f:
                embedding = json.load(f)
            
            collection.add(
                ids=[name],
                embeddings=embedding,
                documents=text,
            )
            n_loaded += 1
            
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
    
    print(f"Successfully loaded {n_loaded} documents")
    return collection

def generate_retrieval_query(user_query: str):
    """Generate optimized search query from user input"""
    response = ollama.generate(
        GENERATE_MODEL,
        prompt=f"""You are a search agent. Transform this user query into optimal keywords for document retrieval.

User Query: {user_query}

Return ONLY the search keywords, nothing else - no explanation or additional text."""
    )
    return response["response"]

def retrieve_documents(query: str, n_results=8):
    """Retrieve relevant documents using vector similarity"""
    if not collection:
        raise RuntimeError("No documents loaded. Call load_documents_* first.")
    
    # Generate retrieval query
    retrieval_query = generate_retrieval_query(query)
    print(f"Search query: {retrieval_query}")
    
    # Get query embedding
    embedding_response = ollama.embed(EMBED_MODEL, retrieval_query)
    
    # Query vector database
    results = collection.query(
        query_embeddings=embedding_response["embeddings"],
        n_results=n_results,
    )
    
    if len(results['ids'][0]) == 0:
        raise RuntimeError("No documents retrieved")
    
    return results

def filter_relevant_documents(query: str, results):
    """Filter documents based on relevance to query"""
    texts = results['documents'][0]
    ids = results['ids'][0]
    
    print("Filtering documents for relevance:")
    relevant_docs = []
    
    for i, (doc_id, text) in enumerate(zip(ids, texts)):
        # Check if document contributes to answering the query
        relevance_response = ollama.generate(
            GENERATE_MODEL,
            prompt=f"""Analyze if this document helps answer the user's question.

User Question: {query}
Document ID: {doc_id}

Does this document contain information that significantly contributes to answering the question?
Answer with: YES or NO"""
        )
        
        if "YES" in relevance_response["response"]:
            relevant_docs.append(text)
            print(f"✓ {doc_id}")
        else:
            print(f"✗ {doc_id}")
    
    return relevant_docs

def ask_question(query: str, use_filtering=True):
    """Answer a question using RAG"""
    # Retrieve relevant documents
    results = retrieve_documents(query)
    
    # Get document texts
    if use_filtering:
        relevant_texts = filter_relevant_documents(query, results)
    else:
        relevant_texts = results['documents'][0]
    
    if not relevant_texts:
        return "I couldn't find relevant information to answer your question."
    
    # Generate answer using retrieved context
    context = '\n'.join(relevant_texts)
    print("Generating answer...")
    
    response = ollama.generate(
        GENERATE_MODEL,
        prompt=f"""Answer the user's question using the provided context. Be accurate and specific.

Context:
{context}

Question: {query}

Answer:"""
    )
    
    return response["response"]

def start_interactive_session():
    """Start interactive question-answering session"""
    print("RAG System ready! Type 'quit' to exit.")
    
    while True:
        try:
            query = input("\n> ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            answer = ask_question(query)
            print(f"\n{answer}\n")
            
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

# Example usage
def demo_rag_system():
    # Create some sample documents and embeddings for demo
    sample_docs = [
        ("doc1.json", {
            "vector": [[0.1, 0.2, 0.3, 0.4, 0.5]],  # Simplified embedding
            "chunk_text": "Apple Inc. reported revenue of $89.5 billion in Q3 2024, up 6% year-over-year. iPhone sales contributed $46.2 billion."
        }),
        ("doc2.json", {
            "vector": [[0.2, 0.3, 0.4, 0.5, 0.6]],
            "chunk_text": "Operating expenses included $7.8 billion for R&D and $1.2 billion for sales and marketing."
        }),
        ("doc3.json", {
            "vector": [[0.3, 0.4, 0.5, 0.6, 0.7]],
            "chunk_text": "Revenue breakdown by geography: Americas $37.2B, Europe $22.5B, Greater China $15.1B."
        })
    ]
    
    # Create sample embedding files
    os.makedirs("sample_embeddings", exist_ok=True)
    for filename, data in sample_docs:
        with open(f"sample_embeddings/{filename}", 'w') as f:
            json.dump(data, f)
    
    print("Sample documents created!")
    print("To use RAG system:")
    print("1. Load documents: load_documents_from_embeddings('embed2')")
    print("2. Ask questions: ask_question('What was the revenue?')")
    print("3. Start interactive mode: start_interactive_session()")

# Run demo
demo_rag_system()
```

## Model Management

### Automated Model Downloads

```python
# Model management system
import ollama
import subprocess

def list_available_models():
    """List currently available models"""
    try:
        models = ollama.list()["models"]
        print("Currently available models:")
        for model in models:
            print(f"  - {model.model}")
        return [model.model for model in models]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def download_models():
    """Download essential models for RAG system"""
    # Essential models for RAG
    models_to_download = [
        'nomic-embed-text',    # Embedding model
        'llama3.1:8b',        # Large general model
        'llama3.2:1b',        # Small fast model
        'llama3.2:3b',        # Medium model (good balance)
        'llama3.3',           # Latest model
        'mistral',            # Alternative architecture
        'qwen2.5:0.5b',       # Very small model for testing
        'qwen2.5:1.5b',       # Small efficient model
    ]
    
    print("Downloading essential models...")
    successful_downloads = []
    failed_downloads = []
    
    for model in models_to_download:
        try:
            print(f"\nDownloading {model}...")
            ollama.pull(model)
            successful_downloads.append(model)
            print(f"✓ {model} downloaded successfully")
        except Exception as e:
            failed_downloads.append((model, str(e)))
            print(f"✗ Failed to download {model}: {e}")
    
    print(f"\n--- Download Summary ---")
    print(f"Successful: {len(successful_downloads)}")
    print(f"Failed: {len(failed_downloads)}")
    
    if successful_downloads:
        print("\nSuccessfully downloaded:")
        for model in successful_downloads:
            print(f"  ✓ {model}")
    
    if failed_downloads:
        print("\nFailed downloads:")
        for model, error in failed_downloads:
            print(f"  ✗ {model}: {error}")

def get_model_info(model_name: str):
    """Get detailed information about a model"""
    try:
        # Test if model is available
        models = ollama.list()["models"]
        available_models = [m.model for m in models]
        
        if model_name not in available_models:
            print(f"Model {model_name} is not available. Available models:")
            for model in available_models:
                print(f"  - {model}")
            return None
        
        # Test model with a simple query
        test_response = ollama.generate(
            model_name, 
            prompt="Say 'Hello' if you're working correctly.",
            options={"num_predict": 10}
        )
        
        print(f"Model {model_name} is working correctly.")
        print(f"Test response: {test_response['response']}")
        
        return True
        
    except Exception as e:
        print(f"Error testing model {model_name}: {e}")
        return False

# Example usage
def demo_model_management():
    print("=== Model Management Demo ===")
    
    # List current models
    current_models = list_available_models()
    
    # Download models (uncomment to actually download)
    print("\n=== Download Models ===")
    # download_models()
    print("[Download simulation - uncomment download_models() to actually download]")
    
    # Test a model
    print("\n=== Model Testing ===")
    if current_models:
        test_model = current_models[0]
        print(f"Testing model: {test_model}")
        # get_model_info(test_model)
        print(f"[Would test {test_model} if available]")

# Uncomment to run demo
# demo_model_management()
```

## Image Processing

### Vision Model Integration

```python
# Image-to-text conversion
import sys
import os
import ollama

def image_to_text(image_path: str, model='llava:7b'):
    """
    Convert an image to descriptive text using a vision model.
    
    Args:
        image_path: Path to the image file
        model: Vision model to use (e.g., 'llava:7b', 'gemma3')
    
    Returns:
        String description of the image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Initial detailed description
        messages = [
            {
                'role': 'user', 
                'content': 'Describe this image in detail, focusing on any text, charts, or important visual elements.',
                'images': [image_path]
            }
        ]
        
        print("Generating detailed description...", file=sys.stderr)
        response1 = ollama.chat(model=model, messages=messages)
        messages.append(response1.message)
        
        # Request concise summary
        messages.append({
            'role': 'user',
            'content': 'Create a concise, informative summary of the image in one paragraph. Focus on the most important information without introductory phrases.'
        })
        
        print("Creating summary...", file=sys.stderr)
        response2 = ollama.chat(model=model, messages=messages)
        
        # Debug output
        for message in messages:
            print(f"Message: {message}", file=sys.stderr)
        
        return response2.message.content
        
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        return None

def process_image_dataset(image_directory: str, output_directory: str, model='llava:7b'):
    """
    Process a directory of images, converting each to text descriptions.
    
    Args:
        image_directory: Directory containing images
        output_directory: Directory to save text descriptions
        model: Vision model to use
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    image_files = []
    for filename in os.listdir(image_directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    print(f"Found {len(image_files)} image files to process")
    
    processed = 0
    errors = 0
    
    for filename in image_files:
        image_path = os.path.join(image_directory, filename)
        output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.txt")
        
        print(f"Processing {filename}... ({processed+1}/{len(image_files)})")
        
        try:
            description = image_to_text(image_path, model)
            if description:
                with open(output_path, 'w') as f:
                    f.write(f"# Image Description: {filename}\n\n")
                    f.write(description)
                processed += 1
                print(f"✓ Saved description to {output_path}")
            else:
                print(f"✗ Failed to generate description for {filename}")
                errors += 1
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            errors += 1
    
    print(f"\n--- Processing Summary ---")
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Total: {len(image_files)}")

# Example usage
def demo_image_processing():
    # Create a sample scenario
    print("=== Image Processing Demo ===")
    print("This demo shows how to convert images to text descriptions.")
    print("\nTo use this functionality:")
    print("1. Ensure you have a vision model installed (e.g., 'ollama pull llava:7b')")
    print("2. Place your images in a directory")
    print("3. Run the processing function")
    
    # Simulated usage example
    print("\nExample usage:")
    print("```python")
    print("# Process a single image")
    print("description = image_to_text('path/to/image.jpg')")
    print("print(description)")
    print()
    print("# Process entire directory")
    print("process_image_dataset('images/', 'text_descriptions/')")
    print("```")
    
    # If you have an actual image, uncomment below:
    # description = image_to_text('sample_image.jpg')
    # print(f"Description: {description}")

# Uncomment to run demo
# demo_image_processing()
```

## Chatbot Example

The following demonstrates how to make a simple REPL (Read-Eval-Print Loop) for the RAG system, combining all the components we've built throughout this tutorial.

```python
# Complete RAG chatbot implementation
import os
import sys
import time
import ollama
import chromadb

# Configuration
EMBED_MODEL = "nomic-embed-text" 
GENERATE_MODEL = os.environ.get("GENERATE_MODEL", "llama3.3")

# Global state
client = chromadb.Client()
collection = None

def initialize_rag_system(embeddings_path="embed2"):
    """Initialize the RAG system with document embeddings"""
    global collection
    
    print("🚀 Initializing RAG system...")
    
    # Create collection
    collection = client.create_collection(name="financial_docs")
    
    # Load documents
    if not os.path.exists(embeddings_path):
        print(f"❌ Error: Embeddings directory '{embeddings_path}' not found!")
        print("Please run the document processing pipeline first:")
        print("1. Run: ./get-pdfs.sh")  
        print("2. Run: ./generate.sh")
        return False
    
    files = [f for f in os.listdir(embeddings_path) 
             if f.endswith('.json') and not f.endswith('.marker')]
    
    if not files:
        print(f"❌ No embedding files found in {embeddings_path}")
        return False
    
    print(f"📚 Loading {len(files)} documents...")
    n_loaded = 0
    
    for filename in files:
        try:
            with open(os.path.join(embeddings_path, filename), 'r') as f:
                data = json.load(f)
                collection.add(
                    ids=[filename],
                    embeddings=data['vector'][0] if 'vector' in data else data['embedding'],
                    documents=data.get('chunk_text', data.get('text', 'No content')),
                )
                n_loaded += 1
        except Exception as e:
            print(f"⚠️  Warning: Failed to load {filename}: {e}")
    
    print(f"✅ Successfully loaded {n_loaded} documents")
    return n_loaded > 0

def search_documents(query, n_results=5):
    """Search for relevant documents using vector similarity"""
    if not collection:
        raise RuntimeError("RAG system not initialized. Call initialize_rag_system() first.")
    
    # Generate search query
    search_response = ollama.generate(
        GENERATE_MODEL,
        prompt=f"""Transform this user question into effective search keywords for a financial document database.

User Question: {query}

Return only the search keywords, no explanation:"""
    )
    search_query = search_response["response"]
    
    print(f"🔍 Search query: {search_query}")
    
    # Get embeddings and search
    embedding = ollama.embed(EMBED_MODEL, search_query)
    results = collection.query(
        query_embeddings=embedding["embeddings"],
        n_results=n_results
    )
    
    return results

def answer_question(query):
    """Generate answer using RAG approach"""
    try:
        # Search for relevant documents
        results = search_documents(query)
        
        if not results['documents'][0]:
            return "❌ I couldn't find any relevant documents to answer your question."
        
        # Show which documents are being used
        doc_ids = results['ids'][0]
        print(f"📄 Using documents: {', '.join(doc_ids[:3])}{'...' if len(doc_ids) > 3 else ''}")
        
        # Combine relevant documents
        context = '\n---\n'.join(results['documents'][0])
        
        # Generate answer
        print("🤔 Generating answer...")
        answer_response = ollama.generate(
            GENERATE_MODEL,
            prompt=f"""Answer the user's question based on the provided financial document excerpts. Be accurate and cite specific figures when available.

Document Context:
{context}

User Question: {query}

Answer (be specific and factual):"""
        )
        
        return answer_response["response"]
        
    except Exception as e:
        return f"❌ Error processing question: {str(e)}"

def start_chatbot():
    """Start the interactive RAG chatbot"""
    print("=" * 60)
    print("🤖 RAG CHATBOT - Financial Document Q&A")
    print("=" * 60)
    
    # Initialize system
    if not initialize_rag_system():
        print("Failed to initialize RAG system. Exiting.")
        return
    
    print("\n✨ RAG system ready!")
    print("💡 Try asking questions like:")
    print("   • What was the revenue for Q3 2024?")
    print("   • How much did the company spend on R&D?")
    print("   • What were the operating expenses?")
    print("   • What was the revenue breakdown by region?")
    print("\n💬 Type your questions below (or 'quit' to exit)")
    print("-" * 60)
    
    conversation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n🧑 Question #{conversation_count + 1}: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 Thanks for using the RAG chatbot!")
                break
            
            if not user_input:
                print("Please enter a question.")
                continue
            
            # Process question and generate answer
            start_time = time.time()
            answer = answer_question(user_input)
            end_time = time.time()
            
            # Display answer
            print(f"\n🤖 Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            print(f"⏱️  Response time: {end_time - start_time:.1f}s")
            
            conversation_count += 1
            
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or type 'quit' to exit.")

# Quick test function
def test_rag_system():
    """Test the RAG system with sample questions"""
    print("🧪 Testing RAG system...")
    
    if not initialize_rag_system():
        print("Cannot test - initialization failed")
        return
    
    test_questions = [
        "What was the total revenue?",
        "How much was spent on research and development?",
        "What were the main operating expenses?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}: {question} ---")
        answer = answer_question(question)
        print(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")

# Example usage
if __name__ == "__main__":
    # Uncomment one of these to run:
    
    # Start interactive chatbot
    start_chatbot()
    
    # Or run quick test
    # test_rag_system()
```

### Usage Instructions

#### Running the Chatbot

1. **Prepare Your Data:**
   ```bash
   # Download dataset
   ./get-pdfs.sh
   
   # Process documents (with marker-pdf and Ollama)
   ./generate.sh
   ```

2. **Start the Chatbot:**
   ```python
   # Run the chatbot directly
   start_chatbot()
   
   # Or test with sample questions first
   test_rag_system()
   ```

3. **Example Interaction:**
   ```
   🤖 RAG CHATBOT - Financial Document Q&A
   ============================================================
   
   🧑 Question #1: What was Apple's revenue in Q3 2024?
   🔍 Search query: Apple revenue Q3 2024 earnings financial results
   📄 Using documents: apple_10q_q3_2024.txt.json, financial_summary.txt.json
   🤔 Generating answer...
   
   🤖 Answer:
   ----------------------------------------
   Apple's revenue for Q3 2024 was $89.5 billion, representing a 6% increase compared to Q3 2023. This growth was driven primarily by iPhone sales which contributed $46.2 billion to the total revenue.
   ----------------------------------------
   ⏱️  Response time: 3.2s
   ```

### Key Features

- **Interactive REPL**: Continuous question-answering loop
- **Smart Document Search**: Uses embeddings to find relevant information
- **Context Display**: Shows which documents are being referenced
- **Performance Metrics**: Displays response times
- **Error Handling**: Graceful handling of failures and edge cases
- **Easy Testing**: Built-in test mode for validation

### Customization Options

```python
# Adjust search parameters
results = search_documents(query, n_results=10)  # Get more documents

# Use different models
EMBED_MODEL = "nomic-embed-text"
GENERATE_MODEL = "llama3.1:8b"  # Use different generation model

# Modify prompt for different behavior
prompt = f"Answer briefly and technically: {query}\nContext: {context}"
```

This chatbot implementation provides a complete, production-ready interface for interacting with your RAG system, bringing together all the components from document processing through to natural language generation.
