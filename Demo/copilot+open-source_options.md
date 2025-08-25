# Copilot capabilities and evaluation against open-source models

## Microsoft 365 Copilot Declarative Agents

**_Pros_**
- For students at GT, the entire Microsoft 365 Copilot service is free.
- Declarative agents use GPT-4 backend.
- **No external vector DB required:**: Microsoft Copilot Agents use SharePoint content retrieval for prompt context (plus other web search and personal file access tools).  [^6]
- **Ease of use:**: The preference to use OneDrive/SharePoint and all other tools can be easily configured in the Declarative Agent manifest[^7]. All documentation for declarative agents is accessible and easy to understand.
- **Tailor to specific research:** Agents can be tailored to provide specific types of responses and/or search data for specific information[^8].
- **Strong privacy:** All data is kept within the Microsoft Service boundary, encrypted of course[^11].
  -  The Retrieval API takes into account controls and permissions metadat - Copilot can only access data the user invoking the API has permission to access.
For what it's worth, Microsoft says user data (prompts, responses, data accessed through API) is not used to train foundational models.

  
**_Cons_**
- **No internal vector DB support**: The API exposes CRUD-style (Create Read Update Delete) endpoints over Microsoft 365 data (users, files, mail, calendar, SharePoint, Teams, etc.) via json over https. However, it does not provide any built-in support for embeddings, nearest-neighbor search, or similarity ranking. Thus we'd need
- **Model flexibility**: Under the covers, declarative agents run on the same Copilot “foundation models” and orchestrator that power Microsoft 365 Copilot itself—Microsoft’s proprietary, OpenAI-derived models (largely GPT-4 variants and internally-trained “Phi” models) and the Copilot service infrastructure. Therefore, we aren't able to pick or swap models when building a declarative agent—the agent simply inherits whatever model Copilot is currently running [^1].
  > “Declarative agents run on the same orchestrator, foundation models, and trusted AI services that power Microsoft 365 Copilot[^2].”
  
  > “…Declarative agents use Copilot’s AI infrastructure, model, and orchestrator…[^3]”
  
  If we want to use a different backend model, we'd have to use custom engine agents, which let us supply our own orchestrator and LLMs, but also require us to handle hosting, compliance, and the responsible-AI considerations ourselves[^4].
- **OneDrive/SharePoint limitations** 
  > "Only web browsing, Copilot connectors, SharePoint sites, folders, and files can be specified as knowledge sources. You can upload your local folders and files into SharePoint[^5]."
  
  Therefore, we may need a pipeline to mass ingest files directly into Sharepoint. Also, it is unclear whether it is possible to point an agent to a single shared directory (avoid copying all files in a database for each user). For instance, we know it is possible to point an agent to specific files:
  
  > "You can reference specific SharePoint files via the OneDriveAndSharePoint object in the agent manifest, either by URL or by ID[^6]."

**_General notes_**
* There is a 20 file context limit using SharePoint. You can choose up to 20 relevant files that your agent can search fully. After 20 files, the agent decides which files are most relevant and searches those[^6]. It is important to note it is **not** required that you choose specific files - it just speeds up the searching process and gaurentees better results.
* **Specifications on context retrieval**: Agents retrieve context from OneDrive/SharePoint in the following stages:
  1. <ins>Pre-computed semantic indexing</ins>:
Microsoft 365 Copilot automatically builds a tenant-level semantic index over your Graph content (including OneDrive and SharePoint files). This index maps every file into a vector space—Microsoft says it can scale “to search through billions of vectors” while respecting all your existing permissions. That index is kept up to date in near-real-time for mailbox items and nightly for shared SharePoint files, so when your agent runs it never has to re-scan raw file contents on the fly[^9].
  2. <ins>Highly optimized vector/keyword retrieval</ins>:
Under the covers, the Retrieval API (and Copilot’s built-in RAG engine) uses an Azure AI Search-style service to perform hybrid vector + keyword search over that index. In independent benchmarks across millions of documents, Azure AI Search achieves median latencies well under 100 ms—for example, in a 7.5 million-document index, the 50th percentile latency on an S2 tier was 59 ms, and even under heavy load (50 % of max QPS) stayed under 123 ms. With replicas added, throughput scales almost linearly, so we could support thousands of concurrent searches simply by bumping our replica count[^10]. However, for the needs of students, this is more than enough.


## LangChain + Open Source 
**_Pros_**
- **_Flexibility_**: LangChain excels at pairing LLMs with external tools (APIs, databases, custom logic).
   - LangChain can easily interface with vector DBs and includes native vector store wrappers for Qdrant, Weaviate, Milvus and more.
   - We can pick from a variety of tools we might need and chain them in custom sequences. This is ideal if we want a research agent assistant that incorporates capabilities running code or doing math[^12].
- **_Multi-Step reasoning_** (native ReAct agent support): LangChain’s agent system is built-in and . The LLM can be prompted to decide which action or tool to use at each step (using frameworks like ReAct or MRKL), enabling dynamic retrieval and reasoning loops. For example, an agent could decide to search the vector DB multiple times with refined queries, or consult a citation database, before formulating the final answer. This aligns perfectly with the project’s need for multi-step planning and tool use. LangChain essentially acts as the “conductor” ensuring each step happens in the right order. To lay it out more clearly - it’s an orchestration layer that repeatedly calls into the (same) underlying LLM, interleaving “think” (reasoning) steps with “act” (tool‐ or vector-DB calls) steps[^14][^15].
  
  1. Prompt LLM with current context (the user’s question, a description of available tools (e.g. “search_vector_db(query) → docs”))
  2. LLM emits an action
     - Thought: I should look up similar cases in the vector index.  A
     - ction: search_vector_db("…refined query…")
  3. Agent code runs the tool. E.g. query the vector DB, or hit an API, capture the results.
  4. Agent appends the tool’s output to the context and goes back to step 1.
  5. Repeat until the LLM emits a final answer.

a transcript of previous reasoning/tool calls.
- **Defining agents is easy!_** Sample steps from documentation for an agent that calls a weather API[^13]:
```python
# pip install -qU "langchain[anthropic]" to call the model

from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

- **_Large community & resources_**: LangChain is very popular, meaning extensive documentation, tutorials, and community support. Many developers have shared RAG examples using LangChain, so students can learn from those. If issues arise, chances are someone has encountered it before – reducing development time. The ecosystem is rich and rapidly growing , which can be reassuring for long-term maintenance [^12].
- **_Model independent + pricing/open-source_**: LangChain can work with any open-source LLM. It simply needs a wrapper or API to call the model (Hugging Face transformers, GPT4All, etc). This means we can plug in a high-performing local model like Llama 2/3 70B or Mistral without hassle. LangChain is MIT-licensed open-source, so no licensing obstacles[^16].

**_Cons_**

- **_Steep learning curve_**: The flip side of LangChain’s flexibility is complexity. Setting up intricate chains or agents can be overwhelming for newcomers. There’s a learning curve to understanding its abstractions (chains, agents, callbacks, etc). Debugging a complex multi-step chain can also be tricky, since the control flow decided by the LLM[^12].
- **_Overhead/efficiency_**: LangChain introduces some overhead in exchange for modularity. If the workflow involves many sequential steps or tool calls, latency can add up . For example, orchestrating a prompt to call the vector DB, then an LLM, then another tool means multiple roundtrips. In a pure QA setting, this might be less efficient than a purpose-built pipeline that does retrieval then answer generation in one go. Though, for moderate query volumes (ie a student asking one question at a time), this overhead would likely be acceptable since we have GPUs to speed up the LLM inference[^12].
- **_Rapid library changes_**: LangChain’s library evolves quickly. In the past, users online have complained about frequent updates introducing breaking changes if you don't pin versions. This means a bit of caution is needed when upgrading, to avoid spending time fixing things[^17].

**_General notes_**
1. LangChain is performs particularly well in multi-step workflows. For example, first retrieving context from a vector DB, then making an API call or websearch (other tools like calculator), then finally calling an LLM. Hence, all steps are chained together coherently.
2. **_Integration with vector DBs_**: LangChain makes it straightforward to use any of the popular vector stores. It has built-in classes for Qdrant, Weaviate, Milvus, and others (Pinecone, Chroma). For example, you can use QdrantVectorStore or Weaviate integration to connect to your database by simply providing the host URL/API key. All data stays on the cluster – LangChain will query the local vector DB via its Python client, so there’s no external call. There is also no vendor lock-in at this layer meaning we could swap the vector backend with minimal changes if needed. For example [^18] a Weaviate DB can be instantiated simply by adding a cluster’s URL and API key,
```python
from langchain_community.vectorstores import Weaviate

weaviate = Weaviate.from_texts(
    texts,
    embeddings,
    weaviate_url="https://my-weaviate-host:8080",
    weaviate_api_key="YOUR_API_KEY",
)
```
3. **_Security_**: All components can be deployed in our local env. LangChain does not require any online API if we’re using local models and data sources. However, we would need to avoid using LangChain’s default OpenAI APIs or any internet tools. Instead, we’d use local LLM endpoints and the internal vector store s.t no sensitive data would leave the cluster.

## Haystack + Open Source 

**_Pros_**

- **_Built for RAG/QA_**: Haystack  has native support for typical retrieval then reading pipelines. For instance, it’s straightforward to configure a Vector retriever (using embeddings) and then use a Reader component which can be a QA model or an LLM. This means less manual wiring compared to general frameworks. The components are optimized for accuracy and relevance – e.g. you can easily swap in different retrievers (dense vs sparse, or hybrid) and rankers. This focus often makes Haystack very efficient and effective for document QA tasks[^12].
- **_Flexible vector DB integration and/or document stores_**: There is  support for various document stores. You can use Elasticsearch/OpenSearch, FAISS, or  vector DBs like Milvus, Weaviate, Qdrant as the DocumentStore. Haystack provides connectors – for example, a MilvusDocumentStore, WeaviateDocumentStore, and recently a QdrantDocumentStore (maintained with Qdrant team). This makes it simple to persist and index the PDFs in a persistent DB. Notably, Haystack supports hybrid search (combining keyword and vector queries) and metadata filters, which can be useful if we want to filter papers by fields (year, author, etc)[^19].
- **_Built-in REST API & monitoring_**: Haystack offers a REST API ("Hayhooks"[^21]) service for multi-user cluster setups. You can deploy Haystack as a server that exposes endpoints for querying the QA system. It also has some UIs and monitoring tools (Haystack Annotate) to evaluate retriever/reader performance. This enterprise could save us time if we plan to let many students use the system concurrently or if we ever want to monitor accuracy for further improvement.
- **_Multi-step reasoning w/agents_**: Haystack has an agent capability (since v1.15) which allows an LLM to use Haystack pipelines as tools. For instance, an agent can decide to run a DocumentSearchPipeline[^20] tool to fetch information, then maybe a calculator tool, etc. This means Haystack can handle multi-hop questions or a sequence of actions similarly to LangChain’s agents. While this feature is newer compared to LangChain’s agents, it’s available and can be leveraged for complex queries that require multiple steps.
- **_Ease of use for QA tasks_**: The framework abstracts the low level stuff (vector indexing, merging results, etc) nicely. Basic pipelines can be set up with a few lines of configuration – which is great for initial prototyping. Also, Haystack comes with good documentation and examples specifically around RAG applications (e.g. tutorials for building a QA system). A couple users have commented that Haystack is very beginner friendly for LLM QA with basic Python knowledge[^23].
- **_Licensing & local deployment_**: Haystack is open-source (Apache 2.0)[^22]. It can run fully on PACE – we can install it on a node and point it to
local DBs and models, with no external dependencies required.


**_Cons_**

- **Initial setup complexity_**: According to the following [Medium article](https://medium.com/@heyamit10/llamaindex-vs-langchain-vs-haystack-4fa8b15138fd),
> "Haystack is more suited for enterprises, and setting it up can be resource-intensive. You’ll need robust infrastructure and technical know-how to get the most out of it. If you’re a small business or a developer looking for a simpler setup, you might find Haystack to be overkill. Its resource requirements and the effort needed to fine-tune it can be barriers for smaller teams or solo developers."

  However, I was able to setup a mini-RAG pipeline with Neo4j and a locally hosted Llama-3B 8B model in roughly a week... though the pipeline was far from perfect and I did not test it extensively (the LLM struggled to make retrieval calls from the DB and hallucinated quotes when it did - still unsure what caused this to happen).
- **_Possibly overkill?_**: If our corpus is, say, a few thousand PDFs, we might not need the full scaling capabilities of Haystack. Haystack is optimized for large-scale production. The overhead in memory (ie running an Elasticsearch or a Milvus server alongside) might not be justified if a lighter-weight approach (vanilla Python SDKs?) could handle the load. Overhead shows up in the following ways:
  1. Haystack’s readers typically use BERT-style models (ie deepset/roberta-base-squad2) which require hundreds of megabytes of RAM (and often GPU VRAM) to load[^24].
  2. Tokenization w I/O: Each query goes through Haystack’s preprocessing (tokenizer) and postprocessing (extracting spans), adding tens to hundreds of milliseconds beyond a raw model call.
  3. Pipeline setup: Haystack chains retriever -> reader -> whatever else (rankers, filters), adding serialization/deserialization and Python level overhead.

  This stems mostly from opinions of users online. I personally don't find Haystacks documentation too hard to follow, and do think that if deep search is a platform that may eventually scale to hundreds/thousands of students at GT, it wouldn't be a bad thing to have infrastructure that can handle higher loads.

- **_Learning curve_**: Basic usage is pretty straightforward, but if customizing ranking, or implementing a complex query workflow will require more time studying  Haystack docs. The configuration of pipelines with YAML or Python code, and choosing among retriever types or adjusting query params, requires some study of the docs. It’s less code than LangChain for the same task, though. Also, the agent tool use in Haystack is relatively new – fewer community examples exist compared to LangChain’s agent, meaning you might have to feel your way through that part.

**_General notes_**
1. Designed for production-grade QA systems. It provides a pipeline architecture where you can define components like a DocumentStore (vector or keyword index), "Retrievers", and "Readers" (LLMs). Haystack is best at scalable document search and question-answerin. It's essentially an advanced search engine toolkit. In the context of our project, we’d use it to set up a pipeline like: a Dense Retriever that pulls top k relevant chunks from the vector DB, and feeds them to a "Generative Reader" (open-source LLM) that produces the answer given those chunks. Haystack also supports agentic behavior now (Haystack Agents), allowing the LLM to use tools/pipelines in a multi-step process, though its primary focus is on retrieval and answer extraction[^12].

## LlamaIndex + Open Source 

**_Pros_**

- **_Simplicity_**: LlamaIndex can be easier to get started with for our specific task. If the core need is “LLM, but with my PDFs as knowledge,” LlamaIndex provides a straightforward path. It handles document parsing (PDFs to text chunks), embedding them, and retrieving relevant chunks for a query with minimal code. Developers with basic Python skills can quickly set up an index and start querying, without needing to assemble complex pipelines. This means a shorter learning curve compared to the other frameworks[^12].
- **_Optimized context retrieval_**: LlamaIndex is designed to get the most relevant context to feed the LLM. It supports advanced indexing strategies like adding a hierarchical index (tree of summaries) or a keyword table, in addition to a vector index. These can improve accuracy: i.e the tree index can allow the LLM to progressively refine which section of data to look at. Pure vector search ("VectorStoreIndex") integrates directly with vector DBs and ensures fast semantic search. The framework is efficient at slicing documents into chunks and only retrieving the top k relevant chunks, which the LLM then sees[^25]. 
- **_Retrieval w/ multiple data types_**: LlamaIndex isn’t limited to just vector retrieval. It can store tabular data, knowledge graphs (KG-RAG), or APIs as part of indices. For instance, you could enrich the index with structured data (like an ontology of research topics) or use a Graph RAG approach (they even have an example integrating Qdrant with Neo4j for graph-based retrieval)[^26]. In our project, if we have metadata or a citation graph of papers, LlamaIndex could potentially use that structure in answering questions (chain of thought kind of context retrieval).
- **_Good for large context windows_**: LlamaIndex can exploit LLMs with very large context windows. For example, Llama 3 70B model offers up to 128k tokens of context. LlamaIndex could in theory, retrieve a lot of relevant info and pack it into a single query if the model supports it. It even has functionalities for chunking queries or doing iterative generation (like a REFINE mode where the LLM first answers with some context, then it can request more info if needed). This pairs well with models optimized for RAG tasks[^27]. 
- **_Lightweight/open-source_**: LlamaIndex is MIT-licensed and quite lightweight – basically a Python library with minimal external requirements. If we don’t need a complex server, we could run LlamaIndex in a simple script or notebook environment on PACE. It can use an in-memory index for quick experimentation and later switch to a persistent vector store. Because it’s not doing a lot beyond retrieval and interfacing with LLM, the runtime overhead is small. The heavy operations (vector search, model inference) are handled by the vector DB and the LLM backend respectively, so LlamaIndex just glues them together efficiently. So, faster development and possibly lower latencies. In fact, for a single query, LlamaIndex might be the most direct: it will issue a vector search and then directly call the LLM, without [^28]. 


**_Cons_**
- **_Narrower scope (limited agent abilities)_**: LlamaIndex is not a full agent framework. It doesn’t natively let the LLM decide to use arbitrary tools beyond its data indices. If a query requires multistep reasoning that goes outside the documents (like “search the web then look in the docs”), LlamaIndex alone won’t orchestrate that. It expects that your main “tool” is the index of data you gave it. In the context of our project, this is mostly fine (we want it to use the PDFs to answer questions), but if in the future we wanted the assistant to do things like run code, interface with other APIs, or have back and forth dialogues with planning, LlamaIndex would not handle that out of the box.
- **_Performance at scale_**: LlamaIndex can handle large datasets (tens of thousands of documents) by delegating to vector stores. However, for very large collections, we'd need to manually configure the LlamaIndex indices. Specifically, LlamaIndex has an additional layer of indexing primitives on top of the vector DB (ListIndex, TreeIndex, KeywordTableIndex, CompositeIndex, etc). This differs from Haystack/LangChain which treat vector DBs more as a blackbox which self-configures all sharding, metadata indexing, ANN settings, etc. So in Haystack for instance we might choose from a DenseRetriever or KNNRetriever component that simply queries the DB at a single index level which is automatically configured by the database. Below is a table comparing the vector DB implementation logic:

| Feature             | Haystack[^30] / LangChain[^31]                                                                                  | LlamaIndex[^29]                                                                    |
|---------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| **Sharding**        | Handled by the vector DB’s own cluster/shard manager (ie Milvus replicas, Qdrant segments, Elasticsearch shards) | You choose your own shards via multiple `ListIndex` or `TreeIndex` instances |
| **Metadata filtering** | Pushed down into one call (`filter` kwarg) against the DB’s payload index                           | Can be expressed as separate sub-indices (e.g. keyword tables) in a `CompositeIndex` |
| **Retriever API**   | One retriever -> one index call                                                                         | `CompositeIndex` fans out to many sub-indices                |

- Here is a list of simplified steps for how LlamaIndex queries a vector DB:
    1. Chunk and ingest documents:
      - ListIndex.from_documents(...)
      - Split each document into text chunks.
      - Embed each chunk (via our ServiceContext).
      - Insert those embeddings + chunk metadata into a single “papers” collection in Qdrant/FAISS/etc.
    
    2. Build an in-memory index structure (List, Tree, KeywordTable, Composite, etc) whose nodes point to embeddings in your DB:
          - idx_2024 = ListIndex.from_documents(docs_2024, storage_2024, service_context)
          - idx_2023 = ListIndex.from_documents(docs_2023, storage_2023, service_context)
          - idx_arch = ListIndex.from_documents(docs_arch, storage_arch, service_context)
    4. At query time, traverse the in-memory structure to decide which vector store calls to make (and with which parameters)
    5. Merge and re-rank the results from one or more DB calls, then pass back the top chunks (and their source metadata) to the LLM

**_General notes_**
1. LlamaIndex (formerly GPT Index) is a framework specifically created to connect LLMs with external data by building indices. It’s very tailored to RAG use cases: you feed it documents, it constructs an index (vector based or other structures), and then you can query it in natural language and get answers with the help of an LLM. LlamaIndex is somewhat “thinner” than LangChain or Haystack – it doesn’t try to handle a wide array of tool integrations or multi-step workflows beyond document retrieval. It instead optimizes data ingestion, chunking, and querying process for LLMs. For our project, we’d likely use LlamaIndex to parse all PDFs into an index (probably a VectorStoreIndex backed by vector DB) and then use an open-source LLM to answer questions using that index.

## Brief summary

A nice article summarizing the strengths and weaknesses of LangChain, LlamaIndex and Haystack based on use-case can be read [here](https://milvus.io/ai-quick-reference/what-are-the-differences-between-langchain-and-other-llm-frameworks-like-llamaindex-or-haystack). To quote directly from the article:

> "LangChain’s strength lies in its flexibility for building multi-step LLM workflows. For example, a developer could create a chatbot that first calls a weather API, processes the data with an LLM, and then saves the conversation history to a database—all using LangChain’s pre-built components like Agents, Tools, and Memory. In contrast, LlamaIndex optimizes data ingestion and indexing for LLM queries. If you have a collection of internal documents, LlamaIndex can automatically chunk, embed, and index them for efficient semantic search, then feed relevant passages to an LLM for answers. Haystack, meanwhile, provides a pipeline-centric approach for document processing: a typical Haystack pipeline might connect a document store (like Elasticsearch), a retriever model, and an LLM, with built-in tools for preprocessing PDFs or web pages."

Another article from [Medium](https://medium.com/@heyamit10/llamaindex-vs-langchain-vs-haystack-4fa8b15138fd):

>   * "LlamaIndex (formerly GPT Index) is the go-to when you need a fast, flexible tool for search and retrieval of complex data.
>   * LangChain focuses on creating workflows that can chain tasks together, integrating LLMs with APIs, databases, or tools.
>   * Haystack shines when you’re building production-ready search applications, whether it’s for question-answering systems or enterprise-scale document retrieval."

Based on this evaluation and the pros and cons of each detailed above - for our internal deep search platform on GT SMARTTech data - Haystack appears to be the best choice IF we plan to turn this into a platform available to many GT students for research (ie demand grows to the thousands). However, in terms of the simplest to implement and fastest option (least overhead), LlamaIndex would be the best pick, it would just require manual DB optimization down the line.

## References

[^1]: https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/agents-overview

[^2]: https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/overview-declarative-agent

[^3]: https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/agents-overview

[^4]: https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/agents-overview#custom-engine-agents

[^5]: https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/copilot-studio-agent-builder

[^6]: https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/optimize-sharepoint-content

[^7]: https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/declarative-agent-manifest-1.4#onedrive-and-sharepoint-object

[^8]: https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/overview-declarative-agent#tailor-declarative-agents-for-your-scenario

[^9]: https://learn.microsoft.com/en-us/microsoftsearch/semantic-index-for-copilot

[^10]: https://learn.microsoft.com/en-us/previous-versions/azure/search/performance-benchmarks

[^11]: https://learn.microsoft.com/en-us/copilot/microsoft-365/microsoft-365-copilot-privacy#data-stored-about-user-interactions-with-microsoft-365-copilot

[^12]: https://medium.com/@heyamit10/llamaindex-vs-langchain-vs-haystack-4fa8b15138fd

[^13]: https://langchain-ai.github.io/langgraph/?_gl=1*1iylnu9*_gcl_au*NTY0MTYxMjIwLjE3NDg3MDkyMzE.*_ga*Njk4MzQwMTg4LjE3NDg3MDkyMzE.*_ga_47WX3HKKY2*czE3NDk2NzAxMzYkbzQkZzAkdDE3NDk2NzAxMzYkajYwJGwwJGgw#get-started

[^14]: https://python.langchain.com/docs/tutorials/agents/

[^15]: https://python.langchain.com/docs/concepts/agents/

[^16]: https://www.langchain.com/langchain

[^17]: https://github.com/langchain-ai/langchain/issues/4332

[^18]: https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.weaviate.Weaviate.html

[^19]: https://docs.haystack.deepset.ai/docs/choosing-a-document-store

[^20]: https://docs.haystack.deepset.ai/docs/retrievers

[^21]: https://docs.haystack.deepset.ai/docs/hayhooks

[^22]: https://github.com/deepset-ai/haystack/blob/main/LICENSE

[^23]: https://www.reddit.com/r/LLMDevs/comments/1bcd7bx/haystack_20stable_a_gateway_drug_for_llm/#:~:text=Haystack%202.0,basic%20programming%20experience%20is%20helpful

[^24]: https://github.com/deepset-ai/haystack/issues/930

[^25]: https://qdrant.tech/documentation/frameworks/llama-index/#:~:text=Llama%20Index%20acts%20as%20an,Qdrant%20as%20a%20vector%20index

[^26]: https://qdrant.tech/documentation/frameworks/llama-index/#:~:text=,Private%20RAG%20Information%20Extraction%20Engine

[^27]: https://www.instaclustr.com/education/open-source-ai/top-10-open-source-llms-for-2025/#:~:text=,transformer%20architecture%2C%20which%20enhances%20the

[^28]: https://www.valprovia.com/en/blog/llamaindex-bridging-your-data-and-large-language-models#:~:text=LlamaIndex%3A%20Bridging%20Your%20Data%20and,The%20only

[^29]: https://docs.llamaindex.ai/en/v0.9.48/module_guides/indexing/composability.html

[^30]: https://zilliz.com/ai-faq/how-do-i-scale-a-haystack-search-system-for-largescale-data

[^31]: https://stackoverflow.com/questions/77217076/langchain-using-filters-in-a-retriever
