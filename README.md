# azure-simple-aisearch

Step 1 — Azure resources setup
Before any code, provision these two resources from the Azure Portal:
Azure AI Search → Create Resource → Azure AI Search → Free tier (1 index, 50MB)
Azure Blob Storage → Create Resource → Storage Account → create a container called pdfs
Collect these values from the Portal:

AI Search: endpoint URL + Admin API key
Blob Storage: connection string
Azure OpenAI: endpoint + API key + deployment names for text-embedding-ada-002 and gpt-4o

Step 2 — Install dependencies
bashpip install langchain langchain-openai langchain-community \
    azure-search-documents azure-storage-blob \
    pypdf python-dotenv openai


Step 3 — Environment config
Create a .env file:
envAZURE_SEARCH_ENDPOINT=https://<your-service>.search.windows.net
AZURE_SEARCH_KEY=<admin-key>
AZURE_SEARCH_INDEX=pdf-rag-index

AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<api-key>
AZURE_OPENAI_API_VERSION=2024-02-01
EMBEDDING_DEPLOYMENT=text-embedding-ada-002
CHAT_DEPLOYMENT=gpt-4o

AZURE_STORAGE_CONNECTION_STRING=<connection-string>
AZURE_STORAGE_CONTAINER=pdfs



Step 4 — Full RAG pipeline code
rag_pipeline.py — this single file handles index creation, PDF ingestion, and querying:



Key concepts to understand		
        
Concept	What it does	Azure component
Chunking	Splits PDFs into ~1000-char overlapping pieces	LangChain RecursiveCharacterTextSplitter
Embedding	Converts text → 1536-dim float vector	Azure OpenAI text-embedding-ada-002
HNSW index	Approximate nearest-neighbor vector search	Azure AI Search vector field
Hybrid search	Combines vector + keyword BM25 in one query	search_text + vector_queries together
Semantic ranking	Re-ranks results using a language model	Azure AI Search semantic config
Grounding	LLM only uses retrieved context, reducing hallucination	System prompt + strict context window


Common issues and fixes
Rate limits during ingestion — add time.sleep(0.5) between embedding calls, or use asyncio with a semaphore.
Chunk size tuning — start with chunk_size=1000, chunk_overlap=200. Reduce to 500 for dense technical docs, increase to 1500 for narrative text.
Free tier limits — the free Search tier caps at 1 index and 50MB. For larger PDFs, upgrade to Basic ($0.10/hour) or use the paid tier.
Empty results — ensure vector_search_dimensions=1536 matches Ada-002's output exactly, and that you called create_index() before uploading documents.
