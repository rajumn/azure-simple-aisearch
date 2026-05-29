import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType,
    SimpleField, SearchableField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields,
    SemanticField
)
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
import uuid

load_dotenv()

# ── Clients ────────────────────────────────────────────────────────────────────

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

index_client = SearchIndexClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
)


# ── Step 1: Create the vector index ───────────────────────────────────────────

def create_index():
    index_name = os.getenv("AZURE_SEARCH_INDEX")

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,          # Ada-002 output dim
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[VectorSearchProfile(
            name="hnsw-profile",
            algorithm_configuration_name="hnsw-algo",
        )],
    )

    # Optional: semantic ranking boosts precision further
    semantic_config = SemanticConfiguration(
        name="semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]
        ),
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )

    result = index_client.create_or_update_index(index)
    print(f"✅ Index '{result.name}' ready.")


# ── Step 2: Embed text ─────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model=os.getenv("EMBEDDING_DEPLOYMENT"),
    )
    return response.data[0].embedding


# ── Step 3: Ingest PDFs ────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str):
    print(f"📄 Loading {pdf_path}...")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"   → {len(chunks)} chunks created")

    documents = []
    for chunk in chunks:
        embedding = get_embedding(chunk.page_content)
        documents.append({
            "id": str(uuid.uuid4()),
            "content": chunk.page_content,
            "source": os.path.basename(pdf_path),
            "page": chunk.metadata.get("page", 0),
            "embedding": embedding,
        })

    # Upload in batches of 100 (Search SDK limit)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        result = search_client.upload_documents(documents[i:i+batch_size])
        print(f"   → Uploaded batch {i//batch_size + 1}: {len(result)} docs")

    print(f"✅ Ingestion complete: {len(documents)} chunks indexed.")


# ── Step 4: RAG query ──────────────────────────────────────────────────────────

def rag_query(question: str, top_k: int = 5) -> str:
    print(f"\n🔍 Query: {question}")

    # Embed the user question
    query_vector = get_embedding(question)

    # Vector search against the index
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    results = search_client.search(
        search_text=question,          # Also run keyword search (hybrid)
        vector_queries=[vector_query],
        select=["content", "source", "page"],
        top=top_k,
    )

    # Build context from retrieved chunks
    context_parts = []
    for r in results:
        context_parts.append(
            f"[Source: {r['source']}, Page {r['page']}]\n{r['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # LLM call with context
    system_prompt = """You are a helpful assistant that answers questions 
based strictly on the provided document context. 
If the answer is not in the context, say "I don't have enough information."
Always cite the source document and page number."""

    user_prompt = f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""

    response = openai_client.chat.completions.create(
        model=os.getenv("CHAT_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,        # Low temp = factual, less hallucination
        max_tokens=1000,
    )

    answer = response.choices[0].message.content
    print(f"\n💬 Answer:\n{answer}")
    return answer


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Create index (idempotent — safe to re-run)
    create_index()

    # 2. Ingest your PDFs
    import glob
    pdf_files = glob.glob("./pdfs/*.pdf")
    for pdf in pdf_files:
        ingest_pdf(pdf)

    # 3. Ask questions
    while True:
        question = input("\nAsk a question (or 'quit'): ").strip()
        if question.lower() == "quit":
            break
        rag_query(question)