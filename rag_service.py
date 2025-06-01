from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
from typing import List, Dict, Any, Optional
import os


app = FastAPI()

# Load embedding and generation models
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # 768-dim
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# In-memory document store and FAISS index
chunks_store = []
doc_summaries = []
index = faiss.IndexFlatL2(768)

# Config
CHUNK_SIZE = 200  # Number of characters per chunk
OVERLAP = 100  # Overlap between chunks to maintain context
MIN_CHUNK_LENGTH = 20  # Minimum character length for a chunk
MAX_TOKEN_LENGTH = 512  # Maximum tokens for embedding
SIMILARITY_THRESHOLD = 0.4  # Threshold for considering a match relevant
MAX_CHUNKS_TO_RETRIEVE = 5  # Number of chunks to retrieve

# Keywords to identify concert-related documents
DOMAIN_KEYWORDS = [
    "concert", "tour", "venue", "performer", "schedule", "dates", "logistics", "guest",
    "soundcheck", "backstage", "rider", "ticket", "audience", "merchandise", "promotion",
    "booking", "crew", "stage", "lighting", "audio", "equipment", "itinerary", "gig",
    "setlist", "load-in", "roadie", "opening act", "travel", "accommodation", "transportation"
]

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": f"An unexpected error occurred: {str(exc)}"}
    )

def is_concert_domain(text: str) -> bool:
    """Check if the document content is within the concert domain."""
    return any(re.search(rf'\b{kw}\b', text, re.IGNORECASE) for kw in DOMAIN_KEYWORDS)

def generate_answer(prompt: str, max_tokens: int = 200) -> str:
    """Generate an answer using the T5 model."""
    try:
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
        output_ids = generator_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

def chunk_document(text: str) -> List[str]:
    """Split document into chunks with overlap."""
    # Chunking by character length with overlap
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - OVERLAP):
        chunk = text[i:i + CHUNK_SIZE]
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)
    
    # If no chunks were created (e.g., very short document), use the full text
    if not chunks:
        chunks = [text]
        
    return chunks

def embed_chunks(chunks: List[str]) -> List[np.ndarray]:
    """Embed document chunks."""
    embeddings = []
    for chunk in chunks:
        # Truncate chunk if it's too long
        if len(tokenizer.encode(chunk)) > MAX_TOKEN_LENGTH:
            tokens = tokenizer.encode(chunk, truncation=True, max_length=MAX_TOKEN_LENGTH)
            chunk = tokenizer.decode(tokens, skip_special_tokens=True)
        
        embedding = embedder.encode(chunk)
        embeddings.append(embedding)
    
    return embeddings

@app.post("/ingest")
async def ingest_document(document: str = Form(...)):
    """Ingest a document into the RAG system."""
    if not document or len(document.strip()) < 10:
        raise HTTPException(status_code=400, detail="Document is too short or empty")
    
    if not is_concert_domain(document):
        return {"message": "Sorry, I cannot ingest documents with other themes."}

    try:
        # First, generate a summary for the document
        summary_prompt = (
            "Summarize all the information of the following document into key points:\n\n"
            f"{document}"
        )
        summary = generate_answer(summary_prompt, max_tokens=250)
        
        # Store document summary
        doc_id = len(doc_summaries)
        doc_summaries.append({
            "doc_id": doc_id,
            "summary": summary,
            "full_text": document
        })
        
        # Chunk the document and embed each chunk
        chunks = chunk_document(document)
        embeddings = embed_chunks(chunks)
        
        # Store chunks and their embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunks_store.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            })
            index.add(embedding.reshape(1, -1))
        
        return {
            "message": "Thank you for sharing! Your document has been successfully added to the database.",
            "summary": summary,
            "chunks_count": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        return v

def compute_similarity(query_embedding, chunk_embedding):
    """Compute cosine similarity between query and chunk."""
    # Normalize embeddings
    query_norm = np.linalg.norm(query_embedding)
    chunk_norm = np.linalg.norm(chunk_embedding)
    
    if query_norm == 0 or chunk_norm == 0:
        return 0.0
    
    # Compute cosine similarity
    similarity = np.dot(query_embedding, chunk_embedding) / (query_norm * chunk_norm)
    return similarity

@app.post("/ask")
def ask_question(request: QueryRequest):
    """Answer a question using the RAG system."""
    if not chunks_store:
        return {"answer": "No documents have been ingested yet. Please upload some concert information first."}

    try:
        # First approach: try to find exact matches by embedding the query
        query_embedding = embedder.encode(request.query).reshape(1, -1)
        k = min(MAX_CHUNKS_TO_RETRIEVE, len(chunks_store))  # Get top k results or all if less than k
        D, I = index.search(query_embedding, k=k)
        
        # Get all retrieved chunks regardless of similarity
        relevant_chunks = []
        for i, distance in zip(I[0], D[0]):
            if i != -1:  # Valid index
                chunk = chunks_store[i]
                relevant_chunks.append(chunk)
        
        # If no chunks found, fall back to using document summaries
        if not relevant_chunks:
            # Try using the entire documents as context
            relevant_contexts = [doc["full_text"] for doc in doc_summaries]
            context = "\n\n===\n\n".join(relevant_contexts[:2])  # Use first 2 documents
        else:
            # Extract text from retrieved chunks
            chunk_texts = [chunk["text"] for chunk in relevant_chunks]
            context = "\n\n".join(chunk_texts)
            
            # If context is too short, augment with document summaries
            if len(context) < 500:
                doc_ids = set(chunk["doc_id"] for chunk in relevant_chunks)
                for doc_id in doc_ids:
                    # Add the document summary
                    if doc_id < len(doc_summaries):
                        context += f"\n\nAdditional context from document {doc_id}:\n{doc_summaries[doc_id]['summary']}"
        
        # Generate answer with the assembled context
        answer_prompt = (
            f"Based on the following concert tour information, answer the question accurately. "
            f"If you don't have enough information to give a complete answer, provide what you know "
            f"based on the context provided.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {request.query}\n\n"
            f"Answer:"
        )
        
        answer = generate_answer(answer_prompt, max_tokens=250)
        return {"answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

class ArtistRequest(BaseModel):
    artist_name: str

@app.post("/search_concerts")
def search_artist_concerts(request: QueryRequest):
    """Search for concerts for a specific artist."""
    artist = request.query
    
    if not artist or len(artist.strip()) < 2:
        raise HTTPException(status_code=400, detail="Artist name is too short")
    
    api_key = os.getenv("SERPAPI_KEY", "b3c5f0fe0299f3272e5013683a6422900563b3a207ce836e13bc5d4ebecb490f")
    search_url = "https://serpapi.com/search.json"
    params = {
        "q": f"{artist} tour 2025 site:ticketmaster.com OR site:songkick.com OR site:livenation.com",
        "engine": "google",
        "api_key": api_key
    }

    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()

        events = results.get("organic_results", [])[:3]  # use top 3 results

        if not events:
            return {"result": f"No upcoming concerts found for {artist}. Try checking the artist's official website or social media for tour announcements."}

        formatted_concerts = []

        for event in events:
            title = event.get("title", "No Title")
            snippet = event.get("snippet", "No snippet available")
            link = event.get("link", "")

            formatted_line = f"🎵 {title} — 📄 {snippet} — 🔗 {link} \n"
            formatted_concerts.append(formatted_line)
        
        concerts_summary = "\n".join(formatted_concerts)

        return {"result": concerts_summary}

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Search request timed out")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error from search API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching artist concerts: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
