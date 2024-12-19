# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import time
import json
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for lazy loading
pc: Optional[Pinecone] = None
index = None
model = None


# Startup state manager
class AppState:
    is_initialized = False


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await initialize_services()
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    yield
    # Shutdown
    logger.info("Shutting down services...")


app = FastAPI(lifespan=lifespan)


class Document(BaseModel):
    text: str


class Query(BaseModel):
    text: str
    top_k: int = 3


async def initialize_services():
    """Initialize all required services"""
    global pc, index, model, app_state

    if app_state.is_initialized:
        return

    try:
        logger.info("Starting application initialization...")

        # Check environment variables
        logger.info("Checking environment variables...")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        if not pinecone_key or not openrouter_key:
            raise ValueError("Missing required environment variables")

        logger.info("Environment variables verified")

        # Initialize Pinecone
        logger.info("Initializing Pinecone...")
        pc = Pinecone(api_key=pinecone_key)
        index_name = "knowledge-base"
        index = pc.Index(index_name)
        logger.info("Pinecone initialized successfully")

        # Initialize model
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")

        app_state.is_initialized = True
        logger.info("Application initialization completed successfully")

    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        app_state.is_initialized = False
        raise


async def ensure_initialized():
    """Ensure all services are initialized"""
    if not app_state.is_initialized:
        await initialize_services()


@app.post("/index")
async def index_documents(documents: List[Document]):
    try:
        await ensure_initialized()

        texts = [doc.text for doc in documents]
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = model.encode(texts)

        logger.info("Upserting documents to Pinecone...")
        for i, embedding in enumerate(embeddings):
            index.upsert([(f"doc-{time.time()}-{i}", embedding.tolist(), {"text": texts[i]})])

        return {"status": "success", "message": f"Indexed {len(texts)} documents"}
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_documents(query: Query):
    try:
        await ensure_initialized()

        # Get embeddings and search
        logger.info(f"Processing query: {query.text}")
        query_embedding = model.encode([query.text])[0]

        logger.info("Searching Pinecone index...")
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=query.top_k,
            include_metadata=True
        )

        context = [match['metadata']['text'] for match in results['matches']]

        # Generate response using OpenRouter
        logger.info("Generating response using OpenRouter...")
        prompt = f"""
        Use the following context to answer the question.

        Context:
        {chr(10).join(context)}

        Question:
        {query.text}

        Answer:
        """

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            },
            json={
                "model": "meta-llama/llama-3.2-3b-instruct:free",
                "messages": [{"role": "user", "content": prompt}],
            }
        )

        if response.status_code == 200:
            llm_response = response.json()['choices'][0]['message']['content']
        else:
            logger.error(f"OpenRouter API error: {response.text}")
            llm_response = "Error generating response"

        return {
            "query": query.text,
            "context": context,
            "response": llm_response
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    try:
        await ensure_initialized()

        # Verify each component
        if not model:
            raise Exception("Model not initialized")
        if not index:
            raise Exception("Pinecone index not initialized")

        return {
            "status": "healthy",
            "message": "All components initialized",
            "initialized": app_state.is_initialized
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))