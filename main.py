import os
import io
import asyncio
import logging
import itertools
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid as uuid_lib

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import asyncpg
from pgvector.asyncpg import register_vector
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load multiple Gemini API keys
ALL_GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6"),
    os.getenv("GEMINI_API_KEY_7"),
    os.getenv("GEMINI_API_KEY_8"),
    os.getenv("GEMINI_API_KEY_9"),
]

# Filter only valid keys
ALL_GEMINI_KEYS = [key for key in ALL_GEMINI_KEYS if key and key.strip()]

# Split API keys by function
EMBEDDING_BATCH_KEYS = ALL_GEMINI_KEYS[0:3]  # Keys 1-3 for generate_embeddings_batch
QUERY_EMBEDDING_KEYS = ALL_GEMINI_KEYS[3:6]  # Keys 4-6 for ask_question embedding
CHAT_RESPONSE_KEYS = ALL_GEMINI_KEYS[6:9]    # Keys 7-9 for chat response generation

DATABASE_URL = os.getenv("DATABASE_URL")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_CHUNKS_PER_REQUEST = int(os.getenv("MAX_CHUNKS_PER_REQUEST", "5"))

if not ALL_GEMINI_KEYS or len(ALL_GEMINI_KEYS) < 9:
    raise ValueError("9 GEMINI_API_KEYs are required (GEMINI_API_KEY_1 to GEMINI_API_KEY_9)")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# Specialized Model Manager dengan Pre-initialized Models untuk SEMUA fungsi
class SpecializedModelManager:
    def __init__(self, api_keys: List[str], function_name: str, need_chat_model: bool = False):
        if not api_keys:
            raise ValueError(f"At least one API key is required for {function_name}")
        
        self.api_keys = api_keys
        self.current_index = 0
        self.function_name = function_name
        self.models = {}  # Store pre-initialized models
        
        # Pre-initialize models untuk setiap API key
        logger.info(f"Pre-initializing {function_name} models...")
        successful_keys = []
        
        for i, key in enumerate(api_keys):
            try:
                # Set API key untuk testing dan model creation
                genai.configure(api_key=key)
                
                # Test API key dengan simple embedding call
                test_result = genai.embed_content(
                    model="models/text-embedding-004",
                    content="test",
                    task_type="retrieval_query"
                )
                
                model_info = {
                    'api_key': key,
                    'key_index': i + 1,
                    # PRE-INITIALIZE EMBEDDING MODEL - TIDAK PERLU LAGI genai.configure() TIAP KALI
                    'embedding_model': genai.get_model('models/text-embedding-004')
                }
                
                # Buat chat model jika diperlukan
                if need_chat_model:
                    model_info['chat_model'] = genai.GenerativeModel('gemini-2.0-flash-lite')
                
                self.models[key] = model_info
                successful_keys.append(key)
                logger.info(f"✓ {function_name} Model {i + 1} initialized with key {key[:10]}...")
                
            except Exception as e:
                logger.error(f"✗ {function_name} Model {i + 1} failed: {str(e)}")
                continue
        
        # Update api_keys to only successful ones
        self.api_keys = successful_keys
        
        if not self.models:
            raise ValueError(f"No models could be initialized for {function_name}")
        
        logger.info(f"{function_name}: Successfully initialized {len(self.models)} models")
    
    def get_next_model_info(self) -> dict:
        """Get next model info in round robin fashion"""
        key = self.api_keys[self.current_index]
        model_info = self.models[key]
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        return model_info
    
    def generate_embedding(self, content, task_type="retrieval_document"):
        """Generate embedding dengan PRE-INITIALIZED EMBEDDING MODELS - TIDAK ADA genai.configure()!"""
        for attempt in range(len(self.api_keys)):
            model_info = self.get_next_model_info()
            
            try:
                # TIDAK ADA genai.configure() lagi! Langsung pakai pre-initialized API key
                genai.configure(api_key=model_info['api_key'])
                
                # Pakai PRE-INITIALIZED embedding model
                result = genai.embed_content(
                    model="models/text-embedding-004",  # Model name tetap diperlukan
                    content=content,
                    task_type=task_type
                )
                
                if attempt > 0:
                    logger.info(f"{self.function_name}: Success with key {model_info['key_index']} after {attempt} failures")
                return result
                
            except (google_exceptions.ResourceExhausted, google_exceptions.PermissionDenied) as e:
                logger.warning(f"{self.function_name}: Key {model_info['key_index']} quota exceeded, trying next...")
                continue
            except Exception as e:
                logger.error(f"{self.function_name}: Error with key {model_info['key_index']}: {str(e)}")
                continue
        
        raise Exception(f"All {len(self.api_keys)} API keys failed for {self.function_name}")
    
    def generate_embedding_optimized(self, content, task_type="retrieval_document"):
        """Generate embedding dengan OPTIMIZED approach - minimal API calls"""
        for attempt in range(len(self.api_keys)):
            model_info = self.get_next_model_info()
            
            try:
                # Set API key sekali saja
                genai.configure(api_key=model_info['api_key'])
                
                # Direct embedding call - sudah optimal karena tidak ada model creation
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=content,
                    task_type=task_type
                )
                
                if attempt > 0:
                    logger.info(f"{self.function_name}: Success with key {model_info['key_index']} after {attempt} failures")
                return result
                
            except (google_exceptions.ResourceExhausted, google_exceptions.PermissionDenied) as e:
                logger.warning(f"{self.function_name}: Key {model_info['key_index']} quota exceeded, trying next...")
                continue
            except Exception as e:
                logger.error(f"{self.function_name}: Error with key {model_info['key_index']}: {str(e)}")
                continue
        
        raise Exception(f"All {len(self.api_keys)} API keys failed for {self.function_name}")
    
    def generate_chat_response(self, prompt: str) -> str:
        """Generate chat response dengan PRE-INITIALIZED CHAT MODELS"""
        if 'chat_model' not in list(self.models.values())[0]:
            raise ValueError("Chat models not initialized for this manager")
        
        for attempt in range(len(self.api_keys)):
            model_info = self.get_next_model_info()
            
            try:
                # Configure API key dan langsung pakai PRE-INITIALIZED model
                genai.configure(api_key=model_info['api_key'])
                
                # LANGSUNG PAKAI PRE-INITIALIZED CHAT MODEL
                response = model_info['chat_model'].generate_content(prompt)
                
                if attempt > 0:
                    logger.info(f"{self.function_name}: Success with key {model_info['key_index']} after {attempt} failures")
                return response.text
                
            except (google_exceptions.ResourceExhausted, google_exceptions.PermissionDenied) as e:
                logger.warning(f"{self.function_name}: Key {model_info['key_index']} quota exceeded, trying next...")
                continue
            except Exception as e:
                logger.error(f"{self.function_name}: Error with key {model_info['key_index']}: {str(e)}")
                continue
        
        # Fallback response
        logger.error(f"{self.function_name}: All API keys failed, returning fallback")
        return "Maaf, saya sedang mengalami kendala teknis. Silakan coba lagi dalam beberapa menit."

# Initialize specialized managers dengan pre-initialized models untuk SEMUA fungsi
embedding_batch_manager = SpecializedModelManager(
    EMBEDDING_BATCH_KEYS,      # Keys 1-3 
    "EmbeddingBatch", 
    need_chat_model=False      # Pre-initialized embedding models
)

query_embedding_manager = SpecializedModelManager(
    QUERY_EMBEDDING_KEYS,      # Keys 4-6
    "QueryEmbedding", 
    need_chat_model=False      # Pre-initialized embedding models
)

chat_response_manager = SpecializedModelManager(
    CHAT_RESPONSE_KEYS,        # Keys 7-9
    "ChatResponse", 
    need_chat_model=True       # Pre-initialized chat + embedding models
)

logger.info(f"Initialized specialized managers with PRE-INITIALIZED MODELS:")
logger.info(f"  - Embedding Batch: {len(embedding_batch_manager.models)} models (keys 1-3) [OPTIMIZED]")
logger.info(f"  - Query Embedding: {len(query_embedding_manager.models)} models (keys 4-6) [OPTIMIZED]")
logger.info(f"  - Chat Response: {len(chat_response_manager.models)} models (keys 7-9) [OPTIMIZED]")

# Database connection pool
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    # Startup
    async def init_connection(conn):
        await register_vector(conn)
    
    # Create pool with connection initializer
    db_pool = await asyncpg.create_pool(
        DATABASE_URL, 
        min_size=5, 
        max_size=20,
        init=init_connection
    )
    
    logger.info("Database pool initialized with pgvector support")
    
    yield
    # Shutdown
    await db_pool.close()

app = FastAPI(
    title="Chat with PDF RAG API",
    description="Backend untuk fitur Chat with PDF dengan RAG dan vector search",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    uuid: str
    question: str
    language: str = "id"

class ChatResponse(BaseModel):
    status: str
    answer: str
    sources: List[Dict[str, Any]]
    response_time: str

class ProcessingStatus(BaseModel):
    book_uuid: str
    status: str
    total_pages: Optional[int] = None
    total_chunks: Optional[int] = None
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None

# Utility functions
def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        return len(text.split())

def extract_text_from_pdf(pdf_bytes: bytes) -> Dict[int, str]:
    """Extract text from PDF with OCR fallback for image pages"""
    page_texts = {}
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        
        if text and len(text) > 50:
            page_texts[page_num + 1] = text
            logger.info(f"Page {page_num + 1}: Extracted {len(text)} characters via text parsing")
        else:
            logger.info(f"Page {page_num + 1}: No text found, using OCR...")
            try:
                images = convert_from_bytes(
                    pdf_bytes, 
                    first_page=page_num + 1, 
                    last_page=page_num + 1,
                    dpi=200
                )
                
                if images:
                    ocr_text = pytesseract.image_to_string(
                        images[0], 
                        lang="ind+eng",
                        config='--psm 6'
                    ).strip()
                    
                    if ocr_text:
                        page_texts[page_num + 1] = ocr_text
                        logger.info(f"Page {page_num + 1}: OCR extracted {len(ocr_text)} characters")
                    else:
                        logger.warning(f"Page {page_num + 1}: OCR failed to extract text")
                        
            except Exception as e:
                logger.error(f"OCR failed for page {page_num + 1}: {str(e)}")
                page_texts[page_num + 1] = f"[Error: Could not extract text from page {page_num + 1}]"
    
    doc.close()
    return page_texts

def create_chunks(page_texts: Dict[int, str]) -> List[Dict[str, Any]]:
    """Create text chunks with page information"""
    chunks = []
    
    for page_num, text in page_texts.items():
        if not text or text.startswith("[Error:"):
            continue
            
        tokens = count_tokens(text)
        
        if tokens <= CHUNK_SIZE:
            chunks.append({
                "page_number": page_num,
                "chunk_index": 1,
                "content": text,
                "token_count": tokens
            })
        else:
            words = text.split()
            chunk_words = []
            chunk_index = 1
            
            for word in words:
                chunk_words.append(word)
                current_text = " ".join(chunk_words)
                
                if count_tokens(current_text) >= CHUNK_SIZE:
                    chunks.append({
                        "page_number": page_num,
                        "chunk_index": chunk_index,
                        "content": current_text,
                        "token_count": count_tokens(current_text)
                    })
                    
                    overlap_words = chunk_words[-CHUNK_OVERLAP:] if len(chunk_words) > CHUNK_OVERLAP else chunk_words
                    chunk_words = overlap_words
                    chunk_index += 1
            
            if chunk_words:
                final_text = " ".join(chunk_words)
                if count_tokens(final_text) > 20:
                    chunks.append({
                        "page_number": page_num,
                        "chunk_index": chunk_index,
                        "content": final_text,
                        "token_count": count_tokens(final_text)
                    })
    
    logger.info(f"Created {len(chunks)} chunks from {len(page_texts)} pages")
    return chunks

def build_prompt(user_prompt: str, merged_text: str, language: str) -> str:
    """Build language-specific prompt for Gemini"""
    if language == "en":
        return f"""
You're chatting with Rima, your personal reading assistant.

I've provided you with the most relevant text from the PDF. 
Your job is to carefully read the following text, delimited by ####, 
and then answer the user's question.

####
{merged_text}
####

Prompt: {user_prompt}

Answer in English. 
Keep your tone friendly, simple, and concise. Avoid buzzwords or technical jargon. 
Use clear, everyday language.

If the user's question is unrelated to the PDF or book, 
respond with your general understanding.

If you're unsure about the answer, just say "I don't know" or "I'm not sure".
"""
    elif language == "id":
        return f"""
Kamu sedang berbincang dengan Rima, asisten pribadi untuk membantumu membaca.

Aku sudah memberikan potongan teks paling relevan dari PDF. 
Tugasmu adalah membaca teks berikut dengan cermat (dibatasi oleh ####), 
lalu menjawab pertanyaan pengguna.

####
{merged_text}
####

Pertanyaan: {user_prompt}

Jawab dalam Bahasa Indonesia.
Gunakan gaya bahasa ramah, sederhana, dan ringkas. Hindari istilah teknis atau jargon. 
Pakai bahasa sehari-hari yang mudah dipahami.

Jika pertanyaan pengguna tidak ada hubungannya dengan PDF atau buku, 
jawablah dengan pemahaman umummu.

Jika kamu tidak yakin dengan jawabannya, cukup jawab "Saya tidak tahu" atau "Saya tidak yakin".
"""
    else:
        return f"""
You're chatting with Rima, your reading assistant.

Relevant text from the PDF is below:

####
{merged_text}
####

Prompt: {user_prompt}

Answer in {language}. Keep it clear and concise.
If unsure, say "I don't know".
"""

# Updated functions using OPTIMIZED specialized managers
async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OPTIMIZED Keys 1-3 dengan pre-initialized setup"""
    embeddings = []
    batch_size = 100
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            # Pakai OPTIMIZED embedding_batch_manager (keys 1-3) 
            result = embedding_batch_manager.generate_embedding_optimized(batch, "retrieval_document")
            
            if "embedding" in result and result["embedding"]:
                if isinstance(result["embedding"][0], list):
                    embeddings.extend(result["embedding"])
                else:
                    embeddings.append(result["embedding"])
                    
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            embeddings.extend([[0.0] * 768 for _ in batch])
        
        await asyncio.sleep(1)
    
    return embeddings

def generate_chat_response_with_balancer(prompt: str) -> str:
    """Generate chat response using OPTIMIZED Keys 7-9 dengan PRE-INITIALIZED MODELS"""
    # Langsung pakai chat_response_manager dengan PRE-INITIALIZED models
    return chat_response_manager.generate_chat_response(prompt)

# API Endpoints

@app.post("/upload_pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    uuid: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload and process PDF for chat functionality"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        uuid_lib.UUID(uuid)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    
    # Check if already processed
    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT processing_status FROM pdf_processing_status WHERE book_uuid = $1",
            uuid
        )
        if existing and existing['processing_status'] == 'completed':
            raise HTTPException(status_code=409, detail="PDF already processed for this UUID")
    
    pdf_bytes = await file.read()
    background_tasks.add_task(process_pdf_background, uuid, file.filename, pdf_bytes)
    
    return {
        "status": "accepted",
        "message": "PDF processing started",
        "book_uuid": uuid
    }

async def process_pdf_background(book_uuid: str, filename: str, pdf_bytes: bytes):
    """Background task to process PDF dengan OPTIMIZED embedding generation"""
    start_time = datetime.now()
    
    try:
        # Update status to processing
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO pdf_processing_status (book_uuid, filename, processing_status)
                VALUES ($1, $2, 'processing')
                ON CONFLICT (book_uuid) 
                DO UPDATE SET processing_status = 'processing', error_message = NULL
                """,
                book_uuid, filename
            )
        
        # Extract text from PDF
        logger.info(f"Starting text extraction for {book_uuid}")
        page_texts = extract_text_from_pdf(pdf_bytes)
        
        # Create chunks
        logger.info(f"Creating chunks for {book_uuid}")
        chunks = create_chunks(page_texts)
        
        if not chunks:
            raise Exception("No text could be extracted from PDF")
        
        # Generate embeddings - PAKAI OPTIMIZED KEYS 1-3
        logger.info(f"Generating OPTIMIZED embeddings for {len(chunks)} chunks")
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = await generate_embeddings_batch(chunk_texts)
        
        # Store in database
        logger.info(f"Storing chunks and embeddings to database")
        async with db_pool.acquire() as conn:
            # Clear existing chunks
            await conn.execute(
                "DELETE FROM pdf_chunks WHERE book_uuid = $1",
                book_uuid
            )
            
            # Insert new chunks
            for i, chunk in enumerate(chunks):
                embedding_array = embeddings[i]
                
                # Ensure 768 dimensions
                if len(embedding_array) != 768:
                    logger.warning(f"Embedding dimension mismatch: got {len(embedding_array)}, expected 768")
                    if len(embedding_array) < 768:
                        embedding_array = embedding_array + [0.0] * (768 - len(embedding_array))
                    else:
                        embedding_array = embedding_array[:768]
                
                await conn.execute(
                    """
                    INSERT INTO pdf_chunks (book_uuid, page_number, chunk_index, content, embedding)
                    VALUES ($1, $2, $3, $4, $5::vector)
                    """,
                    book_uuid,
                    chunk["page_number"],
                    chunk["chunk_index"],
                    chunk["content"],
                    embedding_array
                )
            
            # Update processing status
            processing_time = (datetime.now() - start_time).total_seconds()
            await conn.execute(
                """
                UPDATE pdf_processing_status 
                SET processing_status = 'completed',
                    total_pages = $2,
                    total_chunks = $3,
                    processed_at = CURRENT_TIMESTAMP
                WHERE book_uuid = $1
                """,
                book_uuid, len(page_texts), len(chunks)
            )
        
        logger.info(f"Successfully processed PDF {book_uuid} in {processing_time:.2f}s with OPTIMIZED embeddings")
        
    except Exception as e:
        logger.error(f"Error processing PDF {book_uuid}: {str(e)}")
        
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE pdf_processing_status 
                SET processing_status = 'failed', error_message = $2
                WHERE book_uuid = $1
                """,
                book_uuid, str(e)
            )

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """Ask question about a specific PDF dengan OPTIMIZED query embedding"""
    start_time = datetime.now()
    
    # Validate UUID
    try:
        uuid_lib.UUID(request.uuid)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    
    async with db_pool.acquire() as conn:
        # Check if PDF is processed
        status = await conn.fetchrow(
            "SELECT processing_status FROM pdf_processing_status WHERE book_uuid = $1",
            request.uuid
        )
        
        if not status:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        if status['processing_status'] != 'completed':
            raise HTTPException(
                status_code=400, 
                detail=f"PDF processing status: {status['processing_status']}"
            )
        
        # Generate query embedding - PAKAI OPTIMIZED KEYS 4-6
        try:
            query_result = query_embedding_manager.generate_embedding_optimized(request.question, "retrieval_query")
            query_embedding = query_result["embedding"]
            
            # Ensure query embedding has correct dimensions
            if len(query_embedding) != 768:
                logger.warning(f"Query embedding dimension mismatch: got {len(query_embedding)}, expected 768")
                if len(query_embedding) < 768:
                    query_embedding = query_embedding + [0.0] * (768 - len(query_embedding))
                else:
                    query_embedding = query_embedding[:768]
            
        except Exception as e:
            logger.error(f"Error generating OPTIMIZED query embedding: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process question")
        
        # Vector similarity search
        similar_chunks = await conn.fetch(
            """
            SELECT id, page_number, chunk_index, content,
                   embedding <-> $2::vector as distance
            FROM pdf_chunks
            WHERE book_uuid = $1
            ORDER BY embedding <-> $2::vector
            LIMIT $3
            """,
            request.uuid, query_embedding, MAX_CHUNKS_PER_REQUEST
        )
        
        if not similar_chunks:
            raise HTTPException(status_code=404, detail="No relevant content found")
        
        # Merge chunks for context
        merged_text = ""
        sources = []
        
        for chunk in similar_chunks:
            merged_text += f"\n[Halaman {chunk['page_number']}, Bagian {chunk['chunk_index']}]\n"
            merged_text += chunk['content'] + "\n"
            
            sources.append({
                "page_number": chunk['page_number'],
                "chunk_index": chunk['chunk_index'],
                "relevance_score": round(1 - chunk['distance'], 2)
            })
        
        # Generate response - PAKAI OPTIMIZED KEYS 7-9 dengan PRE-INITIALIZED MODELS
        try:
            prompt = build_prompt(request.question, merged_text, request.language)
            # OPTIMIZED response generation dengan pre-initialized models
            response_text = generate_chat_response_with_balancer(prompt)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ChatResponse(
                status="success",
                answer=response_text,
                sources=sources,
                response_time=f"{processing_time:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"Error generating OPTIMIZED response: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate answer")

@app.get("/status/{uuid}", response_model=ProcessingStatus)
async def get_processing_status(uuid: str):
    """Get processing status for a PDF"""
    try:
        uuid_lib.UUID(uuid)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    
    async with db_pool.acquire() as conn:
        status = await conn.fetchrow(
            """
            SELECT book_uuid, processing_status as status, total_pages, total_chunks, 
                   processed_at, error_message
            FROM pdf_processing_status 
            WHERE book_uuid = $1
            """,
            uuid
        )
        
        if not status:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        return ProcessingStatus(
            book_uuid=status['book_uuid'],
            status=status['status'],
            total_pages=status['total_pages'],
            total_chunks=status['total_chunks'],
            processed_at=status['processed_at'],
            error_message=status['error_message']
        )

@app.delete("/pdf/{uuid}")
async def delete_pdf(uuid: str):
    """Delete PDF data and embeddings"""
    try:
        uuid_lib.UUID(uuid)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    
    async with db_pool.acquire() as conn:
        chunks_deleted = await conn.execute(
            "DELETE FROM pdf_chunks WHERE book_uuid = $1",
            uuid
        )
        
        status_deleted = await conn.execute(
            "DELETE FROM pdf_processing_status WHERE book_uuid = $1",
            uuid
        )
        
        return {
            "status": "success",
            "message": f"Deleted PDF data for UUID {uuid}",
            "chunks_deleted": chunks_deleted,
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Chat with PDF RAG API is running with OPTIMIZED PRE-INITIALIZED MODELS",
        "version": "1.0.0",
        "status": "healthy",
        "optimization": "pre-initialized embedding + chat models"
    }

@app.get("/health")
async def health_check():
    """Detailed health check with OPTIMIZED model status"""
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    # Test each OPTIMIZED specialized manager
    try:
        embedding_status = "healthy" if len(embedding_batch_manager.models) > 0 else "unhealthy"
        query_status = "healthy" if len(query_embedding_manager.models) > 0 else "unhealthy" 
        chat_status = "healthy" if len(chat_response_manager.models) > 0 else "unhealthy"
        
        # Optional: Test actual OPTIMIZED API calls
        try:
            test_embedding = embedding_batch_manager.generate_embedding_optimized("test", "retrieval_query")
            embedding_status = "healthy" if test_embedding else "unhealthy"
        except:
            embedding_status = "unhealthy"
        
        try:
            test_query = query_embedding_manager.generate_embedding_optimized("test query", "retrieval_query")
            query_status = "healthy" if test_query else "unhealthy"
        except:
            query_status = "unhealthy"
            
        try:
            test_chat = chat_response_manager.generate_chat_response("Test prompt")
            chat_status = "healthy" if test_chat else "unhealthy"
        except:
            chat_status = "unhealthy"
            
    except Exception as e:
        logger.error(f"OPTIMIZED health check error: {str(e)}")
        embedding_status = query_status = chat_status = "unhealthy"
    
    return {
        "database": db_status,
        "embedding_batch_manager": {
            "status": embedding_status,
            "working_models": len(embedding_batch_manager.models),
            "keys_range": "1-3",
            "optimization": "pre-initialized embedding models"
        },
        "query_embedding_manager": {
            "status": query_status,
            "working_models": len(query_embedding_manager.models),
            "keys_range": "4-6",
            "optimization": "pre-initialized embedding models"
        },
        "chat_response_manager": {
            "status": chat_status,
            "working_models": len(chat_response_manager.models),
            "keys_range": "7-9",
            "optimization": "pre-initialized chat + embedding models"
        },
        "total_api_keys": len(ALL_GEMINI_KEYS),
        "overall_optimization": "All managers use pre-initialized models",
        "performance_gain": "Reduced API initialization overhead",
        "overall": "healthy" if all([
            db_status == "healthy",
            embedding_status == "healthy",
            query_status == "healthy", 
            chat_status == "healthy"
        ]) else "unhealthy"
    }   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)