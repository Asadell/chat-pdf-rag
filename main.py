import os
import io
import asyncio
import logging
import itertools
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid as uuid_lib
import re

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance, ImageFilter
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

# Enhanced text processing imports with safe fallbacks
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe NLTK resource download with fallback functions
def download_nltk_resources():
    """Download required NLTK resources with proper error handling"""
    resources = ['punkt', 'punkt_tab', 'stopwords']
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"✓ Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.warning(f"⚠ Failed to download NLTK resource {resource}: {str(e)}")

def fallback_sent_tokenize(text):
    """Fallback sentence tokenization using regex patterns"""
    if not text:
        return []
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def safe_sent_tokenize(text):
    """Safe sentence tokenization with fallback"""
    try:
        return sent_tokenize(text)
    except (LookupError, OSError) as e:
        logger.warning(f"NLTK sent_tokenize failed, using fallback: {str(e)}")
        return fallback_sent_tokenize(text)
    except Exception as e:
        logger.error(f"Sentence tokenization error: {str(e)}")
        return fallback_sent_tokenize(text)

# Download NLTK resources at startup
try:
    download_nltk_resources()
except Exception as e:
    logger.warning(f"NLTK resource download failed: {str(e)}")

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
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_CHUNKS_PER_REQUEST = int(os.getenv("MAX_CHUNKS_PER_REQUEST", "5"))

if not ALL_GEMINI_KEYS or len(ALL_GEMINI_KEYS) < 9:
    raise ValueError("9 GEMINI_API_KEYs are required (GEMINI_API_KEY_1 to GEMINI_API_KEY_9)")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# Enhanced Model Manager (KEEP AS IS)
class SpecializedModelManager:
    def __init__(self, api_keys: List[str], function_name: str, need_chat_model: bool = False):
        if not api_keys:
            raise ValueError(f"At least one API key is required for {function_name}")
        
        self.api_keys = api_keys
        self.current_index = 0
        self.function_name = function_name
        self.models = {}
        
        # Pre-initialize models
        logger.info(f"Pre-initializing {function_name} models...")
        successful_keys = []
        
        for i, key in enumerate(api_keys):
            try:
                genai.configure(api_key=key)
                
                # Test API key
                test_result = genai.embed_content(
                    model="models/text-embedding-004",
                    content="test",
                    task_type="retrieval_query"
                )
                
                model_info = {
                    'api_key': key,
                    'key_index': i + 1,
                    'embedding_model': genai.get_model('models/text-embedding-004')
                }
                
                if need_chat_model:
                    model_info['chat_model'] = genai.GenerativeModel('gemini-2.0-flash-lite')
                
                self.models[key] = model_info
                successful_keys.append(key)
                logger.info(f"✓ {function_name} Model {i + 1} initialized")
                
            except Exception as e:
                logger.error(f"✗ {function_name} Model {i + 1} failed: {str(e)}")
                continue
        
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
    
    def generate_embedding_optimized(self, content, task_type="retrieval_document"):
        """Generate embedding with optimized approach"""
        for attempt in range(len(self.api_keys)):
            model_info = self.get_next_model_info()
            
            try:
                genai.configure(api_key=model_info['api_key'])
                
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
        """Generate chat response with pre-initialized models"""
        if 'chat_model' not in list(self.models.values())[0]:
            raise ValueError("Chat models not initialized for this manager")
        
        for attempt in range(len(self.api_keys)):
            model_info = self.get_next_model_info()
            
            try:
                genai.configure(api_key=model_info['api_key'])
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
        
        logger.error(f"{self.function_name}: All API keys failed, returning fallback")
        return "Maaf, saya sedang mengalami kendala teknis. Silakan coba lagi dalam beberapa menit."

# Initialize specialized managers (KEEP AS IS)
embedding_batch_manager = SpecializedModelManager(
    EMBEDDING_BATCH_KEYS, "EmbeddingBatch", need_chat_model=False
)

query_embedding_manager = SpecializedModelManager(
    QUERY_EMBEDDING_KEYS, "QueryEmbedding", need_chat_model=False
)

chat_response_manager = SpecializedModelManager(
    CHAT_RESPONSE_KEYS, "ChatResponse", need_chat_model=True
)

logger.info("Initialized all specialized managers with pre-initialized models")

# Database connection pool
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    async def init_connection(conn):
        await register_vector(conn)
    
    db_pool = await asyncpg.create_pool(
        DATABASE_URL, 
        min_size=5, 
        max_size=20,
        init=init_connection
    )
    
    logger.info("Database pool initialized with pgvector support")
    
    yield
    await db_pool.close()

app = FastAPI(
    title="Simple Chat with PDF RAG API",
    description="Back to basics: Simple vector search + Gemini response",
    version="2.1.0",
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

def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Fix common OCR errors
    ocr_fixes = {
        '|': 'I',
        'ﬂ': 'fl',
        'ﬁ': 'fi',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        '~': '-',
        '`': "'",
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
    }
    
    for old, new in ocr_fixes.items():
        text = text.replace(old, new)
    
    # Remove control characters but keep newlines and tabs
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text

def enhance_image_for_ocr(image):
    """Enhance image for better OCR results"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    # Apply slight blur to reduce noise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image

def extract_text_from_pdf(pdf_bytes: bytes) -> Dict[int, str]:
    """Extract text from PDF with OCR fallback"""
    page_texts = {}
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Try text extraction first
        text = page.get_text("text")
        text = clean_extracted_text(text)
        
        # If no good text found, use OCR
        if not text or len(text) < 100:
            logger.info(f"Page {page_num + 1}: Using OCR for text extraction")
            try:
                images = convert_from_bytes(
                    pdf_bytes, 
                    first_page=page_num + 1, 
                    last_page=page_num + 1,
                    dpi=300,
                    grayscale=True,
                    use_pdftocairo=True
                )
                
                if images:
                    # Enhance image for better OCR
                    enhanced_image = enhance_image_for_ocr(images[0])
                    
                    # Enhanced OCR configuration
                    ocr_config = '--psm 6 --oem 3 -c preserve_interword_spaces=1'
                    ocr_text = pytesseract.image_to_string(
                        enhanced_image, 
                        lang="ind+eng",
                        config=ocr_config
                    )
                    
                    cleaned_ocr = clean_extracted_text(ocr_text)
                    if cleaned_ocr and len(cleaned_ocr) > 50:
                        text = cleaned_ocr
                        logger.info(f"Page {page_num + 1}: OCR extracted {len(text)} characters")
                
            except Exception as e:
                logger.error(f"OCR failed for page {page_num + 1}: {str(e)}")
        
        if text:
            page_texts[page_num + 1] = text
            logger.info(f"Page {page_num + 1}: Extracted {len(text)} characters")
        else:
            logger.warning(f"Page {page_num + 1}: No text could be extracted")
    
    doc.close()
    return page_texts

def create_semantic_chunks(page_texts: Dict[int, str]) -> List[Dict[str, Any]]:
    """Create chunks with sentence boundaries"""
    chunks = []
    
    for page_num, text in page_texts.items():
        if not text or len(text.strip()) < 50:
            continue
        
        try:
            # Split by sentences
            sentences = safe_sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {str(e)}, using fallback")
            sentences = text.split('. ')
        
        current_chunk = []
        current_token_count = 0
        chunk_index = 1
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Ensure sentence ends properly
            if sentence and not sentence.endswith(('.', '!', '?', ';', ':')):
                sentence += '.'
            
            sentence_tokens = count_tokens(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if current_token_count + sentence_tokens > CHUNK_SIZE and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "token_count": current_token_count,
                    "sentence_count": len(current_chunk)
                })
                
                # Create overlap for next chunk
                overlap_size = min(CHUNK_OVERLAP // 10, len(current_chunk) // 2)
                if overlap_size > 0:
                    overlap_sentences = current_chunk[-overlap_size:]
                    overlap_tokens = sum(count_tokens(s) for s in overlap_sentences)
                    current_chunk = overlap_sentences.copy()
                    current_token_count = overlap_tokens
                else:
                    current_chunk = []
                    current_token_count = 0
                
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if count_tokens(chunk_text) > 30:  # Minimum chunk size
                chunks.append({
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "token_count": current_token_count,
                    "sentence_count": len(current_chunk)
                })
    
    logger.info(f"Created {len(chunks)} semantic chunks from {len(page_texts)} pages")
    return chunks

# SIMPLIFIED FUNCTIONS - BACK TO BASICS

def clean_response_text(text: str) -> str:
    """Clean and format response text for better readability"""
    if not text:
        return text
    
    # Remove excessive markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold **text**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove italic *text*
    text = re.sub(r'#{1,6}\s*([^\n]+)', r'\1', text)  # Remove headers
    
    # Clean up line breaks and spacing
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive line breaks
    text = re.sub(r'[ \t]+', ' ', text)     # Multiple spaces to single space
    
    # Fix numbered lists - make them more natural
    text = re.sub(r'(\d+)\.\s*\*([^*\n]+)\*:', r'\1. \2:', text)
    text = re.sub(r'(\d+)\.\s*\*([^*\n]+)\*', r'\1. \2', text)
    
    # Remove bullet points and replace with natural flow
    text = re.sub(r'^\s*[-*•]\s*', '', text, flags=re.MULTILINE)
    
    # Clean up any remaining formatting artifacts
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code formatting
    text = text.replace('\\n', '\n')  # Fix escaped newlines
    
    # Final cleanup
    text = text.strip()
    
    return text

def build_simple_prompt(user_prompt: str, context_chunks: List[Dict], language: str) -> str:
    """Build simple and effective prompt"""
    
    # Structure context
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        context_parts.append(f"""
[Informasi dari halaman {chunk['page_number']}]:
{chunk['content']}
""")
    
    context_text = "\n".join(context_parts)
    
    if language == "id":
        return f"""
Berdasarkan informasi dari dokumen berikut:

{context_text}

Jawab pertanyaan ini dengan bahasa yang natural dan mudah dipahami. Jangan gunakan format markdown, bullet points, atau formatting khusus. Berikan jawaban yang mengalir seperti penjelasan biasa:

{user_prompt}

Jika informasi tidak cukup, jawab dengan jujur bahwa informasi tidak ditemukan dalam dokumen.

Jawaban:
"""
    else:
        return f"""
Based on the following document information:

{context_text}

Answer this question in natural, easy-to-understand language. Don't use markdown formatting, bullet points, or special formatting. Provide a flowing explanation like a normal conversation:

{user_prompt}

If information is insufficient, honestly state that the information is not found in the document.

Answer:
"""

async def retrieve_similar_chunks(book_uuid: str, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """Simple retrieval based on vector similarity only"""
    async with db_pool.acquire() as conn:
        chunks = await conn.fetch(
            """
            SELECT id, page_number, chunk_index, content,
                   embedding <-> $2::vector as distance
            FROM pdf_chunks
            WHERE book_uuid = $1
            ORDER BY embedding <-> $2::vector
            LIMIT $3
            """,
            book_uuid, query_embedding, top_k
        )
        
        return [dict(chunk) for chunk in chunks]

# Enhanced API functions (keep existing ones that work)
async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using optimized batch processing"""
    embeddings = []
    batch_size = 50
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            result = embedding_batch_manager.generate_embedding_optimized(batch, "retrieval_document")
            
            if "embedding" in result and result["embedding"]:
                if isinstance(result["embedding"][0], list):
                    embeddings.extend(result["embedding"])
                else:
                    embeddings.append(result["embedding"])
                    
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            # Add zero embeddings as fallback
            embeddings.extend([[0.0] * 768 for _ in batch])
        
        await asyncio.sleep(0.5)  # Rate limiting
    
    return embeddings

def generate_chat_response_with_balancer(prompt: str) -> str:
    """Generate chat response using optimized model balancer"""
    return chat_response_manager.generate_chat_response(prompt)

# API Endpoints

@app.post("/upload_pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    uuid: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload and process PDF"""
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
    """Background PDF processing"""
    start_time = datetime.now()
    
    try:
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
        
        # Text extraction
        logger.info(f"Starting text extraction for {book_uuid}")
        page_texts = extract_text_from_pdf(pdf_bytes)
        
        # Chunking
        logger.info(f"Creating chunks for {book_uuid}")
        chunks = create_semantic_chunks(page_texts)
        
        if not chunks:
            raise Exception("No text could be extracted from PDF")
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = await generate_embeddings_batch(chunk_texts)
        
        # Store in database
        logger.info(f"Storing chunks and embeddings to database")
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM pdf_chunks WHERE book_uuid = $1", book_uuid)
            
            for i, chunk in enumerate(chunks):
                embedding_array = embeddings[i]
                
                # Ensure 768 dimensions
                if len(embedding_array) != 768:
                    if len(embedding_array) < 768:
                        embedding_array = embedding_array + [0.0] * (768 - len(embedding_array))
                    else:
                        embedding_array = embedding_array[:768]
                
                await conn.execute(
                    """
                    INSERT INTO pdf_chunks (book_uuid, page_number, chunk_index, content, embedding)
                    VALUES ($1, $2, $3, $4, $5::vector)
                    """,
                    book_uuid, chunk["page_number"], chunk["chunk_index"],
                    chunk["content"], embedding_array
                )
            
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
        
        logger.info(f"Successfully processed PDF {book_uuid} in {processing_time:.2f}s")
        
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

# SIMPLIFIED ASK ENDPOINT - BACK TO BASICS
@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """Simple and effective question answering - Back to basics RAG"""
    start_time = datetime.now()
    
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
        
        # Generate query embedding
        try:
            query_result = query_embedding_manager.generate_embedding_optimized(request.question, "retrieval_query")
            query_embedding = query_result["embedding"]
            
            # Ensure correct dimensions
            if len(query_embedding) != 768:
                if len(query_embedding) < 768:
                    query_embedding = query_embedding + [0.0] * (768 - len(query_embedding))
                else:
                    query_embedding = query_embedding[:768]
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process question")
        
        # Simple vector similarity search
        similar_chunks = await retrieve_similar_chunks(
            request.uuid, query_embedding, MAX_CHUNKS_PER_REQUEST
        )
        
        if not similar_chunks:
            if request.language == "id":
                answer = "Maaf, tidak ditemukan informasi yang relevan dengan pertanyaan Anda dalam dokumen."
            else:
                answer = "Sorry, no relevant information was found for your question in the document."
            
            return ChatResponse(
                status="success",
                answer=answer,
                sources=[],
                response_time=f"{(datetime.now() - start_time).total_seconds():.2f}s"
            )
        
        # Build sources information
        sources = []
        for chunk in similar_chunks:
            sources.append({
                "page_number": chunk['page_number'],
                "chunk_index": chunk['chunk_index'],
                "similarity_score": round(1 - chunk['distance'], 3)
            })
        
        # Generate simple response
        try:
            prompt = build_simple_prompt(request.question, similar_chunks, request.language)
            response_text = generate_chat_response_with_balancer(prompt)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ChatResponse(
                status="success",
                answer=response_text,
                sources=sources,
                response_time=f"{processing_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            
            # Fallback response
            if request.language == "id":
                fallback_answer = f"Berdasarkan dokumen, informasi yang ditemukan: {similar_chunks[0]['content'][:200]}..."
            else:
                fallback_answer = f"Based on the document, found information: {similar_chunks[0]['content'][:200]}..."
            
            return ChatResponse(
                status="success",
                answer=fallback_answer,
                sources=sources,
                response_time=f"{(datetime.now() - start_time).total_seconds():.2f}s"
            )

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
        "message": "Simple Chat with PDF RAG API - Back to Basics",
        "version": "2.1.0",
        "status": "healthy",
        "approach": "Simple vector similarity + Gemini response generation",
        "features": [
            "Basic text extraction with OCR fallback",
            "Sentence-based chunking", 
            "Simple pgvector similarity search",
            "Direct Gemini response generation",
            "Pre-initialized model optimization"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check with basic model status"""
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    # Test specialized managers
    try:
        embedding_status = "healthy" if len(embedding_batch_manager.models) > 0 else "unhealthy"
        query_status = "healthy" if len(query_embedding_manager.models) > 0 else "unhealthy" 
        chat_status = "healthy" if len(chat_response_manager.models) > 0 else "unhealthy"
        
        # Simple API tests
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
            test_chat = chat_response_manager.generate_chat_response("Test: Say OK")
            chat_status = "healthy" if test_chat else "unhealthy"
        except:
            chat_status = "unhealthy"
            
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        embedding_status = query_status = chat_status = "unhealthy"
    
    return {
        "database": db_status,
        "embedding_batch_manager": {
            "status": embedding_status,
            "working_models": len(embedding_batch_manager.models),
            "keys_range": "1-3"
        },
        "query_embedding_manager": {
            "status": query_status,
            "working_models": len(query_embedding_manager.models),
            "keys_range": "4-6"
        },
        "chat_response_manager": {
            "status": chat_status,
            "working_models": len(chat_response_manager.models),
            "keys_range": "7-9"
        },
        "total_api_keys": len(ALL_GEMINI_KEYS),
        "approach": "Simple RAG: Vector search → Gemini response",
        "overall": "healthy" if all([
            db_status == "healthy",
            embedding_status == "healthy",
            query_status == "healthy", 
            chat_status == "healthy"
        ]) else "unhealthy"
    }

@app.get("/stats/{uuid}")
async def get_pdf_stats(uuid: str):
    """Get basic statistics for a processed PDF"""
    try:
        uuid_lib.UUID(uuid)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    
    async with db_pool.acquire() as conn:
        # Get processing status
        status = await conn.fetchrow(
            "SELECT * FROM pdf_processing_status WHERE book_uuid = $1",
            uuid
        )
        
        if not status:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        # Get chunk statistics
        chunk_stats = await conn.fetchrow(
            """
            SELECT 
                COUNT(*) as total_chunks,
                AVG(LENGTH(content)) as avg_chunk_length,
                MIN(LENGTH(content)) as min_chunk_length,
                MAX(LENGTH(content)) as max_chunk_length,
                COUNT(DISTINCT page_number) as pages_with_chunks
            FROM pdf_chunks 
            WHERE book_uuid = $1
            """,
            uuid
        )
        
        # Get page distribution
        page_distribution = await conn.fetch(
            """
            SELECT 
                page_number,
                COUNT(*) as chunk_count,
                AVG(LENGTH(content)) as avg_length
            FROM pdf_chunks 
            WHERE book_uuid = $1
            GROUP BY page_number
            ORDER BY page_number
            """,
            uuid
        )
        
        return {
            "book_uuid": uuid,
            "processing_status": dict(status),
            "chunk_statistics": dict(chunk_stats) if chunk_stats else None,
            "page_distribution": [dict(row) for row in page_distribution],
            "chunking_strategy": "sentence_based",
            "chunk_size_limit": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "max_chunks_per_query": MAX_CHUNKS_PER_REQUEST
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)