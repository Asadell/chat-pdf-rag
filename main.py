import os
import io
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid as uuid_lib

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import asyncpg
import google.generativeai as genai
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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_CHUNKS_PER_REQUEST = int(os.getenv("MAX_CHUNKS_PER_REQUEST", "5"))

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Database connection pool
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    # Startup
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)
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
    allow_origins=["*"],  # Sesuaikan dengan domain aplikasi Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    uuid: str
    question: str
    language: str = "id"  # Default bahasa Indonesia

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
        return len(text.split())  # Fallback to word count

def extract_text_from_pdf(pdf_bytes: bytes) -> Dict[int, str]:
    """Extract text from PDF with OCR fallback for image pages"""
    page_texts = {}
    
    # Try text extraction first
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        
        if text and len(text) > 50:  # Threshold untuk text yang meaningful
            page_texts[page_num + 1] = text
            logger.info(f"Page {page_num + 1}: Extracted {len(text)} characters via text parsing")
        else:
            # Fallback to OCR
            logger.info(f"Page {page_num + 1}: No text found, using OCR...")
            try:
                # Convert specific page to image
                images = convert_from_bytes(
                    pdf_bytes, 
                    first_page=page_num + 1, 
                    last_page=page_num + 1,
                    dpi=200
                )
                
                if images:
                    # OCR with Indonesian + English support
                    ocr_text = pytesseract.image_to_string(
                        images[0], 
                        lang="ind+eng",
                        config='--psm 6'  # Assume uniform block of text
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
            
        # Simple chunking by token count
        tokens = count_tokens(text)
        
        if tokens <= CHUNK_SIZE:
            # Entire page fits in one chunk
            chunks.append({
                "page_number": page_num,
                "chunk_index": 1,
                "content": text,
                "token_count": tokens
            })
        else:
            # Split page into multiple chunks
            words = text.split()
            chunk_words = []
            chunk_index = 1
            
            for word in words:
                chunk_words.append(word)
                current_text = " ".join(chunk_words)
                
                if count_tokens(current_text) >= CHUNK_SIZE:
                    # Save current chunk
                    chunks.append({
                        "page_number": page_num,
                        "chunk_index": chunk_index,
                        "content": current_text,
                        "token_count": count_tokens(current_text)
                    })
                    
                    # Start new chunk with overlap
                    overlap_words = chunk_words[-CHUNK_OVERLAP:] if len(chunk_words) > CHUNK_OVERLAP else chunk_words
                    chunk_words = overlap_words
                    chunk_index += 1
            
            # Save remaining words as final chunk
            if chunk_words:
                final_text = " ".join(chunk_words)
                if count_tokens(final_text) > 20:  # Minimum chunk size
                    chunks.append({
                        "page_number": page_num,
                        "chunk_index": chunk_index,
                        "content": final_text,
                        "token_count": count_tokens(final_text)
                    })
    
    logger.info(f"Created {len(chunks)} chunks from {len(page_texts)} pages")
    return chunks

async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Gemini in batches"""
    embeddings = []
    batch_size = 100  # Gemini API limit
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Use Gemini embedding model
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type="retrieval_document"
            )
            
            if isinstance(result["embedding"][0], list):
                # Multiple embeddings returned
                embeddings.extend(result["embedding"])
            else:
                # Single embedding returned
                embeddings.append(result["embedding"])
                
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
            # Create zero embeddings as fallback
            embeddings.extend([[0.0] * 768 for _ in batch])
            
        # Rate limiting
        await asyncio.sleep(1)
    
    return embeddings

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
    
    # Validate UUID format
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
    
    # Read PDF bytes
    pdf_bytes = await file.read()
    
    # Add to background processing
    background_tasks.add_task(process_pdf_background, uuid, file.filename, pdf_bytes)
    
    return {
        "status": "accepted",
        "message": "PDF processing started",
        "book_uuid": uuid
    }

async def process_pdf_background(book_uuid: str, filename: str, pdf_bytes: bytes):
    """Background task to process PDF"""
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
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = await generate_embeddings_batch(chunk_texts)
        
        # Store in database
        logger.info(f"Storing chunks and embeddings to database")
        async with db_pool.acquire() as conn:
            # Clear existing chunks for this UUID
            await conn.execute(
                "DELETE FROM pdf_chunks WHERE book_uuid = $1",
                book_uuid
            )
            
            # Insert new chunks
            for i, chunk in enumerate(chunks):
                await conn.execute(
                    """
                    INSERT INTO pdf_chunks (book_uuid, page_number, chunk_index, content, embedding)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    book_uuid,
                    chunk["page_number"],
                    chunk["chunk_index"],
                    chunk["content"],
                    embeddings[i]
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
        
        logger.info(f"Successfully processed PDF {book_uuid} in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error processing PDF {book_uuid}: {str(e)}")
        
        # Update status to failed
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
    """Ask question about a specific PDF"""
    start_time = datetime.now()
    
    # Validate UUID
    try:
        uuid_lib.UUID(request.uuid)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    
    # Check if PDF is processed
    async with db_pool.acquire() as conn:
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
            query_result = genai.embed_content(
                model="models/text-embedding-004",
                content=request.question,
                task_type="retrieval_query"
            )
            query_embedding = query_result["embedding"]
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process question")
        
        # Vector similarity search - Fixed to use L2 distance
        similar_chunks = await conn.fetch(
            """
            SELECT id, page_number, chunk_index, content,
                   embedding <-> $2 as distance
            FROM pdf_chunks
            WHERE book_uuid = $1
            ORDER BY embedding <-> $2
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
                "relevance_score": round(1 - chunk['distance'], 2)  # Convert distance to similarity
            })
        
        # Generate response with Gemini
        try:
            prompt = build_prompt(request.question, merged_text, request.language)
            
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ChatResponse(
                status="success",
                answer=response.text,
                sources=sources,
                response_time=f"{processing_time:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
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
        # Delete chunks
        chunks_deleted = await conn.execute(
            "DELETE FROM pdf_chunks WHERE book_uuid = $1",
            uuid
        )
        
        # Delete processing status
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
        "message": "Chat with PDF RAG API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    try:
        # Test Gemini API
        test_result = genai.embed_content(
            model="models/text-embedding-004",
            content="test",
            task_type="retrieval_query"
        )
        gemini_status = "healthy" if test_result else "unhealthy"
    except:
        gemini_status = "unhealthy"
    
    return {
        "database": db_status,
        "gemini_api": gemini_status,
        "overall": "healthy" if db_status == "healthy" and gemini_status == "healthy" else "unhealthy"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)