import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from typing import List, Union
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import shutil
from tempfile import NamedTemporaryFile

load_dotenv()


class EnhancedRAGPipeline:
    """
    Enhanced RAG Pipeline with:
    - Dynamic file upload (PDF + TXT)
    - Conversation memory
    - Multi-query retrieval
    """

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.retriever = None

        # REPLACES ConversationBufferMemory
        self.chat_history: List[Union[HumanMessage, AIMessage]] = []

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Initialize LLMs (keep your model)
        self.query_llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.3)
        self.answer_llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.2)

        print("Enhanced RAG Pipeline initialized!")

    def parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def parse_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text for RAG retrieval"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        return splitter.split_text(text)

    def process_files(self, file_paths: List[str]) -> int:
        """
        Process multiple PDF and TXT files dynamically

        Args:
            file_paths: List of file paths (PDF or TXT)

        Returns:
            Total number of chunks created
        """
        all_chunks = []

        for file_path in file_paths:
            file_ext = Path(file_path).suffix.lower()

            try:
                if file_ext == '.pdf':
                    print(f"Reading PDF: {file_path}")
                    text = self.parse_pdf(file_path)
                elif file_ext == '.txt':
                    print(f"Reading TXT: {file_path}")
                    text = self.parse_txt(file_path)
                else:
                    print(f"Unsupported file type: {file_path}")
                    continue

                chunks = self.chunk_text(text)
                all_chunks.extend(chunks)
                print(f"   Created {len(chunks)} chunks")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        # Build vector store
        if all_chunks:
            documents = [Document(page_content=chunk) for chunk in all_chunks]

            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            else:
                # Add to existing vector store
                self.vector_store.add_documents(documents)

            # Update retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 15,
                    "lambda_mult": 0.7
                }
            )

            print(f"\nTotal chunks in vector store: {len(all_chunks)}")

        return len(all_chunks)

    def save_vector_store(self, path: str = "faiss_store_enhanced"):
        """Save the vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"Vector store saved to {path}")

    def load_vector_store(self, path: str = "faiss_store_enhanced"):
        """Load vector store from disk"""
        try:
            self.vector_store = FAISS.load_local(
                path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 15,
                    "lambda_mult": 0.7
                }
            )
            print(f"Vector store loaded from {path}")
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")

    def generate_multi_queries(self, user_query: str) -> List[str]:
        """Generate multiple search queries from user question"""
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""Rephrase the following question into 3 different, yet related, search queries. List them without any explanation or numbering.

Question: {question}
"""
        )

        # REPLACES LLMChain
        chain = prompt | self.query_llm | StrOutputParser()
        output = chain.invoke({"question": user_query})
        queries = [line.strip() for line in output.strip().split("\n") if line.strip()]
        return queries[:3]

    def multi_query_retrieve(self, user_query: str, top_n: int = 5) -> List[Document]:
        """Retrieve documents using multi-query strategy"""
        if not self.retriever:
            raise ValueError("No documents loaded! Please process files first.")

        # Generate reformulated queries
        queries = self.generate_multi_queries(user_query)

        # Retrieve documents
        all_docs = []

        # Original query
        all_docs.extend(self.retriever.invoke(user_query))

        # Reformulated queries
        for query in queries:
            all_docs.extend(self.retriever.invoke(query))

        # Deduplicate
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

        return unique_docs[:top_n]

    def format_docs(self, docs: List[Document]) -> str:
        """Format documents for context"""
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, question: str) -> dict:
        """
        Ask a question with memory

        Returns:
            dict with 'answer' and 'source_documents'
        """
        if not self.retriever:
            return {
                "answer": "No documents loaded. Please upload files first.",
                "source_documents": []
            }

        # Retrieve relevant documents
        docs = self.multi_query_retrieve(question)
        context = self.format_docs(docs)

        # Build history text from in-class memory (last 2 exchanges)
        history_text = ""
        if self.chat_history:
            history_text = "\n".join([
                f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in self.chat_history[-4:]
            ])

        # Create prompt with memory
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""You are a knowledgeable AI assistant. Answer questions based on the provided context and conversation history.

Instructions:
- Use the context and conversation history to construct your answer.
- If the context is insufficient, say "The provided context does not contain enough information".
- Be concise, factual, and avoid speculation.
- Reference previous parts of the conversation when relevant.
- Format your answer as readable markdown with short paragraphs and bullet lists where helpful.
- Do NOT use markdown tables (no content starting with | col1 | col2 | etc.).
- Avoid HTML tags like <br>; use normal line breaks instead.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""
        )

        # Generate answer
        chain = prompt | self.answer_llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "chat_history": history_text,
            "question": question
        })

        # Store in simple memory
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "source_documents": docs
        }

    def clear_memory(self):
        """Clear conversation memory"""
        self.chat_history = []
        print("Memory cleared!")

    def get_chat_history(self) -> List[str]:
        """Get formatted chat history"""
        if self.chat_history:
            return [
                f"{'You' if isinstance(msg, HumanMessage) else 'Bot'}: {msg.content}"
                for msg in self.chat_history
            ]
        return []


app = FastAPI(title="Enhanced RAG Pipeline API", version="1.0.0")

# Initialize RAG pipeline globally
rag_pipeline = EnhancedRAGPipeline()


# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    source_documents: List[str]


class UploadResponse(BaseModel):
    message: str
    files_processed: int
    total_chunks: int


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Enhanced RAG Pipeline API",
        "endpoints": {
            "/upload": "POST - Upload PDF/TXT files",
            "/ask": "POST - Ask a question",
            "/history": "GET - Get chat history",
            "/clear": "POST - Clear chat memory",
            "/health": "GET - Health check"
        }
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDF or TXT files
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    temp_file_paths = []

    try:
        # Save uploaded files temporarily
        for file in files:
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ['.pdf', '.txt']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}. Only PDF and TXT files are allowed."
                )

            # Create temporary file
            with NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_paths.append(temp_file.name)

        # Process files
        total_chunks = rag_pipeline.process_files(temp_file_paths)

        return UploadResponse(
            message="Files processed successfully",
            files_processed=len(files),
            total_chunks=total_chunks
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

    finally:
        # Clean up temporary files
        for temp_path in temp_file_paths:
            try:
                os.unlink(temp_path)
            except:
                pass


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the RAG pipeline
    """
    try:
        result = rag_pipeline.ask(request.question)

        return AnswerResponse(
            answer=result["answer"],
            source_documents=[doc.page_content[:200] + "..." for doc in result["source_documents"]]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/history")
async def get_history():
    """
    Get chat history
    """
    try:
        history = rag_pipeline.get_chat_history()
        return {"chat_history": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


@app.post("/clear")
async def clear_memory():
    """
    Clear conversation memory
    """
    try:
        rag_pipeline.clear_memory()
        return {"message": "Memory cleared successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "vector_store_loaded": rag_pipeline.vector_store is not None,
        "retriever_ready": rag_pipeline.retriever is not None
    }


# ===========================
# Run the application
# ===========================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Enhanced RAG Pipeline API Server")
    print("="*50)
    print("\nAPI Endpoints:")
    print("   - http://localhost:8000/docs (Swagger UI)")
    print("   - http://localhost:8000/redoc (ReDoc)")
    print("   - http://localhost:8000/ (Root)")
    print("\n" + "="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
