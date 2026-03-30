from dotenv import load_dotenv 
from langchain_groq import ChatGroq
import json
import re
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import io

# ---------------------------------------------------------
# 1. Load environment + initialize Groq LLM
# ---------------------------------------------------------

load_dotenv()

llm = ChatGroq(
    model_name="openai/gpt-oss-20b"
)

# ---------------------------------------------------------
# 2. PDF Extraction Function
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_file_bytes):
    """
    Extract text from PDF file bytes
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF: {str(e)}")

# ---------------------------------------------------------
# 3. Prompt Template for Question Generation
# ---------------------------------------------------------

def create_prompt(resume_text, job_description):
    """
    Creates a structured prompt for the LLM to generate questions
    """
    prompt = f"""You are an expert technical interviewer. Based on the candidate's resume and the job description provided, generate interview questions.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

TASK:
Generate exactly 5 MCQ (Multiple Choice Questions) and 5 Text-based questions that assess the candidate's fit for this role.

IMPORTANT: You MUST respond ONLY with valid JSON in the exact format below. Do not include any explanations, markdown formatting, or additional text.

{{
  "mcq_questions": [
    {{
      "question": "Question text here?",
      "options": {{
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
      }},
      "correct_answer": "A",
      "explanation": "Brief explanation of why this is correct"
    }}
  ],
  "text_questions": [
    {{
      "question": "Question text here?",
      "sample_answer": "A comprehensive sample answer that demonstrates what a good response looks like",
      "key_points": ["Point 1", "Point 2", "Point 3"]
    }}
  ]
}}

Generate questions that:
1. Are directly relevant to skills mentioned in the resume
2. Match the requirements in the job description
3. Test both technical knowledge and practical application
4. Vary in difficulty (mix of easy, medium, hard)

Remember: Respond with ONLY the JSON object, nothing else."""
    
    return prompt

# ---------------------------------------------------------
# 4. Custom Function to Parse LLM Response
# ---------------------------------------------------------

def parse_llm_response(response_text):
    """
    Parses the LLM response and extracts JSON
    Handles cases where LLM might add extra text or markdown
    """
    try:
        # Try direct JSON parsing first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from markdown or text
        json_pattern = r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```|(\{.*\})'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            for group in match:
                if group.strip():
                    try:
                        return json.loads(group)
                    except json.JSONDecodeError:
                        continue
        
        # If still fails, raise error
        raise ValueError("Could not extract valid JSON from LLM response")

# ---------------------------------------------------------
# 5. Main Function to Generate Questions
# ---------------------------------------------------------

def generate_interview_questions(resume_text, job_description):
    """
    Main function that takes resume and job description,
    generates questions using LLM, and returns structured JSON
    """
    # Create the prompt
    prompt = create_prompt(resume_text, job_description)
    
    # Get response from LLM
    response = llm.invoke(prompt)
    response_text = response.content
    
    # Parse the response to JSON
    questions_data = parse_llm_response(response_text)
    
    # Validate the structure
    if "mcq_questions" not in questions_data or "text_questions" not in questions_data:
        raise ValueError("LLM response missing required fields")
    
    if len(questions_data["mcq_questions"]) != 5:
        print(f"Warning: Expected 5 MCQs, got {len(questions_data['mcq_questions'])}")
    
    if len(questions_data["text_questions"]) != 5:
        print(f"Warning: Expected 5 text questions, got {len(questions_data['text_questions'])}")
    
    return questions_data

# ---------------------------------------------------------
# 6. FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="Interview Question Generator API",
    description="Generate interview questions from resume PDF and job description",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for response
class QuestionResponse(BaseModel):
    status: str
    data: dict
    message: str

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Interview Question Generator API",
        "version": "1.0.0",
        "description": "Upload resume PDF and provide job description to generate interview questions",
        "endpoint": "/generate-questions (POST)"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "question-generator"}

@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_questions_endpoint(
    resume_pdf: UploadFile = File(..., description="Resume in PDF format"),
    job_description: str = Form(..., description="Job description as text")
):
    """
    Generate interview questions from resume PDF and job description
    
    Args:
        resume_pdf: Resume file in PDF format
        job_description: Job description as text
    
    Returns:
        JSON with 5 MCQ and 5 text-based questions with answers
    """
    try:
        # Validate file type
        if not resume_pdf.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are accepted for resume upload"
            )
        
        # Validate job description
        if not job_description or len(job_description.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Job description must be at least 10 characters long"
            )
        
        # Read and extract text from PDF
        pdf_content = await resume_pdf.read()
        resume_text = extract_text_from_pdf(pdf_content)
        
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from PDF. Please ensure the PDF contains readable text."
            )
        
        # Generate questions
        questions = generate_interview_questions(resume_text, job_description)
        
        return QuestionResponse(
            status="success",
            data=questions,
            message="Questions generated successfully"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating questions: {str(e)}"
        )

# ---------------------------------------------------------
# Run the application
# ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Interview Question Generator API")
    print("="*60)
    print("Server will run on: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)