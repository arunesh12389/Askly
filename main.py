# main.py
import os
import time
import requests
import tempfile
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your existing, excellent utility functions from utils.py
from utils import extract_text_from_pdf, chunk_text, create_vector_store

# Import your chain and LLM
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set in your .env file.")

# --- Initialize the FastAPI app ---
app = FastAPI()
# --- Set up CORS middleware to allow requests from your MERN app ---
origins = [
    "*"  # Allows all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

# --- Define the data models for API requests ---
class AskRequest(BaseModel):
    question: str
    pdf_url: str

# --- Re-use your chain creation logic from app.py ---
def create_retrieval_chain(retriever, llm):
    """Creates a RetrievalQA chain."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain

# --- Define your API endpoint ---
@app.post("/ask-ai")
def ask_ai_endpoint(request: AskRequest):
    """
    This endpoint downloads a PDF from a URL, processes it,
    and answers a question about it.
    """
    tmp_pdf_path = None # Define outside try block for cleanup
    try:
        # 1. Download the PDF from the Cloudinary URL
        pdf_response = requests.get(request.pdf_url)
        pdf_response.raise_for_status() # Raise an error if download fails

        # 2. Save PDF content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_response.content)
            tmp_pdf_path = tmp_pdf.name

        # 3. Process the PDF using your existing utils.py functions
        extracted_text = extract_text_from_pdf(tmp_pdf_path)
        chunks = chunk_text(extracted_text)
        vector_store = create_vector_store(chunks)

        # 4. Set up the LLM and QA chain
        llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0)
        retriever = vector_store.as_retriever()
        qa_chain = create_retrieval_chain(retriever, llm)

        # 5. Get the answer
        t0 = time.perf_counter()
        response = qa_chain.invoke({"query": request.question})
        t1 = time.perf_counter()
        
        answer = response.get("result") if isinstance(response, dict) else str(response)
        
        # 6. Clean up the temporary file
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)

        # 7. Return the final JSON response to your MERN app
        return {
            "success": True,
            "answer": answer,
            "response_time_seconds": (t1 - t0)
        }

    except Exception as e:
        # Explicitly log the full traceback to Heroku's logs
        logging.exception("An error occurred in /ask-ai endpoint:")
        
        # Clean up the file even if an error occurs
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)
        
        # Return an error message
        return {"success": False, "error": str(e)}

# --- NEW DEBUGGING ENDPOINT ---
@app.get("/test-connection")
def test_groq_connection():
    """
    Tries to connect to the Groq API /models endpoint
    to see if the Heroku dyno can reach it.
    """
    try:
        logging.info("Attempting to connect to Groq API...")
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}"
            }
        )
        
        # Check if the request was successful (even if it's an auth error)
        if response.status_code == 200:
            return {"success": True, "message": "Successfully connected to Groq and got models.", "data": response.json()}
        elif response.status_code == 401:
             return {"success": True, "message": "Connection to Groq successful, but API key is invalid."}
        else:
            return {"success": False, "message": f"Connected to Groq, but got an error.", "status_code": response.status_code, "response": response.text}

    except requests.exceptions.ConnectionError as e:
        logging.exception("Connection test FAILED:")
        # This is the error we suspect is happening
        return {"success": False, "message": "A requests.exceptions.ConnectionError occurred.", "error": str(e)}
    except Exception as e:
        logging.exception("An unknown error occurred during connection test:")
        return {"success": False, "message": "An unknown error occurred.", "error": str(e)}

# Test endpoint
@app.get("/")
def read_root():
    return {"message": "EduVerseAI Microservice is running!"}