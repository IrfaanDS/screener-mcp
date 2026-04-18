# MCP Verification: Functional Check
from fastmcp import FastMCP
import pdfplumber, io, base64, json, os
from google import genai
from pydantic import BaseModel

# Initialize MCP Server
mcp = FastMCP("ResumeScreener")

# Recommendation: Use environment variables for API keys
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBXnKYDZaERLaOdWj6P3e3dXUg-8-VZfKQ")
client = genai.Client(api_key=API_KEY)

class ScreeningResult(BaseModel):
    score: int
    decision: str
    reason: str
    key_skills: list[str]
    missing_requirements: list[str]

@mcp.tool()
def screen_resume(file_b64: str, job_desc: str) -> str:
    """
    Extracts text from a base64 encoded PDF resume and evaluates it against a job description.
    Returns a JSON string with score, decision, and reasoning.
    """
    try:
        # 1. Extract PDF Text
        pdf_data = base64.b64decode(file_b64)
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        
        if not text.strip():
            return json.dumps({"error": "No text could be extracted from the PDF."})

        # 2. LLM Inference (Using Gemini 2.0 Flash for speed & cost)
        prompt = f"""
        Analyze the following resume against the job description provided.
        
        Job Description:
        {job_desc}
        
        Resume Text:
        {text}
        
        Respond ONLY with a valid JSON object containing:
        - score: 0-100 (match quality)
        - decision: 'Shortlist' or 'Reject'
        - reason: A brief explanation of why.
        - key_skills: List of matching top skills found.
        - missing_requirements: List of critical gaps.
        """
        
        # Note: Using 'gemini-2.0-flash' or 'gemini-1.5-flash' for reliability
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt,
            config={
                "response_mime_type": "application/json",
            }
        )
        
        return response.text

    except Exception as e:
        return json.dumps({"error": f"Internal Server Error: {str(e)}"})

if __name__ == "__main__":
    # Workato expects the MCP server to be accessible via SSE for remote connections
    mcp.run(transport="sse")
