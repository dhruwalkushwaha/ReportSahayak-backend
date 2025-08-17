import fitz
import re
import json
import os
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Fetch the API key securely
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is missing! Please set it in .env")

genai.configure(api_key=GOOGLE_API_KEY)

def identify_lab(text):
    text_lower = text.lower()
    if "dr" in text_lower and "lal" in text_lower and "pathlabs" in text_lower:
        return "lal_pathlabs"
    if "healthians" in text_lower and "smart report" in text_lower:
        return "healthians"
    if "apollo" in text_lower and "diagnostics" in text_lower:
        return "apollo"
    if "awadh" in text_lower and "pathology" in text_lower:
        return "awadh"
    return "unknown"

def parse_lal_pathlabs(text):
    results = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    KNOWN_UNITS = {"g/dL", "%", "mill/mm3", "fL", "pg", "thou/mm3", "nmol/L", "U/L"}
    def is_value(s): return re.fullmatch(r"[\d\.<>]+", s) is not None
    def is_range(s): return '-' in s and re.search(r"\d", s)
    def is_unit(s): return s in KNOWN_UNITS

    try:
        start_index = next((i for i, line in enumerate(lines) if re.search(r"Bio\.?\s*Ref\.?\s*Interval", line, re.I)), None)
        if start_index is None: return results
        
        stop_keywords = {"comment", "clinical", "interpretation"}
        end_index = len(lines)
        for i in range(start_index + 1, len(lines)):
            if lines[i].strip().lower() in stop_keywords:
                end_index = i
                break

        table_tokens = lines[start_index + 1:end_index]
        for i in range(0, len(table_tokens), 4):
            chunk = table_tokens[i:i+4]
            if len(chunk) == 4:
                identified_parts, name_parts = {}, []
                for part in chunk:
                    if is_range(part): identified_parts['ref_interval'] = part
                    elif is_unit(part): identified_parts['unit'] = part
                    elif is_value(part): identified_parts['value'] = part
                    else: name_parts.append(part)
                if {'value', 'unit', 'ref_interval'} <= identified_parts.keys():
                    test_name = " ".join(name_parts).strip(" -:;")
                    if test_name:
                        identified_parts['test_name'] = test_name
                        results.append(identified_parts)
    except Exception: pass

    unique_results = []
    seen = set()
    for r in results:
        sig = (r.get('test_name', ''), r.get('value', ''))
        if sig not in seen:
            seen.add(sig)
            unique_results.append(r)
    return unique_results

def parse_apollo(text):
    results = []
    pattern = re.compile(r"^(.*?)\n([\d\.]+)\n([a-zA-Z\dµ/]+)\n(.*?)\n(?:[a-zA-Z\d-]+)?$", re.MULTILINE)
    matches = pattern.findall(text)
    for match in matches:
        test_name = ' '.join(match[0].strip().split('\n')).strip(" -:;")
        if len(test_name) < 3 or test_name.isupper() or re.match(r"^\d", test_name): continue
        if any(k in test_name.lower() for k in ["order id", "age", "reported", "sample", "patient"]): continue
        results.append({
            "test_name": test_name, "value": match[1].strip(), "unit": match[2].strip(),
            "ref_interval": match[3].strip() if match[3].strip() else "N/A"
        })
    return results

def parse_healthians(text):
    results = []
    pattern = re.compile(r"^(.*?)\nMethod:.*?\n(?:Machine:.*?\n)?([\d\.]+)\n([a-zA-Z\d\^/µl]+)\n([\d\.\s-]+)$", re.MULTILINE)
    matches = pattern.findall(text)
    for match in matches:
        test_name = match[0].replace('\n', ' ').strip(" -:;")
        results.append({
            "test_name": test_name, "value": match[1].strip(), "unit": match[2].strip(),
            "ref_interval": match[3].strip()
        })
    return results

# --- Pydantic Models & AI Layer ---
class ReportDataItem(BaseModel):
    test_name: str
    value: str
    unit: str
    ref_interval: str

class ParsedReport(BaseModel):
    lab_name: str
    data: List[ReportDataItem]

async def get_ai_analysis(parsed_data: ParsedReport):
    model = genai.GenerativeModel('gemini-1.5-flash')
    data_string = ""
    for item in parsed_data.data:
        data_string += f"- {item.test_name}: {item.value} {item.unit} (Normal Range: {item.ref_interval})\n"
    prompt = f"""
    **ROLE:** You are ReportSahayak, an AI assistant that explains blood reports simply.
    **TASK:** Analyze the blood test data below and generate a JSON object with "summary", "details", and "disclaimer" keys.
    **DATA:**
    {data_string}
    **JSON OUTPUT FORMAT:**
    - "summary": A brief, one-paragraph overview.
    - "details": A list of objects. For each test, provide: "test_name", "value", "status" (with emoji: "✅ Normal", "🔽 Low", or "🔼 High"), "analogy", and "explanation".
    - "disclaimer": Use the exact text: "This is an AI-generated analysis and is for informational purposes only. It is not a substitute for professional medical advice. Please consult with a qualified doctor for any health concerns."
    Generate only the JSON object.
    """
    try:
        response = await model.generate_content_async(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        ai_json = json.loads(cleaned_response)
        if 'details' not in ai_json or 'summary' not in ai_json:
            raise ValueError("AI response missing required keys.")
        return ai_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

# --- FastAPI App ---
app = FastAPI(title="ReportSahayak API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.post("/upload-report/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = ""
    try:
        with fitz.open(stream=contents, filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF file.")
    
    lab_name = identify_lab(full_text)
    if lab_name == "unknown":
        raise HTTPException(status_code=400, detail="Could not identify the lab.")
        
    parsed_data = []
    if lab_name == "lal_pathlabs":
        parsed_data = parse_lal_pathlabs(full_text)
    elif lab_name == "apollo":
        parsed_data = parse_apollo(full_text)
    elif lab_name == "healthians":
        parsed_data = parse_healthians(full_text)

    if not parsed_data:
        raise HTTPException(status_code=400, detail="Could not parse any results.")
        
    return {"lab_name": lab_name, "data": parsed_data}

@app.post("/analyze-report/")
async def analyze_report(report: ParsedReport):
    analysis = await get_ai_analysis(report)
    return analysis

@app.get("/")
def read_root():
    return {"message": "Welcome to the ReportSahayak API! Old Faithful Engine is online."}