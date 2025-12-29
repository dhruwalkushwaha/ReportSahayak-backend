# Use a lightweight Python version
FROM python:3.10-slim

# 1. Install Tesseract OCR (The critical step for reading scanned PDFs)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up the app directory
WORKDIR /app

# 3. Copy dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your code
COPY . .

# 5. Run the app
# (Host 0.0.0.0 is required for cloud deployment)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]