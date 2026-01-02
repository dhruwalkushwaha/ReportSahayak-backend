# 1. Use Python 3.11 (Matching your requirements)
FROM python:3.11-slim

# 2. Install Tesseract OCR AND the Hindi Language Pack
# We also add 'libgl1' and 'poppler-utils' which prevents common image errors
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-hin \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory
WORKDIR /app

# 4. Copy the current directory contents
COPY . /app

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Make port 10000 available
EXPOSE 10000

# 7. Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]