import fitz
import os

def debug_healthians():
    """
    This diagnostic script extracts and prints all text from the Healthians PDF
    so we can analyze its structure.
    """
    pdf_to_debug = 'data/Anushtha Kushwaha_report.pdf'
    
    print(f"--- Starting Diagnostic for: {pdf_to_debug} ---")
    full_text = ""
    try:
        with fitz.open(pdf_to_debug) as doc:
            for page in doc:
                full_text += page.get_text() + "\n--- PAGE BREAK ---\n"
        
        print("\n--- RAW TEXT OUTPUT BEGIN ---")
        print(full_text)
        print("--- RAW TEXT OUTPUT END ---")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    debug_healthians()