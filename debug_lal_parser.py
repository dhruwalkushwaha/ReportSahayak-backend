import fitz
import re
import json

def build_and_verify_parser(text):
    """
    This function contains the correct logic, built from the debug log.
    It prints its actions at every step for verification.
    """
    print("\n--- STARTING V2.2 LOGIC VERIFICATION ---")
    results = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # --- Logic for CBC-style multi-test pages ---
    try:
        # The most reliable anchor is the "Bio. Ref. Interval" header.
        start_index = lines.index("Bio. Ref. Interval") + 1
        end_index = lines.index("Comment")
        
        table_tokens = lines[start_index:end_index]
        print(f"\n[DEBUG] Found CBC-style table with {len(table_tokens)} tokens.")
        
        # The data consistently appears in 4-line chunks: Name, Range, Unit, Value
        for i in range(0, len(table_tokens), 4):
            chunk = table_tokens[i:i+4]
            if len(chunk) == 4:
                # Map the chunk based on the observed order from the debug log
                name, ref_interval, unit, value = chunk
                record = {
                    "test_name": name,
                    "value": value,
                    "unit": unit,
                    "ref_interval": ref_interval
                }
                # A quality check to filter out junk
                if len(name) > 2 and "page" not in name.lower() and "status" not in name.lower():
                    results.append(record)
                    print(f"--> SUCCESS: Assembled record: {json.dumps(record)}")
            else:
                print(f"--> WARN: Found incomplete chunk at end of table: {chunk}")

    except ValueError:
        print("\n[DEBUG] CBC-style table not found on this page. Checking for other formats.")
        pass

    # --- Logic for single-test style pages (like Vitamin D) ---
    try:
        # The test name is the reliable anchor for this format
        start_index = lines.index("VITAMIN D, 25 - HYDROXY, SERUM")
        
        # The data is in the lines immediately following the name
        chunk = lines[start_index : start_index + 5]
        
        # Helper functions to correctly identify the parts
        def is_value(s): return re.fullmatch(r"[\d\.<>]+", s) is not None
        def is_range(s): return '-' in s and re.search(r"\d", s)
        def is_unit(s): return s == "nmol/L"

        identified_parts = {}
        name_parts = []
        for part in chunk:
            if is_range(part): identified_parts['ref_interval'] = part
            elif is_unit(part): identified_parts['unit'] = part
            elif is_value(part): identified_parts['value'] = part
            else: name_parts.append(part)
        
        if 'value' in identified_parts and 'unit' in identified_parts and 'ref_interval' in identified_parts:
            identified_parts['test_name'] = " ".join(name_parts)
            results.append(identified_parts)
            print(f"\n[DEBUG] Found single-test style format.")
            print(f"--> SUCCESS: Assembled record: {json.dumps(identified_parts)}")
        
    except (ValueError, IndexError):
        print("[DEBUG] Single-test style format not found on this page.")
        pass
        
    print(f"\n--- VERIFICATION COMPLETE: Assembled {len(results)} total records. ---")
    return results

# --- Main execution block ---
pdf_to_debug = 'data/184866260_SL_2025-08-01T140700_report.pdf' # The Vitamin D report
# pdf_to_debug = 'data/490405184_Report_F2025-07-29T130400.pdf' # The complex CBC report

print(f"--- Starting Final Diagnostic for: {pdf_to_debug} ---")
full_text = ""
try:
    with fitz.open(pdf_to_debug) as doc:
        for page in doc:
            full_text += page.get_text()
    
    # Run the verification function
    build_and_verify_parser(full_text)

except Exception as e:
    print(f"An error occurred: {e}")