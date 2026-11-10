import pymupdf
import re
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

def diagnose_pdf(pdf_name: str):
    pdf_path = DATA_RAW / pdf_name
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return

    doc = pymupdf.open(pdf_path)
    print(f"\n{'='*60}")
    print(f"Diagnosing: {pdf_name} (Total Pages: {len(doc)})")
    print(f"{'='*60}")

    # 1. Check if the PDF has actual extractable text or is a scanned image
    non_empty_pages = 0
    total_characters = 0
    for page_idx in range(min(15, len(doc))):
        t = doc[page_idx].get_text("text").strip()
        if t:
            non_empty_pages += 1
            total_characters += len(t)

    print(f"[*] Text density in first 15 pages: {non_empty_pages}/15 pages have text ({total_characters} total chars).")
    if total_characters == 0:
        print(f"    [!] WARNING: {pdf_name} appears to be a SCANNED image PDF (needs OCR).")
        return

    # 2. Search for the substantive Section 1 (looking for "Short title")
    print("[*] Searching for substantive Section 1...")
    found_substantive = False
    for page_idx in range(len(doc)):
        text = doc[page_idx].get_text("text")
        
        # In Indian Acts, the substantive Section 1 contains:
        # "This Act may be called the Bharatiya..." or "Short title, commencement"
        if "may be called the" in text or "Short title" in text:
            # Check if this is NOT just the Table of Contents
            if "BE it enacted by Parliament" in text or "An Act to" in text or "may be called" in text:
                print(f"    [+] Found Substantive Section 1 on PAGE {page_idx + 1}!")
                print("    --- Line Breakdown ---")
                lines = [l for l in text.splitlines() if l.strip()]
                # Print 20 lines around this area
                for idx, line in enumerate(lines):
                    if "may be called" in line or "Short title" in line:
                        start_idx = max(0, idx - 3)
                        end_idx = min(len(lines), idx + 12)
                        for snippet_line in lines[start_idx:end_idx]:
                            print(f"        repr: {repr(snippet_line)}")
                        break
                found_substantive = True
                break

    if not found_substantive:
        print("    [-] Could not locate substantive Section 1 using standard markers.")
        # Print a non-empty page sample
        for page_idx in range(len(doc)):
            txt = doc[page_idx].get_text("text").strip()
            if len(txt) > 200:
                print(f"    [Sample Page {page_idx + 1} (length {len(txt)})]:")
                for line in txt.splitlines()[:15]:
                    print(f"        {repr(line)}")
                break

if __name__ == "__main__":
    for act in ["bns.pdf", "bnss.pdf", "bsa.pdf"]:
        diagnose_pdf(act)