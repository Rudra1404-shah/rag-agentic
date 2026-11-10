import sys
import json
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = None

for parent in [CURRENT_DIR, CURRENT_DIR.parent, CURRENT_DIR.parent.parent]:
    if (parent / "data").exists():
        PROJECT_ROOT = parent
        break

sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from rag_agentic.parser import StatuteParser
except ImportError:
    from parser import StatuteParser

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

def main():
    statutes = [
        {"act": "Bharatiya Nyaya Sanhita, 2023", "file": DATA_RAW / "bns.pdf", "start_page": 74},
        {"act": "Bharatiya Nagarik Suraksha Sanhita, 2023", "file": DATA_RAW / "bnss.pdf", "start_page": 31},
        {"act": "Bharatiya Sakshya Adhiniyam, 2023", "file": DATA_RAW / "bsa.pdf", "start_page": 10}
    ]

    all_sections = []

    for item in statutes:
        pdf_path = item["file"]
        if not pdf_path.exists():
            print(f"Skipping {item['act']} - file not found at: {pdf_path}")
            continue

        parser = StatuteParser(
            act_name=item["act"],
            pdf_path=pdf_path,
            start_page=item["start_page"]
        )
        sections = parser.parse_sections()
        all_sections.extend(sections)

    if all_sections:
        output_file = DATA_PROCESSED / "statute_sections.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in all_sections:
                f.write(json.dumps({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }) + "\n")

        print(f"\n[DONE] Successfully saved {len(all_sections)} sections to {output_file}")
        
        # Verification preview
        sample = next((s for s in all_sections if s.metadata["section"] == "1"), all_sections[0])
        print("\n--- Sample Verification ---")
        print(f"Metadata: {sample.metadata}")
        print(f"Content:\n{sample.page_content[:350]}...\n")

if __name__ == "__main__":
    main()