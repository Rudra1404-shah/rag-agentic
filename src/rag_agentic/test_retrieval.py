import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables (.env) from project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from rag_agentic.indexer import get_hybrid_retriever

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("[!] Error: OPENAI_API_KEY not found in environment or .env file.")
        return

    retriever = get_hybrid_retriever(top_k=3)

    test_queries = [
        # 1. Exact section citation query (Tests BM25)
        "Section 103 BNS",
        # 2. Concrete offense with punishment query (Tests Hybrid)
        "Punishment for murder under Bharatiya Nyaya Sanhita",
        # 3. Procedural / bail question (Tests BNSS)
        "Provisions regarding anticipatory bail or bail for non-bailable offence",
        # 4. Evidence / admissibility question (Tests BSA)
        "Admissibility of electronic and digital records in evidence",
        # 5. Specialized offense query (Tests BNS)
        "Definition and punishment for petty organised crime"
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: '{q}'")
        print(f"{'='*70}")
        results = retriever.invoke(q)
        for idx, doc in enumerate(results, 1):
            act = doc.metadata.get("act", "Unknown Act")
            sec = doc.metadata.get("section", "N/A")
            title = doc.metadata.get("title", "")
            print(f"[{idx}] {act} | Section {sec}: {title}")
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"    Content: {snippet}...\n")

if __name__ == "__main__":
    main()