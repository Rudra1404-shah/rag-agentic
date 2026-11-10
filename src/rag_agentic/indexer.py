import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
JSONL_FILE = DATA_PROCESSED / "statute_sections.jsonl"

# Legal abbreviation map
ACT_ACRONYMS = {
    "bns": "Bharatiya Nyaya Sanhita",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita",
    "bsa": "Bharatiya Sakshya Adhiniyam",
    "ipc": "Bharatiya Nyaya Sanhita",
    "crpc": "Bharatiya Nagarik Suraksha Sanhita",
    "iea": "Bharatiya Sakshya Adhiniyam",
}

def get_short_code(act_name: str) -> str:
    """Returns standard legal acronym for metadata tagging."""
    act_lower = act_name.lower()
    if "nyaya" in act_lower:
        return "BNS"
    if "nagarik" in act_lower or "suraksha" in act_lower:
        return "BNSS"
    if "sakshya" in act_lower:
        return "BSA"
    return ""


def load_documents_clean() -> List[Document]:
    """
    Loads JSONL documents, cleans trailing text, and injects 
    act acronyms into content so BM25 can match shortcodes.
    """
    if not JSONL_FILE.exists():
        raise FileNotFoundError(f"Missing {JSONL_FILE}. Run run_ingestion.py first.")

    raw_docs = []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            content = item["page_content"].strip()
            metadata = item["metadata"]

            # Trim trailing next-section title artifact if present
            lines = content.splitlines()
            if lines and len(lines[-1].split()) <= 6 and not lines[-1].endswith((".", "—", ":", ";", ")", "”", '"')):
                content = "\n".join(lines[:-1]).strip()

            short_code = get_short_code(metadata.get("act", ""))
            metadata["short_code"] = short_code

            # Prepend acronym keyword so BM25 indexes both full name and short code
            # Example: [BNS | Bharatiya Nyaya Sanhita, 2023 | Section 103]
            if short_code and f"[{short_code}" not in content:
                content = f"[{short_code} | Section {metadata.get('section', '')}]\n" + content

            raw_docs.append(Document(page_content=content, metadata=metadata))

    print(f"[*] Loaded and enriched {len(raw_docs)} legal sections from JSONL.")
    return raw_docs


class ReciprocalRankFusionRetriever(BaseRetriever):
    """
    Custom RRF Hybrid Retriever with exact citation resolution.
    """
    bm25_retriever: BaseRetriever
    vector_retriever: BaseRetriever
    top_k: int = 4
    c: int = 60
    bm25_weight: float = 0.5
    vector_weight: float = 0.5

    def _extract_citation_intent(self, query: str) -> Optional[Dict[str, str]]:
        """
        Detects exact citation intent like 'Section 103 BNS' or 'BNS Section 103'.
        """
        # Match 'Section 103 BNS' or 'Sec 103 BNSS'
        pattern1 = re.search(r"\b(?:section|sec\.?)\s*(\d+[A-Z]?)\s+([a-zA-Z]+)\b", query, re.IGNORECASE)
        # Match 'BNS Section 103' or 'BSA Sec 65'
        pattern2 = re.search(r"\b([a-zA-Z]+)\s+(?:section|sec\.?)\s*(\d+[A-Z]?)\b", query, re.IGNORECASE)

        sec_num, act_code = None, None
        if pattern1:
            sec_num = pattern1.group(1).strip()
            act_code = pattern1.group(2).strip().lower()
        elif pattern2:
            act_code = pattern2.group(1).strip().lower()
            sec_num = pattern2.group(2).strip()

        if sec_num and act_code in ACT_ACRONYMS:
            return {"section": sec_num, "short_code": get_short_code(ACT_ACRONYMS[act_code])}
        return None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # Preprocess query: expand legal acronyms for BM25 and Vector search
        expanded_query = query
        for acronym, full_name in ACT_ACRONYMS.items():
            expanded_query = re.sub(
                rf"\b{acronym}\b", 
                f"{acronym.upper()} {full_name}", 
                expanded_query, 
                flags=re.IGNORECASE
            )

        # 1. Check for exact citation intent
        citation = self._extract_citation_intent(query)
        exact_doc = None

        if citation:
            target_sec = citation["section"]
            target_code = citation["short_code"]
            # Search candidate pool for exact metadata match
            for doc in self.bm25_retriever.docs:  # type: ignore
                if (
                    str(doc.metadata.get("section", "")).strip().lower() == target_sec.lower()
                    and doc.metadata.get("short_code", "") == target_code
                ):
                    exact_doc = doc
                    break

        # 2. Run standard BM25 and Vector queries using expanded query
        bm25_docs = self.bm25_retriever.invoke(expanded_query)
        vector_docs = self.vector_retriever.invoke(expanded_query)

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # If exact citation was found, give it top priority boost
        if exact_doc:
            exact_id = exact_doc.metadata.get("section", "") + "_" + exact_doc.metadata.get("act", "")
            doc_map[exact_id] = exact_doc
            rrf_scores[exact_id] = 2.0  # Dominant score

        # Score BM25 results
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("section", "") + "_" + doc.metadata.get("act", "")
            doc_map[doc_id] = doc
            score = self.bm25_weight * (1.0 / (self.c + rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score

        # Score Vector results
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("section", "") + "_" + doc.metadata.get("act", "")
            doc_map[doc_id] = doc
            score = self.vector_weight * (1.0 / (self.c + rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids[: self.top_k]]


def get_or_build_vectorstore(docs: List[Document]) -> Chroma:
    """Builds or loads persistent Chroma vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"[*] Loading Chroma vector store from: {CHROMA_DIR}")
        return Chroma(
            collection_name="indian_statutes",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR)
        )

    print(f"[*] Building Chroma vector store at: {CHROMA_DIR}")
    vectorstore = Chroma(
        collection_name="indian_statutes",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    batch_size = 100
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding batches to Chroma"):
        batch = docs[i : i + batch_size]
        vectorstore.add_documents(batch)

    print("[+] Vector store build complete.")
    return vectorstore


def get_hybrid_retriever(top_k: int = 4) -> BaseRetriever:
    """Builds hybrid retriever with citation handling."""
    docs = load_documents_clean()

    print("[*] Initializing BM25 index with acronym awareness...")
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = top_k * 2

    vectorstore = get_or_build_vectorstore(docs)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k * 2}
    )

    return ReciprocalRankFusionRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector_retriever,
        top_k=top_k,
        bm25_weight=0.5,
        vector_weight=0.5
    )


if __name__ == "__main__":
    retriever = get_hybrid_retriever()
    print("[+] Hybrid retriever ready.")