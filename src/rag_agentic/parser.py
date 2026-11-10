import pymupdf
import re
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_core.documents import Document

class StatuteParser:
    def __init__(self, act_name: str, pdf_path: str | Path, start_page: int = 1):
        self.act_name = act_name
        self.pdf_path = Path(pdf_path)
        self.start_page = start_page

    def parse_sections(self) -> List[Document]:
        doc = pymupdf.open(self.pdf_path)
        total_pages = len(doc)
        print(f"\nProcessing {self.pdf_path.name} ({total_pages} pages, substantive text starts on page {self.start_page})...")

        raw_lines = []
        header_regex = re.compile(
            r"(THE GAZETTE OF INDIA|EXTRAORDINARY|PART II—Section 1|PUBLISHED BY AUTHORITY|HomePage)",
            re.IGNORECASE
        )
        page_num_regex = re.compile(
            r"^\s*(?:\[?\s*Part II\s*—\s*Sec\. 1\s*\]?|\d+|SEC\.\s*1\])\s*$", 
            re.IGNORECASE
        )

        for page_idx in range(self.start_page - 1, total_pages):
            text = doc[page_idx].get_text("text")
            for line in text.splitlines():
                clean = line.strip()
                if not clean:
                    continue
                if header_regex.search(clean) or page_num_regex.match(clean):
                    continue
                raw_lines.append(clean)

        # RegEx Patterns matching the 3 layouts:
        # Pattern 1 (BSA inline): "1. Short title...–– (1) This Act" or "2. Definitions.— (1) In this"
        pat_bsa = re.compile(
            r"^(\d+[A-Z]?)\.\s+([A-Za-z][^\.–—\-]+?)(?:[\.–—\-]{1,3})\s*(.*)$", 
            re.IGNORECASE
        )
        # Pattern 2 (BNSS title-only line): "1. Short title, extent and commencement"
        pat_bnss = re.compile(
            r"^(\d+[A-Z]?)\.\s+([A-Za-z][A-Za-z0-9\s,\(\)\'\"—\-]+)$", 
            re.IGNORECASE
        )
        # Pattern 3 (BNS title on previous line): "1. (1) This Act may be called..."
        pat_bns = re.compile(
            r"^(\d+[A-Z]?)\.\s+(\([0-9a-zA-Z]+\).*)", 
            re.IGNORECASE
        )

        sections = []
        current_sec_num = None
        current_title = ""
        current_body = []
        seen_sections = set()

        for idx, line in enumerate(tqdm(raw_lines, desc=f"Parsing {self.pdf_path.name}", unit="line")):
            m_bsa = pat_bsa.match(line)
            m_bnss = pat_bnss.match(line)
            m_bns = pat_bns.match(line)

            # Check Pattern 1 (BSA inline)
            if m_bsa:
                sec_cand = m_bsa.group(1).strip()
                title_cand = m_bsa.group(2).strip()
                trailing = m_bsa.group(3).strip()

                if not title_cand.lower().startswith(("of", "and", "or", "to", "for", "in", "by", "the")):
                    if current_sec_num and current_body and current_sec_num not in seen_sections:
                        seen_sections.add(current_sec_num)
                        sections.append(self._create_doc(current_sec_num, current_title, "\n".join(current_body)))

                    current_sec_num = sec_cand
                    current_title = title_cand
                    current_body = [trailing] if trailing else []
                    continue

            # Check Pattern 2 (BNSS header line)
            elif m_bnss:
                sec_cand = m_bnss.group(1).strip()
                title_cand = m_bnss.group(2).strip()

                if not title_cand.lower().startswith(("of", "and", "or", "to", "for", "in", "by")):
                    if current_sec_num and current_body and current_sec_num not in seen_sections:
                        seen_sections.add(current_sec_num)
                        sections.append(self._create_doc(current_sec_num, current_title, "\n".join(current_body)))

                    current_sec_num = sec_cand
                    current_title = title_cand
                    current_body = []
                    continue

            # Check Pattern 3 (BNS title on previous line)
            elif m_bns:
                sec_cand = m_bns.group(1).strip()
                body_start = m_bns.group(2).strip()

                prev_line = raw_lines[idx - 1] if idx > 0 else "General Provision"
                if "CHAPTER" in prev_line or "PRELIMINARY" in prev_line or "HomePage" in prev_line:
                    prev_line = raw_lines[idx - 2] if idx > 1 else "General Provision"

                if current_sec_num and current_body and current_sec_num not in seen_sections:
                    seen_sections.add(current_sec_num)
                    sections.append(self._create_doc(current_sec_num, current_title, "\n".join(current_body)))

                current_sec_num = sec_cand
                current_title = prev_line
                current_body = [body_start]
                continue

            # Accumulate body text
            if current_sec_num:
                current_body.append(line)

        # Flush final section
        if current_sec_num and current_body and current_sec_num not in seen_sections:
            seen_sections.add(current_sec_num)
            sections.append(self._create_doc(current_sec_num, current_title, "\n".join(current_body)))

        print(f"Extracted {len(sections)} sections from {self.pdf_path.name}.")
        return sections

    def _create_doc(self, sec_num: str, title: str, body: str) -> Document:
        clean_title = " ".join(title.split()).strip()
        context_header = f"[{self.act_name} | Section {sec_num}: {clean_title}]"
        full_clause = f"Section {sec_num}. {clean_title}—\n{body.strip()}"
        enriched_content = f"{context_header}\n\n{full_clause}"

        return Document(
            page_content=enriched_content,
            metadata={
                "act": self.act_name,
                "section": sec_num,
                "title": clean_title,
                "doc_type": "statute",
                "source": self.pdf_path.name
            }
        )