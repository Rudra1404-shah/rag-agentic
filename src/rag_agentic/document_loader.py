import os
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from bs4 import BeautifulSoup

from dotenv import load_dotenv

load_dotenv()

import os
import warnings
import os
os.environ["USER_AGENT"] = "rag-agentic/1.0"
# 2. Silence the LangChain deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
def load_text_file():
    # Create a temporary text file for demonstration
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello, this is a sample text file.\nThis file is used to demonstrate the TextLoader."
        )
        temp_file_path = temp_file.name

    try:
        # Load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} document(s)")
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")

        # Print the loaded documents
        # for doc in documents:
        #     print("Document Content:")
        #     print(doc)
        #     print(doc.page_content)
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


if __name__ == "__main__":
    # Example: replace load_documents() with whatever function/variable you defined
     load_text_file()  # or your loader.load() call

    # print(f"\n[+] Successfully loaded {len(docs)} document(s).")
    # if docs:
    #     print(f"[+] First document metadata: {docs[0].metadata}")
    #     print(f"[+] First document preview: {docs[0].page_content[:150]}...\n")