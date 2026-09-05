import re

def clean_data(docs : str):

    for doc in docs:
        text = doc.page_content

        # Normalize backspace
        text = re.sub(r'\s+', ' ', text)

        # Remove Gazette headers/footers
        text = re.sub(r'THE GAZETTE OF INDIA EXTRAORDINARY', '', text)
        text = re.sub(r'MINISTRY OF LAW AND JUSTICE', '', text)

        # Remove page numbers and codes
        text = re.sub(r'SEC\.\s*\d+\]', '', text)
        text = re.sub(r'No\.\s*\d+\]', '', text)
        text = re.sub(r'CG-DL-[A-Z0-9-]+', '', text)

        # Fix hyphenated words
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

        # Remove publishing boilerplate
        text = re.sub(r'PUBLISHED BY AUTHORITY', '', text)
        text = re.sub(r'REGISTERED NO\.[* ]+', '', text)

        text = text.strip()
        doc.page_content = text
    return docs

if __name__ == "__main__":
    from rag_backend.doc_loader_pipeline import doc_loader_and_cleaner
    doc = doc_loader_and_cleaner()
    clean_data = clean_data(doc[1])
    print(clean_data)