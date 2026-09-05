from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitting(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    doc = splitter.split_documents(documents)
    return doc