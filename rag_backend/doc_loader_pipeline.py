import os
from langchain_community.document_loaders import PyPDFLoader
from rag_backend.clean_data import clean_data
from rag_backend.splitting import splitting
from rag_backend.vector_store import vector_store

def doc_loader_pipeline(loaded_file :str = "loaded_file.txt", folder_path : str = "documents"):

    loaded_documents = []

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loaded_file = os.path.join(base_dir, loaded_file)
    folder_path = os.path.join(base_dir, folder_path)

    existing = set()

    if os.path.exists(loaded_file):
                with open(loaded_file, "r") as f:
                    existing = set(line.strip() for line in f.readlines())

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            print(f"Skipping file {filename} as it is non-pdf file")
            continue

        if filename in existing:
            print(f"{filename} was already loaded")

        else:
            new_filename = filename

            with open(loaded_file, "a") as f:
                f.write(new_filename + "\n")

            print(f"Loading New file {new_filename}....")
            file_path = os.path.join(folder_path, new_filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            print(f"New file {new_filename} has been loaded")

            print(f"Cleaning the file {new_filename}")
            cleaned_docs = clean_data(docs)
            print("file has been cleared")

            print(f"Chunking the {new_filename}....")
            split_docs = splitting(cleaned_docs)
            print("Chunking has been completed")

            print(f"Creating metadata for {new_filename}....")
            source_type = ["act", "rules"]

            for keyword in source_type:
                 if keyword in new_filename.lower():
                      source = keyword
                      break
            for doc in split_docs:
                 doc.metadata["source_type"] = source
            print(f"Metadata has been created for {new_filename} as {source}")
            
            print("Creating/Updating Vector Store")
            vector_store(split_docs)
            print("Vector Store create or updated")
            
            loaded_documents.append(split_docs)

    return loaded_documents

if __name__ == "__main__":
    load = doc_loader_pipeline()
    print(load[0])
