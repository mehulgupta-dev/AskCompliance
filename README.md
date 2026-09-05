# AskCompliance

AskCompliance is a Retrieval-Augmented Generation (RAG) application that lets you ask natural-language questions about compliance documents (e.g. policies, regulations, internal guidelines) and get answers grounded in the actual source text.

Documents are chunked, embedded, and stored in a local vector database. When you ask a question, the most relevant chunks are retrieved and passed to an LLM (via `langchain-openai`) to generate a grounded answer, served through a simple Streamlit UI.

## Features

- 📄 Ingests compliance documents (PDF, via `pypdf`) and splits them into chunks
- 🔎 Embeds chunks with `sentence-transformers` and stores/retrieves them with `chromadb`
- 🤖 Uses an OpenAI chat model (`gpt-4.1-mini` by default) through `langchain` / `langchain-openai` to answer questions based on retrieved context
- 💬 Streamlit chat interface (`frontend_streamlit.py`)
- 📊 Evaluation harness using `deepeval` against a golden dataset (`eval/`, `golden_dataset/`)

## Project Structure

```
AskCompliance/
├── documents/          # Source compliance documents to ingest
├── eval/               # Evaluation scripts
├── golden_dataset/     # Reference Q&A pairs used for evaluation
├── graph/              # Orchestration / pipeline graph logic
├── rag_backend/        # Core RAG backend (retrieval + generation)
├── embedding.py         # Embedding generation / vector store population
├── extract_chunks.py    # Document loading and chunking
├── llm_model.py         # LLM configuration (OpenAI chat model)
├── prompt.py            # Prompt templates
├── frontend_streamlit.py# Streamlit chat UI
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An OpenAI API key

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/mehulgupta-dev/AskCompliance.git
cd AskCompliance

# using uv (recommended, matches uv.lock)
uv sync

# or using pip
pip install -e .
```

## Configuration

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

1. **Add documents** — place the compliance documents (PDFs) you want to query into the `documents/` folder.

2. **Build the index** — chunk the documents and generate embeddings:

   ```bash
   python extract_chunks.py
   python embedding.py
   ```

3. **Launch the app**:

   ```bash
   streamlit run frontend_streamlit.py
   ```

   Then open the local URL Streamlit prints (usually `http://localhost:8501`) and start asking questions about your documents.

## Evaluation

The `eval/` and `golden_dataset/` folders support automated quality checks using `deepeval`, comparing model answers against a curated set of reference question/answer pairs. Run the evaluation scripts in `eval/` to score retrieval and answer quality.

## Tech Stack

| Component        | Library |
|-------------------|---------|
| Orchestration     | LangChain / LangChain Community |
| LLM               | OpenAI (via `langchain-openai`) |
| Embeddings        | sentence-transformers |
| Vector store      | ChromaDB |
| PDF parsing       | pypdf |
| UI                | Streamlit |
| Evaluation        | DeepEval |

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License