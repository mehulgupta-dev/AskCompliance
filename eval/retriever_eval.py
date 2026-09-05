from rag_backend.reranker import compress_retriever
import os
import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

load_dotenv()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
golden_dataset_path = os.path.join(base_dir, "golden_dataset")

GOLDEN_PATH = os.path.join(golden_dataset_path, "retriever_dataset.json")
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

# Step-1 -- Load the dataset

with open(GOLDEN_PATH) as f:
    golden_dataset = json.load(f)

# Step-2 -- Run the retriever on each question to fill retrival context, then build one text case per golden.

test_case = []

for dataset in golden_dataset:
    retrieved = compress_retriever.invoke(dataset["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_case.append(
        LLMTestCase(
            input = dataset["query"],
            expected_output = dataset["ideal_answer"],
            retrieval_context = retrieval_context,
            actual_output = "(generator not evaluated in this run)" 
        )
    )

# Step - 3 The Metrics, recall(did we miss?) and precision(did we rank well ?)

metrics = [
    ContextualRecallMetric(threshold = THRESHOLD, model = JUDGE_MODEL, include_reason = True),
    ContextualPrecisionMetric(threshold = THRESHOLD, model = JUDGE_MODEL, include_reason = True)
]

# Step - 4 Evaluate every metric on every testcase batched and in-parallel, with a printed report.

evaluate(
    test_cases = test_case,
    metrics = metrics,
    hyperparameters = {
        "retriever" : "reranked",
        "embedding_model" : "text-embedding-3-large",
        "chunk_size" : 1200,
        "chunk_overlap" : 250,
        "top_k" : 5,
        "judge_model" : JUDGE_MODEL,
        "golden_set" : GOLDEN_PATH
    }
)