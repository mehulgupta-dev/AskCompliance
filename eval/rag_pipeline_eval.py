from rag_backend.rag_pipeline import rag_pipeline
from dotenv import load_dotenv
import os
import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric

load_dotenv()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
golden_dataset_path = os.path.join(base_dir, "golden_dataset")

GOLDEN_PATH = os.path.join(golden_dataset_path, "generator_dataset.json")
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

# Step - 1 Load the Dataset

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    golden_dataset = json.load(f)

# Step -2 Run the RAG pipeline on the golden context, for each test cases

test_cases = []

for dataset in golden_dataset:
    result = rag_pipeline(dataset["question"])

    test_cases.append(
        LLMTestCase(
            input = dataset["question"],
            retrieval_context = result["retrieved_context"],
            actual_output = result["generated_answer"]
        )
    )

# Step - 3 The metrics faithfulnees, answer revelency and contextual relevancy, decompose answers in claims and then check

metrics = [FaithfulnessMetric(
    threshold = THRESHOLD,
    model = JUDGE_MODEL,
    include_reason = True
),
AnswerRelevancyMetric(
    threshold = THRESHOLD,
    model = JUDGE_MODEL,
    include_reason = True
),
ContextualRelevancyMetric(
    threshold = THRESHOLD,
    model = JUDGE_MODEL,
    include_reason = True
)]

# Step - 4 Evaluate every metric on every testcase batched and in-parallel, with a printed report.

evaluate(
    test_cases = test_cases,
    metrics = metrics
)