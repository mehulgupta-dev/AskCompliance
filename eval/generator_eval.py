import json
import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from rag_backend.generator import generator

load_dotenv()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
golden_dataset_path = os.path.join(base_dir, "golden_dataset")

GOLDEN_PATH = os.path.join(golden_dataset_path, "generator_dataset.json")
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

# Step - 1 Load the dataset

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    golden_dataset = json.load(f)

# Step - 2 Run the Generator on the golden context, for each test cases

test_cases = []

for dataset in golden_dataset:
    answer = generator(query = dataset["question"], context = dataset["relevant_chunk"])

    test_cases.append(
        LLMTestCase(
            input = dataset["question"],
            retrieval_context = dataset["relevant_chunk"],
            actual_output = answer
        )
    )

# Step - 3 The metrics faithfulnees and answer revelency, decompose answers in claims and then check

metrics = [FaithfulnessMetric(
    threshold = THRESHOLD,
    model = JUDGE_MODEL,
    include_reason = True
),
AnswerRelevancyMetric(
    threshold = THRESHOLD,
    model = JUDGE_MODEL,
    include_reason = True
)]

# Step - 4 Evaluate every metric on every testcase batched and in-parallel, with a printed report.

evaluate(
    test_cases = test_cases,
    metrics = metrics
)