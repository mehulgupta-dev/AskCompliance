from dotenv import load_dotenv
import os
import json
from rag_backend.rag_pipeline import rag_pipeline
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.test_case import SingleTurnParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

load_dotenv()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
golden_dataset_path = os.path.join(base_dir, "golden_dataset")

GOLDEN_PATH = os.path.join(golden_dataset_path, "geval_application_level_dataset.json")
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

# Step-1 -- Load the dataset

with open(GOLDEN_PATH, "r", encoding = "utf-8") as f:
    golden_dataset = json.load(f)

# Step -2 -- Run the generator on each question to actual output, then build one text case per golden.

test_cases = []

for dataset in golden_dataset:
    result = rag_pipeline(dataset["question"])

    test_cases.append(
        LLMTestCase(
            input = dataset["question"],
            expected_output = dataset["ideal_answer"],
            actual_output = result["generated_answer"]
        )
    )

# Step - 3 Application level metrics.

# Step - 3a -- Evaluate correctness. reference based, judge Truth. (Not coverage or length)

correctness = GEval(
    name = "correctness",
    evaluation_steps = [
        "Extract every factual claim from 'expected output', including exact figures — monetary amounts, time periods, deadlines, section/rule numbers, named entities/roles.",

        "Check whether 'actual output' contains each of these facts with the same meaning. Paraphrasing is fine; a different number, deadline, or amount is not — treat any numeric or figure mismatch as a serious error, not a minor one.",

        "Check whether 'actual output' introduces any claim, figure, or provision that is not present in 'expected output' — treat this as a possible hallucination and penalize it, even if it sounds plausible or legally reasonable.",

        "If the question has multiple parts (e.g. asks about two related provisions), verify both parts are addressed. If one part is fully correct and the other is missing or wrong, this is a partial answer, not a correct one.",

        "Do not penalize for differences in phrasing, structure, ordering, added transitional language, or conciseness/verbosity, as long as all required facts are present and accurate.",

        "Do not penalize for reasonable elaboration that is consistent with 'expected output', only for content that is missing, contradictory, or fabricated."
    ],
    rubric = [
        Rubric(score_range = (0, 2), expected_outcome = "Contradicts or fabricates facts not in expected_output."),
        Rubric(score_range = (3, 5), expected_outcome = "Partially correct; one part of a multi-part question wrong or missing."),
        Rubric(score_range = (6,8), expected_outcome = "Core facts correct; minor omission, no errors."),
        Rubric(score_range = (9, 10), expected_outcome = "Fully matches expected_output, all figures and parts covered.")
    ],

    evaluation_params = [SingleTurnParams.INPUT, SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    threshold = THRESHOLD,
    model = JUDGE_MODEL,
    strict_mode = False
    )

completeness = GEval(
    name = "completeness",
    evaluation_steps = [
        "Check whether 'actual output' addresses all parts of the question, even if some parts are answered incorrectly. If any part is missing, this is incomplete.",

        "Check whether 'actual output' includes all required facts and figures from 'expected output'. If any fact or figure is omitted, this is incomplete.",

        "Do not penalize for minor phrasing differences or added transitional language, as long as all required content is present."
    ],

    evaluation_params = [SingleTurnParams.INPUT, SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    rubric = [
        Rubric(score_range = (0, 2), expected_outcome = "Major omissions; multiple parts of the question or key facts missing."),
        Rubric(score_range = (3, 5), expected_outcome = "Some omissions; one part of a multi-part question or some key facts missing."),
        Rubric(score_range = (6, 8), expected_outcome = "Minor omissions; most parts and key facts present."),
        Rubric(score_range = (9, 10), expected_outcome = "All parts of the question and all key facts present."),
    ],
    threshold = THRESHOLD,
    model = JUDGE_MODEL,
    strict_mode = False
)

# Step - 4 Evaluate every metric on every testcase batched and in-parallel, with a printed report.

evaluate(
    test_cases = test_cases,
    metrics = [correctness, completeness]
)