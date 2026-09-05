from dotenv import load_dotenv
from rag_backend.rag_pipeline import rag_pipeline
from prompt import filter_query_prompt
from prompt import rules_prompt
from llm_model import model
from rag_backend.retriever import format_instructions

load_dotenv()

retrival_chain_cost = filter_query_prompt | model
full_pipeline_cost = rules_prompt | model

# Config

QUESTIONS = ["Who is defined as a Data Fiduciary and how does that role differ from a Data Processor when it comes to responsibility for complying with the Act?", "What must a Data Fiduciary include in the consent notice given to a Data Principal, and what right does the Data Principal have if she later wants to stop that processing?", "Under what circumstances is a Data Fiduciary exempted from providing personal or domestic use data, and how does the definition of processing help determine what counts as processing digital personal data?", "How does the Act define deemed consent for medical emergencies and disasters, and what specific epidemic-related purpose is listed as an example?"]

REPEAT = 3

# As we are using gpt-4o-mini

PRICE_INPUT_PER_1M = 0.15
PRICE_CACHED_INPUT_PER_1M = 0.075
PRICE_OUTPUT_PER_1M = 0.60

# Buisness projection knobs

QUERIES_PER_DAY = 2000

# Budget 

COST_QUERY_PER_SLO = 0.0015

# Token Measurement

def measure_retrival_tokens(question):
    retriever_cost = retrival_chain_cost.invoke({"user_query" : question, "format_instructions": format_instructions})

    retriever_usage = retriever_cost.usage_metadata or {}

    retriever_input_tokens = retriever_usage.get("input_tokens", 0)
    retriever_output_tokens = retriever_usage.get("output_tokens", 0)
    retriever_details = retriever_usage.get("input_token_details", {})
    retriever_cache_tokens = retriever_details.get("cache_read", 0)

    return {
        "input_tokens" : retriever_input_tokens,
        "output_tokens" : retriever_output_tokens,
        "cached_tokens" : retriever_cache_tokens
    }

def measure_pipeline_cost(question):
    pipeline = rag_pipeline(question)
    context = pipeline.get("retrieved_context")

    pipeline_cost = full_pipeline_cost.invoke({"response_doc" : context, "user_query" : question})
    pipeline_usage = pipeline_cost.usage_metadata or {}

    pipeline_input_tokens = pipeline_usage.get("input_tokens", 0)
    pipeline_output_tokens = pipeline_usage.get("output_tokens", 0)
    pipeline_details = pipeline_usage.get("input_token_details", {})
    pipeline_cached_tokens = pipeline_details.get("cache_read", 0)

    return {
        "input_tokens" : pipeline_input_tokens,
        "output_tokens" : pipeline_output_tokens,
        "cached_tokens" : pipeline_cached_tokens
    }

# Cost Math

def cost_usd(input_tokens, output_tokens, cached_tokens):
    uncached_tokens = max(input_tokens - cached_tokens, 0)

    cost_input = (uncached_tokens/1000000) * PRICE_INPUT_PER_1M
    cost_output = (output_tokens/1000000) * PRICE_OUTPUT_PER_1M
    cost_cached = (cached_tokens/1000000) * PRICE_CACHED_INPUT_PER_1M

    return {
        "cost_input" : cost_input,
        "cost_output" : cost_output,
        "cost_cached" : cost_cached,
        "total_cost" : cost_input + cost_output + cost_cached
    }

# Benchmarks Loop

def retrieve_benchmark():
    retrieve_rows = []
    print("Measuring retriver token usage....")
    for question in QUESTIONS:
        for _ in range(REPEAT):
            retrival_tok = measure_retrival_tokens(question)

            retrieve_cost = cost_usd(retrival_tok["input_tokens"], retrival_tok["output_tokens"], retrival_tok["cached_tokens"])

            retrieve_rows.append({**retrival_tok, **{f"{k}" : v for k,v in retrieve_cost.items()}})

    return retrieve_rows

def pipeline_benchmarks():
    pipeline_rows = []
    print("Measuring pipeline token usage....")
    for question in QUESTIONS:
        for _ in range(REPEAT):
            pipeline_tok = measure_pipeline_cost(question)

            pipeline_cost = cost_usd(pipeline_tok["input_tokens"], pipeline_tok["output_tokens"], pipeline_tok["cached_tokens"])

            pipeline_rows.append({**pipeline_tok, **{f"{k}" : v for k,v in pipeline_cost.items()}})

    return pipeline_rows

# Aggregate + Report

def avg(rows, key):
    return sum(r[key] for r in rows) / len(rows)

def report(rows):
    n = len(rows)

    avg_in     = avg(rows, "input_tokens")
    avg_out    = avg(rows, "output_tokens")
    avg_cached = avg(rows, "cached_tokens")
    avg_cost   = avg(rows, "total_cost")
    min_cost   = min(r["total_cost"] for r in rows)
    max_cost   = max(r["total_cost"] for r in rows)

    # split: how much of the bill is input vs output

    avg_cost_in  = avg(rows, "cost_input") + avg(rows, "cost_cached")
    avg_cost_out = avg(rows, "cost_output")
    out_share = 100 * avg_cost_out / avg_cost if avg_cost else 0

    print("\n" + "=" * 70)
    print(f"COST  (gpt-4o-mini @ ${PRICE_INPUT_PER_1M}/${PRICE_OUTPUT_PER_1M} per 1M in/out)")
    print("=" * 70)
    print(f"samples                : {n}")
    print(f"avg input tokens       : {avg_in:8.0f}   ({avg_cached:.0f} cached)")
    print(f"avg output tokens      : {avg_out:8.0f}")
    print("-" * 70)
    print(f"avg cost / query       : ${avg_cost:.6f}")
    print(f"   min / max           : ${min_cost:.6f} / ${max_cost:.6f}   "
          f"<- tight range = cost is stable, unlike latency")
    print(f"   input vs output     : {100 - out_share:.0f}% input / {out_share:.0f}% output "
          f"(output is 4x the rate -> long answers dominate)")
    print("-" * 70)

    # --- projection: the number a founder actually cares about ---
    daily   = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print(f"projection @ {QUERIES_PER_DAY}/day :")
    print(f"   per day             : ${daily:8.2f}")
    print(f"   per month           : ${monthly:8.2f}")
    print("=" * 70)

    # --- budget verdict (the offline pass/fail) ---
    verdict = "PASS" if avg_cost <= COST_QUERY_PER_SLO else "FAIL"
    print(f"BUDGET: cost/query <= ${COST_QUERY_PER_SLO:.6f}  ->  "
          f"${avg_cost:.6f}   [{verdict}]")
    print("=" * 70)
    print("note: production caching of the (large, fixed) system prompt can push the")
    print("real bill BELOW this estimate -- watch the 'cached' count grow online.")

def main():
    retrieval_rows = retrieve_benchmark()
    report(retrieval_rows)
    pipeline_rows = pipeline_benchmarks()
    report(pipeline_rows)

if __name__ == "__main__":
    main()
