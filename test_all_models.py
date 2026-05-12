import argparse
import ast
import asyncio
import json
import multiprocessing
import re
import statistics
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx


DEFAULT_BASE_URL = "http://localhost:3005"
DEFAULT_API_KEY = "abc"
DEFAULT_TIMEOUT = 60.0
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_TOKENS = 700
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_FILE = SCRIPT_DIR / "all_free_models.txt"

FALLBACK_MODELS = [
    "kilo-auto/free",
    "x-ai/grok-code-fast-1:optimized:free",
    "stepfun/step-3.5-flash:free",
    "poolside/laguna-m.1:free",
    "poolside/laguna-xs.2:free",
    "baidu/cobuddy:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

CODE_TESTS = [
    ([1, 2, 3, 5, 7, 8, 9], "1-3,5,7-9"),
    ([4, 4, 2, 3, 10], "2-4,10"),
    ([-2, -1, 0, 2], "-2-0,2"),
    ([9], "9"),
    ([5, 3, 4, 10, 11, 12, 12], "3-5,10-12"),
    ([0, 2, 4], "0,2,4"),
    ([-5, -4, -3, 1, 2, 4, 6, 7, 8], "-5--3,1-2,4,6-8"),
]


@dataclass
class Evaluation:
    score: int
    max_score: int
    passed: bool
    note: str
    parsed: Any | None = None


@dataclass
class BenchmarkCase:
    name: str
    max_score: int
    prompt: str
    evaluator: Callable[[str], Evaluation]


@dataclass
class SingleRunResult:
    case_name: str
    score: int
    max_score: int
    latency: float
    passed: bool
    note: str
    response: str
    error: str | None = None


@dataclass
class CaseAggregate:
    case_name: str
    score: float
    max_score: int
    avg_latency: float
    passed: bool
    notes: list[str]
    sample_response: str
    errors: list[str]


def build_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="logic_trap",
            max_score=20,
            prompt=textwrap.dedent(
                """
                Return ONLY minified JSON with exactly these keys: sisters, reason.

                Question:
                Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?

                Rules:
                - sisters must be an integer.
                - reason must be under 18 words.
                - No markdown and no extra keys.
                """
            ).strip(),
            evaluator=evaluate_logic_case,
        ),
        BenchmarkCase(
            name="instruction_chain",
            max_score=20,
            prompt=textwrap.dedent(
                """
                Return ONLY minified JSON with exactly one key: result.

                Start with this list:
                [19, 4, 11, 4, 8, 19, 7, 2]

                Apply the rules in this exact order:
                1. Remove duplicates while keeping the first occurrence.
                2. Keep only odd numbers.
                3. Square the remaining numbers.
                4. Sort the final numbers descending.

                The value of result must be a JSON array of integers.
                """
            ).strip(),
            evaluator=evaluate_instruction_case,
        ),
        BenchmarkCase(
            name="data_extraction",
            max_score=20,
            prompt=textwrap.dedent(
                """
                Return ONLY minified JSON with exactly these keys:
                owner, priority_sum, earliest_due, active_ticket_ids

                Dataset:
                - Ticket AA-17 | owner=Maya | priority=4 | due=2026-05-19 | status=active
                - Ticket B-204 | owner=Liam | priority=1 | due=2026-05-13 | status=active
                - Ticket OPS-9 | owner=Maya | priority=5 | due=2026-05-12 | status=active
                - Ticket Z-99 | owner=Maya | priority=10 | due=2026-05-01 | status=canceled
                - Ticket QA-2 | owner=Noah | priority=3 | due=2026-05-14 | status=active

                Rules:
                - Use only Maya's active tickets.
                - priority_sum is the sum of their priorities.
                - earliest_due must be in YYYY-MM-DD format.
                - active_ticket_ids must preserve the original dataset order.
                """
            ).strip(),
            evaluator=evaluate_extraction_case,
        ),
        BenchmarkCase(
            name="python_ranges",
            max_score=40,
            prompt=textwrap.dedent(
                """
                Write only Python code in a single fenced code block.

                Implement:
                def summarize_ranges(values: list[int]) -> str:

                Behavior:
                - Input may be unsorted and may contain duplicates.
                - Sort unique values ascending.
                - Collapse consecutive runs as "start-end".
                - Keep single numbers as just the number.
                - Join segments with commas and no spaces.

                Examples:
                [1, 2, 3, 5, 7, 8, 9] -> "1-3,5,7-9"
                [4, 4, 2, 3, 10] -> "2-4,10"
                [-2, -1, 0, 2] -> "-2-0,2"

                Constraints:
                - Do not print anything.
                - Do not read input.
                - No explanation outside the code block.
                """
            ).strip(),
            evaluator=evaluate_code_case,
        ),
    ]


def extract_json_candidate(text: str | None) -> Any | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start_index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def extract_code_block(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[0].strip()
    return text.strip()


def evaluate_logic_case(content: str) -> Evaluation:
    parsed = extract_json_candidate(content)
    json_bonus = 4 if isinstance(parsed, dict) else 0
    sisters = None
    if isinstance(parsed, dict):
        sisters = parsed.get("sisters")
    elif re.search(r"\b1\b", content):
        sisters = 1

    score = json_bonus
    if sisters == 1:
        score += 16
        return Evaluation(score=score, max_score=20, passed=True, note="correct", parsed=parsed)
    return Evaluation(score=score, max_score=20, passed=False, note=f"wrong sisters value: {sisters!r}", parsed=parsed)


def evaluate_instruction_case(content: str) -> Evaluation:
    expected = [361, 121, 49]
    parsed = extract_json_candidate(content)
    score = 0
    result = None

    if isinstance(parsed, dict):
        score += 4
        result = parsed.get("result")
    elif isinstance(parsed, list):
        result = parsed

    if result == expected:
        score += 16
        return Evaluation(score=score, max_score=20, passed=True, note="correct", parsed=parsed)

    partial = 0
    if isinstance(result, list):
        partial = sum(1 for got, want in zip(result, expected) if got == want) * 4
    score += min(partial, 16)
    return Evaluation(score=score, max_score=20, passed=False, note=f"expected {expected}, got {result!r}", parsed=parsed)


def evaluate_extraction_case(content: str) -> Evaluation:
    parsed = extract_json_candidate(content)
    if not isinstance(parsed, dict):
        return Evaluation(score=0, max_score=20, passed=False, note="not valid JSON object", parsed=parsed)

    expected = {
        "owner": "Maya",
        "priority_sum": 9,
        "earliest_due": "2026-05-12",
        "active_ticket_ids": ["AA-17", "OPS-9"],
    }

    score = 4
    for key, expected_value in expected.items():
        if parsed.get(key) == expected_value:
            score += 4
    passed = all(parsed.get(key) == value for key, value in expected.items())
    note = "correct" if passed else f"expected {expected}, got {parsed}"
    return Evaluation(score=score, max_score=20, passed=passed, note=note, parsed=parsed)


def evaluate_code_case(content: str) -> Evaluation:
    code = extract_code_block(content)
    result = grade_code_submission(code)
    return Evaluation(
        score=result["score"],
        max_score=40,
        passed=result["passed"],
        note=result["note"],
        parsed={"passed_tests": result.get("passed_tests", 0), "total_tests": len(CODE_TESTS)},
    )


def grade_code_submission(source: str) -> dict[str, Any]:
    start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    context = multiprocessing.get_context(start_method)
    queue: multiprocessing.Queue[Any] = context.Queue()
    process = context.Process(target=_grade_code_submission_worker, args=(source, queue))
    process.start()
    process.join(timeout=3.0)

    if process.is_alive():
        process.terminate()
        process.join()
        return {"score": 0, "passed": False, "note": "code execution timed out", "passed_tests": 0}

    if queue.empty():
        return {"score": 0, "passed": False, "note": "no grading result returned", "passed_tests": 0}

    return queue.get()


def _grade_code_submission_worker(source: str, queue: multiprocessing.Queue[Any]) -> None:
    try:
        result = _grade_code_submission_inner(source)
    except Exception as exc:
        result = {"score": 0, "passed": False, "note": f"grader crashed: {exc}", "passed_tests": 0}
    queue.put(result)


def _grade_code_submission_inner(source: str) -> dict[str, Any]:
    if not source.strip():
        return {"score": 0, "passed": False, "note": "empty code", "passed_tests": 0}

    if "__" in source:
        return {"score": 0, "passed": False, "note": "dunder usage is not allowed", "passed_tests": 0}

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"score": 0, "passed": False, "note": f"syntax error: {exc.msg}", "passed_tests": 0}

    banned_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.ClassDef,
        ast.Lambda,
        ast.Global,
        ast.Nonlocal,
        ast.Delete,
        ast.AsyncFunctionDef,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
    )
    banned_names = {
        "open",
        "exec",
        "eval",
        "compile",
        "input",
        "globals",
        "locals",
        "vars",
        "help",
        "dir",
        "os",
        "sys",
        "subprocess",
        "pathlib",
        "shutil",
        "socket",
        "requests",
        "httpx",
    }

    for node in ast.walk(tree):
        if isinstance(node, banned_nodes):
            return {"score": 0, "passed": False, "note": f"disallowed syntax: {type(node).__name__}", "passed_tests": 0}
        if isinstance(node, ast.Name) and node.id in banned_names:
            return {"score": 0, "passed": False, "note": f"disallowed name: {node.id}", "passed_tests": 0}

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "enumerate": enumerate,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace = {"__builtins__": safe_builtins}

    exec(compile(tree, filename="<benchmark>", mode="exec"), namespace, namespace)
    function = namespace.get("summarize_ranges")
    if not callable(function):
        return {"score": 5, "passed": False, "note": "function summarize_ranges not found", "passed_tests": 0}

    passed_tests = 0
    failures = []
    for values, expected in CODE_TESTS:
        actual = function(list(values))
        if actual == expected:
            passed_tests += 1
        else:
            failures.append(f"{values} -> {actual!r}, expected {expected!r}")

    score = 5 + round((passed_tests / len(CODE_TESTS)) * 35)
    passed = passed_tests == len(CODE_TESTS)
    note = "correct" if passed else f"passed {passed_tests}/{len(CODE_TESTS)} tests; first failure: {failures[0]}"
    return {"score": score, "passed": passed, "note": note, "passed_tests": passed_tests}


async def fetch_models_from_proxy(client: httpx.AsyncClient, base_url: str) -> list[str]:
    models_url = f"{base_url.rstrip('/')}/v1/models"
    response = await client.get(models_url, timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", [])
    discovered = []
    for item in models:
        if isinstance(item, dict) and item.get("id"):
            discovered.append(item["id"])
    return discovered


def load_models_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def build_completion_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def clip_text(text: str, limit: int = 180) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


async def run_single_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    completion_url: str,
    api_key: str,
    model: str,
    case: BenchmarkCase,
    max_tokens: int,
) -> SingleRunResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt}],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    try:
        async with semaphore:
            response = await client.post(
                completion_url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        latency = time.perf_counter() - start
    except Exception as exc:
        return SingleRunResult(
            case_name=case.name,
            score=0,
            max_score=case.max_score,
            latency=0.0,
            passed=False,
            note="request failed",
            response="",
            error=str(exc),
        )

    if response.status_code != 200:
        return SingleRunResult(
            case_name=case.name,
            score=0,
            max_score=case.max_score,
            latency=time.perf_counter() - start,
            passed=False,
            note="non-200 response",
            response=response.text[:400],
            error=f"HTTP {response.status_code}",
        )

    try:
        content = response.json()["choices"][0]["message"]["content"]
        if content is None:
            content = ""
    except Exception as exc:
        return SingleRunResult(
            case_name=case.name,
            score=0,
            max_score=case.max_score,
            latency=latency,
            passed=False,
            note="bad response shape",
            response=response.text[:400],
            error=str(exc),
        )

    evaluation = case.evaluator(content)
    return SingleRunResult(
        case_name=case.name,
        score=evaluation.score,
        max_score=evaluation.max_score,
        latency=latency,
        passed=evaluation.passed,
        note=evaluation.note,
        response=content,
    )


def aggregate_case_runs(runs: list[SingleRunResult], max_score: int) -> CaseAggregate:
    if not runs:
        raise ValueError("aggregate_case_runs requires at least one run")

    score = statistics.mean(run.score for run in runs)
    avg_latency = statistics.mean(run.latency for run in runs)
    passed = all(run.passed for run in runs)
    notes = [run.note for run in runs]
    errors = [run.error for run in runs if run.error]
    sample_response = max(runs, key=lambda run: run.score).response

    return CaseAggregate(
        case_name=runs[0].case_name,
        score=score,
        max_score=max_score,
        avg_latency=avg_latency,
        passed=passed,
        notes=notes,
        sample_response=sample_response,
        errors=errors,
    )


async def benchmark_model(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    completion_url: str,
    api_key: str,
    model: str,
    cases: list[BenchmarkCase],
    repeats: int,
    max_tokens: int,
) -> dict[str, Any]:
    case_results = []

    for case in cases:
        runs = []
        for _ in range(repeats):
            runs.append(
                await run_single_case(
                    client=client,
                    semaphore=semaphore,
                    completion_url=completion_url,
                    api_key=api_key,
                    model=model,
                    case=case,
                    max_tokens=max_tokens,
                )
            )
        case_results.append(aggregate_case_runs(runs, case.max_score))

    total_score = sum(case.score for case in case_results)
    total_max_score = sum(case.max_score for case in case_results)
    avg_latency = statistics.mean(case.avg_latency for case in case_results)
    pass_count = sum(1 for case in case_results if case.passed)
    error_count = sum(len(case.errors) for case in case_results)

    return {
        "model": model,
        "total_score": total_score,
        "max_score": total_max_score,
        "avg_latency": avg_latency,
        "pass_count": pass_count,
        "error_count": error_count,
        "cases": case_results,
    }


def print_summary(results: list[dict[str, Any]], total_max_score: int) -> None:
    print()
    print("Kilo model benchmark")
    print()
    print(f"{'Rank':<5} {'Model':<48} {'Score':>10} {'Pass':>7} {'Avg Latency':>13}")
    print("-" * 90)
    for index, result in enumerate(results, start=1):
        print(
            f"{index:<5} "
            f"{result['model']:<48} "
            f"{result['total_score']:>6.1f}/{total_max_score:<3} "
            f"{result['pass_count']:>4}/{len(result['cases']):<2} "
            f"{result['avg_latency']:>11.2f}s"
        )
    print()


def print_breakdown(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(f"Model: {result['model']}")
        for case in result["cases"]:
            print(
                f"  - {case.case_name:<18} "
                f"score={case.score:.1f}/{case.max_score} "
                f"latency={case.avg_latency:.2f}s "
                f"status={'pass' if case.passed else 'fail'}"
            )
            print(f"    note: {case.notes[0]}")
        print()


def print_response_samples(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(f"Responses from {result['model']}:")
        for case in result["cases"]:
            print(f"  - {case.case_name}: {clip_text(case.sample_response)}")
        print()


def save_results(path: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    serializable = {
        "base_url": args.base_url,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "generated_at_epoch": time.time(),
        "results": [
            {
                "model": result["model"],
                "total_score": result["total_score"],
                "max_score": result["max_score"],
                "avg_latency": result["avg_latency"],
                "pass_count": result["pass_count"],
                "error_count": result["error_count"],
                "cases": [
                    {
                        "case_name": case.case_name,
                        "score": case.score,
                        "max_score": case.max_score,
                        "avg_latency": case.avg_latency,
                        "passed": case.passed,
                        "notes": case.notes,
                        "sample_response": case.sample_response,
                        "errors": case.errors,
                    }
                    for case in result["cases"]
                ],
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(serializable, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harder benchmark for Kilo proxy chat models.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Proxy base URL, default: %(default)s")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer token for /v1/chat/completions")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max in-flight requests")
    parser.add_argument("--repeats", type=int, default=1, help="Run each case multiple times per model")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="max_tokens for model responses")
    parser.add_argument("--models", help="Comma-separated model IDs to test")
    parser.add_argument("--models-file", default=str(DEFAULT_MODELS_FILE), help="Fallback path to newline-separated model IDs")
    parser.add_argument("--skip-discovery", action="store_true", help="Do not query /v1/models before using file or fallback models")
    parser.add_argument("--save-json", help="Write full benchmark results to a JSON file")
    parser.add_argument("--show-breakdown", action="store_true", help="Print per-case scoring details")
    parser.add_argument("--show-responses", action="store_true", help="Print a clipped sample response per case")
    return parser.parse_args()


async def resolve_models(args: argparse.Namespace, client: httpx.AsyncClient) -> list[str]:
    if args.models:
        return dedupe_keep_order([model.strip() for model in args.models.split(",") if model.strip()])

    discovered: list[str] = []
    if not args.skip_discovery:
        try:
            discovered = await fetch_models_from_proxy(client, args.base_url)
            if discovered:
                print(f"Discovered {len(discovered)} models from {args.base_url.rstrip('/')}/v1/models")
        except Exception as exc:
            print(f"Model discovery failed: {exc}")

    if discovered:
        return dedupe_keep_order(discovered)

    file_models = load_models_from_file(Path(args.models_file))
    if file_models:
        print(f"Loaded {len(file_models)} models from {args.models_file}")
        return dedupe_keep_order(file_models)

    print("Falling back to built-in model list")
    return FALLBACK_MODELS


async def async_main() -> None:
    args = parse_args()
    cases = build_cases()
    total_max_score = sum(case.max_score for case in cases)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        models = await resolve_models(args, client)
        if not models:
            raise SystemExit("No models found to benchmark.")

        print(f"Benchmarking {len(models)} models across {len(cases)} cases x {args.repeats} repeat(s)")
        completion_url = build_completion_url(args.base_url)
        semaphore = asyncio.Semaphore(max(1, args.concurrency))

        tasks = [
            benchmark_model(
                client=client,
                semaphore=semaphore,
                completion_url=completion_url,
                api_key=args.api_key,
                model=model,
                cases=cases,
                repeats=max(1, args.repeats),
                max_tokens=max(64, args.max_tokens),
            )
            for model in models
        ]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda item: (-item["total_score"], item["avg_latency"], item["error_count"], item["model"]))
    print_summary(results, total_max_score)

    if args.show_breakdown:
        print_breakdown(results)

    if args.show_responses:
        print_response_samples(results)

    if args.save_json:
        output_path = Path(args.save_json)
        save_results(output_path, results, args)
        print(f"Saved JSON results to {output_path}")


if __name__ == "__main__":
    asyncio.run(async_main())
