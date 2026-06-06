import json
import time

from parser import (
    extract_ground_truth,
    build_inference_messages,
    parse_prediction
)


class Evaluator:

    def __init__(self, model_runner):
        self.model_runner = model_runner

    def load_dataset(self, jsonl_path):

        rows = []

        with open(
            jsonl_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:
                rows.append(json.loads(line))

        return rows

    def evaluate_sample(
            self,
            sample,
            idx):

        gt = extract_ground_truth(sample)

        messages = build_inference_messages(sample)

        tools = sample["tools"]

        start = time.perf_counter()

        output = self.model_runner.generate(
            tools,
            messages
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        pred_tool, pred_args = parse_prediction(
            output["generated_text"]
        )

        tool_match = (
            pred_tool == gt["tool_name"]
        )

        args_match = (
            pred_args == gt["arguments"]
        )

        exact_match = (
            tool_match and args_match
        )

        print(f"Sample {idx}: "
              f"Tool Match: {tool_match}, "
              f"Args Match: {args_match}, "
              f"Exact Match: {exact_match}, "
              f"Latency: {latency_ms:.1f} ms")

        return {
            "row": idx,

            "tool_gt":
                gt["tool_name"],

            "tool_pred":
                pred_tool,

            "args_gt":
                str(gt["arguments"]),

            "args_pred":
                str(pred_args),

            "tool_match":
                tool_match,

            "args_match":
                args_match,

            "exact_match":
                exact_match,

            "latency":
                latency_ms,

            "prompt_tokens":
                output["prompt_tokens"],

            "generated_tokens":
                output["generated_tokens"],

            "raw_output":
                output["generated_text"]
        }