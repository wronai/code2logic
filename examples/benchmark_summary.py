#!/usr/bin/env python3
"""Print benchmark summary from JSON result files."""
import json
import os
import sys


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/output"

    files = {
        "Format": os.path.join(out, "benchmark_format.json"),
        "Token": os.path.join(out, "benchmark_token.json"),
        "Project": os.path.join(out, "benchmark_project.json"),
        "Function": os.path.join(out, "benchmark_function.json"),
    }

    print()
    print(
        f"{'Benchmark':<12} {'Files':>6} {'Avg Score':>10} "
        f"{'Syntax OK':>10} {'Runs OK':>10} {'Best Format':>14} {'Time':>8}"
    )
    print("-" * 75)

    for name, path in files.items():
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        if d.get("benchmark_type") == "function":
            fr = d.get("function_results") or []
            total = d.get("total_functions", len(fr))
            sims = [x.get("similarity", 0.0) for x in fr if x.get("similarity", 0.0) > 0]
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            syn = (sum(1 for x in fr if x.get("syntax_ok")) / len(fr) * 100) if fr else 0.0
            print(
                f"{name:<12} {total:>6} "
                f"{avg_sim:>9.1f}% "
                f"{syn:>9.0f}% "
                f"{'-':>9} "
                f"{'-':>14} "
                f"{d.get('total_time', 0):>7.1f}s"
            )
        else:
            total = d.get("total_files", "-")
            print(
                f"{name:<12} {total:>6} "
                f"{d.get('avg_score', 0):>9.1f}% "
                f"{d.get('syntax_ok_rate', 0):>9.0f}% "
                f"{d.get('runs_ok_rate', 0):>9.0f}% "
                f"{d.get('best_format', '-'):>14} "
                f"{d.get('total_time', 0):>7.1f}s"
            )

    print()

    fmt_path = os.path.join(out, "benchmark_format.json")
    if os.path.exists(fmt_path):
        d = json.load(open(fmt_path))
        scores = d.get("format_scores", {})
        if scores:
            print("Format scores:")
            for fmt, sc in sorted(scores.items(), key=lambda x: -x[1]):
                bar = "\u2588" * int(sc / 2)
                print(f"  {fmt:<10} {sc:>6.1f}%  {bar}")
            print()


if __name__ == "__main__":
    main()
