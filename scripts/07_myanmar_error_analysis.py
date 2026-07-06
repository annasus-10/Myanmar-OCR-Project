"""
scripts/07_myanmar_error_analysis.py

Reads a ground_truth,prediction CSV and produces:
  - standard char-level CER (cross-check against jiwer)
  - grapheme-cluster-aware CER (a Myanmar consonant+medial+vowel+tone cluster
    counts as ONE unit, so one wrong diacritic = one error, not several)
  - error taxonomy: substitution/deletion/insertion counts by Myanmar
    character category (consonant, vowel/diacritic, stacking/asat, digit,
    punctuation, space)
  - error rate by ground-truth length bucket
  - a failure-case table using REAL predictions
"""

import argparse
import csv
import json
from collections import Counter, defaultdict

import regex  # UAX#29 extended grapheme clusters via \X


def char_type(ch):
    if ch.isspace():
        return "space"
    cp = ord(ch)
    if 0x1040 <= cp <= 0x1049:
        return "digit"
    if 0x104A <= cp <= 0x104F:
        return "punctuation"
    if cp in (0x1039, 0x103A):
        return "stacking/asat"
    if (
        0x102B <= cp <= 0x103E
        or 0x1056 <= cp <= 0x1059
        or 0x1062 <= cp <= 0x1064
        or 0x1067 <= cp <= 0x106D
        or 0x1071 <= cp <= 0x1074
        or 0x1082 <= cp <= 0x108D
        or cp == 0x108F
    ):
        return "vowel/diacritic"
    if 0x1000 <= cp <= 0x102A or 0x1050 <= cp <= 0x1055:
        return "consonant"
    if 0x1000 <= cp <= 0x109F:
        return "other_myanmar"
    return "latin/other"


def grapheme_clusters(text):
    return regex.findall(r"\X", text)


def levenshtein_ops(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", a[i - 1], None)); i -= 1
        else:
            ops.append(("ins", None, b[j - 1])); j -= 1
    ops.reverse()
    return dp[n][m], ops


def char_cer(gt, pred):
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    dist, _ = levenshtein_ops(list(gt), list(pred))
    return dist / len(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/sample_predictions.csv")
    ap.add_argument("--out_summary", default="results/error_taxonomy_summary.json")
    ap.add_argument("--out_failures", default="results/failure_cases.md")
    ap.add_argument("--max_failure_examples", type=int, default=20)
    args = ap.parse_args()

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["ground_truth"], r["prediction"]))

    n = len(rows)
    total_char_dist = sum(char_cer(gt, pred) * max(len(gt), 1) for gt, pred in rows)
    total_gt_chars = sum(len(gt) for gt, _ in rows)
    total_grapheme_dist_sum = 0
    total_gt_graphemes = 0

    op_counts = Counter()
    category_counts = Counter()
    length_bucket_total = Counter()
    length_bucket_errors = Counter()
    failure_rows = []
    n_exact = 0

    for gt, pred in rows:
        g_gt = grapheme_clusters(gt)
        g_pred = grapheme_clusters(pred)
        dist, ops = levenshtein_ops(g_gt, g_pred)
        total_grapheme_dist_sum += dist
        total_gt_graphemes += max(len(g_gt), 1)

        if gt == pred:
            n_exact += 1

        n_gt_len = len(g_gt)
        if n_gt_len <= 3:
            bucket = "<=3"
        elif n_gt_len <= 8:
            bucket = "4-8"
        elif n_gt_len <= 15:
            bucket = "9-15"
        else:
            bucket = ">15"
        length_bucket_total[bucket] += 1

        sample_cat_counts = Counter()
        has_error = False
        for op, src, dst in ops:
            if op == "match":
                continue
            has_error = True
            op_counts[op] += 1
            token = src if src is not None else dst
            cat = char_type(token[0]) if token else "other_myanmar"
            category_counts[f"{op}:{cat}"] += 1
            sample_cat_counts[cat] += 1

        if has_error:
            length_bucket_errors[bucket] += 1
            if len(failure_rows) < args.max_failure_examples * 3:
                main_type = sample_cat_counts.most_common(1)[0][0] if sample_cat_counts else "unknown"
                failure_rows.append((gt, pred, main_type))

    total_errors = sum(op_counts.values())
    diacritic_errors = sum(v for k, v in category_counts.items() if "vowel/diacritic" in k or "stacking/asat" in k)

    summary = {
        "n_samples": n,
        "exact_match_accuracy": n_exact / n if n else 0.0,
        "char_level_cer": total_char_dist / total_gt_chars if total_gt_chars else 0.0,
        "grapheme_cluster_cer": total_grapheme_dist_sum / total_gt_graphemes if total_gt_graphemes else 0.0,
        "op_counts": dict(op_counts),
        "category_counts": dict(category_counts.most_common(30)),
        "diacritic_or_stacking_related_errors": diacritic_errors,
        "diacritic_share_of_all_errors": diacritic_errors / total_errors if total_errors else 0.0,
        "error_rate_by_grapheme_length_bucket": {
            b: (length_bucket_errors[b] / length_bucket_total[b] if length_bucket_total[b] else None)
            for b in length_bucket_total
        },
    }

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    by_type = defaultdict(list)
    for gt, pred, main_type in failure_rows:
        by_type[main_type].append((gt, pred))
    chosen = []
    types_cycle = list(by_type.keys())
    i = 0
    while len(chosen) < min(args.max_failure_examples, len(failure_rows)) and types_cycle:
        t = types_cycle[i % len(types_cycle)]
        if by_type[t]:
            gt, pred = by_type[t].pop(0)
            chosen.append((gt, pred, t))
        else:
            types_cycle.remove(t)
            continue
        i += 1

    with open(args.out_failures, "w", encoding="utf-8") as f:
        f.write("| Sample | Ground truth | Prediction | Main error type |\n")
        f.write("|---|---|---|---|\n")
        for idx, (gt, pred, t) in enumerate(chosen, 1):
            f.write(f"| {idx} | {gt} | {pred} | {t} |\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote summary to {args.out_summary}")
    print(f"Wrote {len(chosen)} failure-case examples to {args.out_failures}")


if __name__ == "__main__":
    main()
