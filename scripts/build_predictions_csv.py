"""
Merge PaddleOCR infer_rec.py raw output with ground-truth labels into a
ground_truth,prediction CSV (the format scripts/04_evaluate_predictions.py
already expects).

Handles the two common PaddleOCR save_res_path line formats:
  1. image_path<TAB>{"transcription": "...", "score": 0.98}
  2. image_path<TAB>text<TAB>score
"""

import argparse
import csv
import json
import os


def load_gt(path):
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            img, txt = line.split("\t", 1)
            gt[img.strip()] = txt.strip()
            gt[os.path.basename(img.strip())] = txt.strip()
    return gt


def parse_pred_line(line):
    line = line.rstrip("\n")
    if not line or "\t" not in line:
        return None
    parts = line.split("\t")
    img_path = parts[0].strip()
    rest = "\t".join(parts[1:])

    try:
        obj = json.loads(rest)
        text = obj.get("transcription") or obj.get("label") or obj.get("text")
        if text is not None:
            return img_path, text
    except (json.JSONDecodeError, AttributeError):
        pass

    sub_parts = rest.split("\t")
    text = sub_parts[0].strip()
    return img_path, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gt_file", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    gt = load_gt(args.gt_file)

    rows = []
    n_matched, n_unmatched = 0, 0
    unmatched_examples = []

    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_pred_line(line)
            if parsed is None:
                continue
            img_path, pred_text = parsed

            gt_text = gt.get(img_path) or gt.get(os.path.basename(img_path))
            if gt_text is None:
                n_unmatched += 1
                if len(unmatched_examples) < 5:
                    unmatched_examples.append(img_path)
                continue

            rows.append((gt_text, pred_text))
            n_matched += 1

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth", "prediction"])
        writer.writerows(rows)

    print(f"Matched: {n_matched}")
    print(f"Unmatched (pred had no gt match): {n_unmatched}")
    if unmatched_examples:
        print(f"Unmatched examples: {unmatched_examples}")
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    if n_matched == 0:
        print("WARNING: zero matches. Check that image_path formats in "
              "pred_file and gt_file line up (try printing a few raw lines "
              "from both files to compare).")


if __name__ == "__main__":
    main()
