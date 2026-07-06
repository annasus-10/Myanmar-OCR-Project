"""
Convert chuuhtetnaing/myOCR-hf (Hugging Face) into PaddleOCR recognition
format (image_path<TAB>label), matching the CHN pipeline's format so the
existing dictionary/checkpoint code can be reused unchanged.
"""

import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_name", default="chuuhtetnaing/myOCR-hf")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.15,
                     help="Fraction held out as myOCR test set.")
    ap.add_argument("--finetune_frac", type=float, default=0.5,
                     help="Of the non-test remainder, fraction used for fine-tuning.")
    ap.add_argument("--existing_dict", default=None,
                     help="Path to existing Myanmar char dictionary to check OOV coverage against.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    ds = load_dataset(args.hf_name)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    data = ds[split_name]

    rng = random.Random(args.seed)
    n = len(data)
    idx = list(range(n))
    rng.shuffle(idx)

    n_test = int(n * args.val_frac)
    test_idx = set(idx[:n_test])
    remaining = idx[n_test:]
    n_ft = int(len(remaining) * args.finetune_frac)
    ft_idx = set(remaining[:n_ft])
    dev_idx = set(remaining[n_ft:])

    label_files = {
        "test": open(os.path.join(args.out_dir, "myocr_test.txt"), "w", encoding="utf-8"),
        "finetune": open(os.path.join(args.out_dir, "myocr_finetune.txt"), "w", encoding="utf-8"),
        "dev": open(os.path.join(args.out_dir, "myocr_dev.txt"), "w", encoding="utf-8"),
    }

    all_chars = set()
    for i, example in enumerate(data):
        image = example["image"]
        text = example["text"]
        all_chars.update(text)

        fname = f"myocr_{i:06d}.png"
        fpath = os.path.join(img_dir, fname)
        image.save(fpath)

        if i in test_idx:
            split = "test"
        elif i in ft_idx:
            split = "finetune"
        else:
            split = "dev"

        rel_path = os.path.join("images", fname)
        label_files[split].write(f"{rel_path}\t{text}\n")

    for f in label_files.values():
        f.close()

    print(f"Converted {n} examples -> {args.out_dir}")
    print(f"  test:     {len(test_idx)}")
    print(f"  finetune: {len(ft_idx)}")
    print(f"  dev:      {len(dev_idx)}")

    if args.existing_dict:
        with open(args.existing_dict, "r", encoding="utf-8") as f:
            chn_dict = set(line.rstrip("\n") for line in f if line.rstrip("\n"))
        oov = all_chars - chn_dict
        print(f"\nDictionary coverage check against {args.existing_dict}:")
        print(f"  myOCR unique chars: {len(all_chars)}")
        print(f"  OOV (not in CHN dict): {len(oov)}")
        if oov:
            print(f"  OOV chars (first 30): {sorted(oov)[:30]}")
    else:
        print("\n(Pass --existing_dict to also run the OOV coverage check.)")


if __name__ == "__main__":
    main()
