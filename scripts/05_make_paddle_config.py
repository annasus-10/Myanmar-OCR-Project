"""
Create a Myanmar PaddleOCR recognition config from an existing PaddleOCR config.

Run from the project repo:

    python scripts/05_make_paddle_config.py \
      --paddleocr_dir ~/PaddleOCR \
      --output configs/myanmar_rec_config.yml \
      --save_model_dir outputs/puffer_2080ti_run1 \
      --epoch_num 5 \
      --batch_size 32
"""

import argparse
from pathlib import Path
import yaml


CANDIDATE_CONFIGS = [
    "configs/rec/rec_mv3_none_bilstm_ctc.yml",
    "configs/rec/rec_r34_vd_none_bilstm_ctc.yml",
    "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml",
    "configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml",
]


def find_template(paddleocr_dir: Path) -> Path:
    for rel in CANDIDATE_CONFIGS:
        path = paddleocr_dir / rel
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a recognition config. Checked:\n"
        + "\n".join(str(paddleocr_dir / p) for p in CANDIDATE_CONFIGS)
    )


def patch_dataset_section(section, data_dir, label_file):
    if section is None:
        return

    dataset = section.get("dataset", {})
    dataset["name"] = "SimpleDataSet"
    dataset["data_dir"] = str(data_dir)
    dataset["label_file_list"] = [str(label_file)]
    section["dataset"] = dataset


def patch_batch_size(section, batch_size):
    if section is None:
        return

    loader = section.get("loader", {})
    if "batch_size_per_card" in loader:
        loader["batch_size_per_card"] = batch_size
    elif "batch_size" in loader:
        loader["batch_size"] = batch_size
    else:
        loader["batch_size_per_card"] = batch_size
    section["loader"] = loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paddleocr_dir", required=True)
    parser.add_argument("--output", default="configs/myanmar_rec_config.yml")
    parser.add_argument("--save_model_dir", default="outputs/puffer_2080ti_run1")
    parser.add_argument("--epoch_num", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    paddleocr_dir = Path(args.paddleocr_dir).expanduser().resolve()

    template_path = find_template(paddleocr_dir)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using template config: {template_path}")

    with template_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = project_root / "data/chn_subset"
    train_label = data_dir / "rec_gt_train.txt"
    test_label = data_dir / "rec_gt_test.txt"
    dict_path = data_dir / "myanmar_dict.txt"
    save_model_dir = project_root / args.save_model_dir

    save_model_dir.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("Global", {})
    cfg["Global"]["use_gpu"] = True
    cfg["Global"]["epoch_num"] = args.epoch_num
    cfg["Global"]["save_model_dir"] = str(save_model_dir)
    cfg["Global"]["character_dict_path"] = str(dict_path)
    cfg["Global"]["use_space_char"] = True

    # Train from scratch first because Myanmar dictionary is custom.
    cfg["Global"]["pretrained_model"] = None
    cfg["Global"]["checkpoints"] = None

    patch_dataset_section(cfg.get("Train"), data_dir, train_label)
    patch_dataset_section(cfg.get("Eval"), data_dir, test_label)

    patch_batch_size(cfg.get("Train"), args.batch_size)
    patch_batch_size(cfg.get("Eval"), args.batch_size)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"Saved config: {output_path}")
    print(f"Data dir: {data_dir}")
    print(f"Train label: {train_label}")
    print(f"Eval label: {test_label}")
    print(f"Dictionary: {dict_path}")
    print(f"Save model dir: {save_model_dir}")


if __name__ == "__main__":
    main()
