# evaluate.py
import argparse
from pathlib import Path

import cv2

from main import load_config, crop_roi, make_bw, pick_best_contour, contour_circularity


def predict_one(frame, cfg):
    """Run the same logic as the main pipeline, but return just prediction + score."""
    view, _ = crop_roi(frame, cfg.get("roi"))
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    bw = make_bw(gray, int(cfg["blur_ksize"]))

    contour = pick_best_contour(bw, cfg)
    if contour is None:
        return "FAIL", 0.0

    circ = contour_circularity(contour)
    decision = "PASS" if circ >= float(cfg["circularity_threshold"]) else "FAIL"
    return decision, float(circ)


def evaluate_folder(cfg: dict, labeled_dir: str):
    labeled_path = Path(labeled_dir)
    pass_dir = labeled_path / "pass"
    fail_dir = labeled_path / "fail"

    if not pass_dir.exists() or not fail_dir.exists():
        print("[error] Expected folders:")
        print("  data/labeled/pass/")
        print("  data/labeled/fail/")
        return 2

    TP = FP = TN = FN = 0

    def process_one(img_path: Path, true_label: str):
        nonlocal TP, FP, TN, FN

        frame = cv2.imread(str(img_path))
        if frame is None:
            return

        pred, _ = predict_one(frame, cfg)

        if true_label == "PASS" and pred == "PASS":
            TP += 1
        elif true_label == "FAIL" and pred == "PASS":
            FP += 1
        elif true_label == "FAIL" and pred == "FAIL":
            TN += 1
        elif true_label == "PASS" and pred == "FAIL":
            FN += 1

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for img_path in pass_dir.glob(ext):
            process_one(img_path, "PASS")

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for img_path in fail_dir.glob(ext):
            process_one(img_path, "FAIL")

    total = TP + FP + TN + FN
    if total == 0:
        print("[error] No images found in pass/fail folders.")
        return 2

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    accuracy = (TP + TN) / total if total else 0.0

    print("=== Evaluation Results (PASS is positive) ===")
    print(f"Total samples: {total}")
    print(f"TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--labeled", default="data/labeled")
    args = ap.parse_args()

    cfg = load_config(args.config)
    return evaluate_folder(cfg, args.labeled)


if __name__ == "__main__":
    raise SystemExit(main())
