# main.py
import argparse
import json
import math
import time
from pathlib import Path

import cv2


def load_config(path: str) -> dict:
    """Load settings. Keep tuning knobs in config.json so you don't edit code every time."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("source", 0)
    cfg.setdefault("output_dir", "runs")
    cfg.setdefault("roi", None)

    # Roundness threshold: 1.0 is a perfect circle, lower means less circle-like.
    cfg.setdefault("circularity_threshold", 0.80)

    # Basic preprocessing + filtering
    cfg.setdefault("blur_ksize", 5)
    cfg.setdefault("min_contour_area", 200)

    # Extra filter: reject contours that don't fill their enclosing circle enough
    # (helps avoid picking UI boxes / borders / random blobs)
    cfg.setdefault("min_fill_ratio", 0.35)

    # Preview / saving
    cfg.setdefault("show_preview", True)
    cfg.setdefault("save_every_n_frames", 5)

    return cfg


def ensure_run_dir(base_dir: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    return run_dir


def crop_roi(frame, roi):
    """Optional crop if you want to only inspect part of the image."""
    if roi is None:
        return frame, (0, 0)
    try:
        x, y, w, h = map(int, roi)
        return frame[y:y + h, x:x + w], (x, y)
    except Exception:
        return frame, (0, 0)


def make_bw(gray, blur_ksize: int):
    """Convert grayscale -> clean binary mask so contours are easier to find."""
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    g = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    bw = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    # Light cleanup so tiny noise doesn't become a contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    return bw


def contour_circularity(contour) -> float:
    """Roundness score: 1.0 is ideal circle."""
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return 0.0
    return float((4.0 * math.pi * area) / (perim * perim))


def pick_best_contour(bw, cfg):
    """
    Choose the most likely 'part' contour.
    We filter out obvious junk (full screenshot border, too small, not circle-like),
    then take the largest remaining contour.
    """
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = bw.shape[:2]
    min_area = int(cfg["min_contour_area"])
    min_fill = float(cfg["min_fill_ratio"])

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(c)

        # Common screenshot problem: a big rectangle around the whole image
        if cw > 0.90 * w and ch > 0.90 * h:
            continue

        # Circle-likeness filter: compare contour area to area of its enclosing circle
        (xc, yc), r = cv2.minEnclosingCircle(c)
        circle_area = math.pi * (r * r)
        if circle_area <= 0:
            continue

        fill_ratio = area / circle_area
        if fill_ratio < min_fill:
            continue

        candidates.append((area, c))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def inspect_circle_quality(frame, cfg):
    """Returns (contour, decision, circularity, offset)."""
    view, offset = crop_roi(frame, cfg.get("roi"))
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    bw = make_bw(gray, int(cfg["blur_ksize"]))

    contour = pick_best_contour(bw, cfg)
    if contour is None:
        return None, "FAIL", 0.0, offset

    circ = contour_circularity(contour)
    decision = "PASS" if circ >= float(cfg["circularity_threshold"]) else "FAIL"
    return contour, decision, float(circ), offset


def annotate(frame, contour, decision, circularity, cfg, offset):
    """Draw the found contour + a simple status label."""
    out = frame.copy()
    ox, oy = offset

    if contour is not None:
        contour_shifted = contour.copy()
        contour_shifted[:, 0, 0] += ox
        contour_shifted[:, 0, 1] += oy

        cv2.drawContours(out, [contour_shifted], -1, (0, 255, 255), 2)

        (x, y), r = cv2.minEnclosingCircle(contour)
        cv2.circle(out, (int(x + ox), int(y + oy)), int(r), (255, 255, 0), 2)

    color = (0, 255, 0) if decision == "PASS" else (0, 0, 255)
    cv2.putText(out, f"{decision}  circularity={circularity:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(out, f"threshold={cfg['circularity_threshold']:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if cfg.get("roi") is not None:
        x, y, w, h = map(int, cfg["roi"])
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return out


def run_folder(cfg: dict, folder: str):
    run_dir = ensure_run_dir(cfg["output_dir"])
    print(f"[run] Saving outputs to: {run_dir}")

    folder_path = Path(folder)
    if not folder_path.exists():
        print("[error] Folder not found:", folder)
        return 2

    (run_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    log_path = run_dir / "predictions.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("image_name,decision,circularity\n")

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_files.extend(folder_path.glob(ext))

    if not image_files:
        print("[error] No images found in folder.")
        return 2

    for img_path in sorted(image_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        contour, decision, circ, offset = inspect_circle_quality(frame, cfg)
        annotated = annotate(frame, contour, decision, circ, cfg, offset)

        out_path = run_dir / "frames" / f"{img_path.stem}_{decision}.jpg"
        cv2.imwrite(str(out_path), annotated)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{img_path.name},{decision},{circ:.5f}\n")

    print(f"[run] Saved annotated images to: {run_dir / 'frames'}")
    print(f"[run] Saved log to: {log_path}")
    return 0


def run_webcam(cfg: dict):
    run_dir = ensure_run_dir(cfg["output_dir"])
    print(f"[run] Saving outputs to: {run_dir}")
    (run_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    cap = cv2.VideoCapture(cfg["source"])
    if not cap.isOpened():
        print("[error] Could not open source:", cfg["source"])
        return 2

    log_path = run_dir / "predictions.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("frame_idx,decision,circularity\n")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        contour, decision, circ, offset = inspect_circle_quality(frame, cfg)
        annotated = annotate(frame, contour, decision, circ, cfg, offset)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{frame_idx},{decision},{circ:.5f}\n")

        save_every = max(1, int(cfg["save_every_n_frames"]))
        if frame_idx % save_every == 0:
            out_path = run_dir / "frames" / f"frame_{frame_idx:06d}_{decision}.jpg"
            cv2.imwrite(str(out_path), annotated)

        if cfg["show_preview"]:
            cv2.imshow("Circle Quality Inspection (q to quit)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        frame_idx += 1

    cap.release()
    if cfg["show_preview"]:
        cv2.destroyAllWindows()

    print(f"[run] Saved log: {log_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--mode", choices=["webcam", "folder"], default="folder")
    ap.add_argument("--folder", default="data/labeled/pass")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.mode == "webcam":
        return run_webcam(cfg)
    return run_folder(cfg, args.folder)


if __name__ == "__main__":
    raise SystemExit(main())
