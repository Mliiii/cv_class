import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_points(value: str):
    if not value:
        return None, None
    coords = []
    labels = []
    for item in value.split(";"):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 3:
            raise ValueError("Point format must be 'x,y,label;x,y,label', label is 1 or 0.")
        x, y, label = parts
        coords.append([float(x), float(y)])
        labels.append(int(label))
    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int32)


def parse_box(value: str):
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Box format must be 'x1,y1,x2,y2'.")
    return np.array([float(part) for part in parts], dtype=np.float32)


def load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def draw_prompt_marks(image: np.ndarray, points=None, labels=None, box=None) -> np.ndarray:
    canvas = image.copy()
    if box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 0), 2)
    if points is not None and labels is not None:
        for (x, y), label in zip(points, labels):
            color = (0, 220, 0) if int(label) == 1 else (220, 0, 0)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 6, color, -1)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 8, (255, 255, 255), 2)
    return canvas


def save_overlay(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    score: float,
    points=None,
    labels=None,
    box=None,
) -> None:
    color = np.array([30, 144, 255], dtype=np.float32)
    overlay = image.astype(np.float32).copy()
    overlay[mask] = overlay[mask] * 0.45 + color * 0.55
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay = draw_prompt_marks(overlay, points=points, labels=labels, box=box)
    cv2.putText(
        overlay,
        f"score={score:.3f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def run_prediction(predictor, mode: str, points, labels, box, multimask_output: bool):
    if mode == "point":
        if points is None:
            raise ValueError("Point mode requires --points.")
        return predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output,
        )
    if mode == "box":
        if box is None:
            raise ValueError("Box mode requires --box.")
        return predictor.predict(
            box=box,
            multimask_output=multimask_output,
        )
    if mode == "point_box":
        if points is None or box is None:
            raise ValueError("Point-box mode requires both --points and --box.")
        return predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            multimask_output=multimask_output,
        )
    raise ValueError(f"Unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive segmentation with SAM point and box prompts.")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--checkpoint", type=str, default="", help="SAM checkpoint path, such as sam_vit_b_01ec64.pth.")
    parser.add_argument("--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--output-dir", type=str, default="./runs/sam_interactive")
    parser.add_argument("--points", type=str, default="", help="Format: x,y,label;x,y,label. label=1 positive, 0 negative.")
    parser.add_argument("--box", type=str, default="", help="Format: x1,y1,x2,y2.")
    parser.add_argument(
        "--mode",
        choices=["point", "box", "point_box", "all"],
        default="all",
        help="Prompt mode to run.",
    )
    parser.add_argument("--single-mask", action="store_true", help="Disable SAM multimask output.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Check image and prompts without loading SAM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    points, labels = parse_points(args.points)
    box = parse_box(args.box)
    image = load_image_rgb(image_path)

    prompt_preview = draw_prompt_marks(image, points=points, labels=labels, box=box)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "prompt_preview.png"), cv2.cvtColor(prompt_preview, cv2.COLOR_RGB2BGR))

    config = {
        "image": str(image_path.resolve()),
        "image_shape": list(image.shape),
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "points": points.tolist() if points is not None else None,
        "point_labels": labels.tolist() if labels is not None else None,
        "box": box.tolist() if box is not None else None,
        "mode": args.mode,
        "single_mask": args.single_mask,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Image shape: {image.shape}")
    print(f"Saved prompt preview to: {output_dir / 'prompt_preview.png'}")
    if args.dry_run:
        return

    if not args.checkpoint:
        raise ValueError("SAM checkpoint is required unless --dry-run is used.")

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "segment_anything is not installed. Install requirements.txt or use --dry-run."
        ) from exc

    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    modes = ["point", "box"] if args.mode == "all" else [args.mode]
    if args.mode == "all" and points is not None and box is not None:
        modes.append("point_box")

    summary = []
    for mode in modes:
        masks, scores, logits = run_prediction(
            predictor,
            mode=mode,
            points=points,
            labels=labels,
            box=box,
            multimask_output=not args.single_mask,
        )
        best_index = int(np.argmax(scores))
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_path = output_dir / f"{mode}_mask_{i}.png"
            overlay_path = output_dir / f"{mode}_overlay_{i}.png"
            cv2.imwrite(str(mask_path), mask_to_uint8(mask))
            save_overlay(
                overlay_path,
                image=image,
                mask=mask,
                score=float(score),
                points=points if mode in {"point", "point_box"} else None,
                labels=labels if mode in {"point", "point_box"} else None,
                box=box if mode in {"box", "point_box"} else None,
            )
            summary.append(
                {
                    "mode": mode,
                    "mask_index": i,
                    "score": float(score),
                    "area_pixels": int(mask.sum()),
                    "is_best": i == best_index,
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                }
            )
        print(f"{mode}: best mask={best_index}, score={float(scores[best_index]):.4f}")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved SAM outputs to: {output_dir}")


if __name__ == "__main__":
    main()
