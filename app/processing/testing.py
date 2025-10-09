#!/usr/bin/env python3
import os
import uuid
import zipfile
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import pandas as pd
import matplotlib.pyplot as plt

from . import models

# ================= PATHS =================
MODELS_DIR = Path("app/models")
CNN_WEIGHTS = MODELS_DIR / "model_cnn.pth"
DEEPCNN_WEIGHTS = MODELS_DIR / "model_deepcnn.pth"
AE_WEIGHTS = MODELS_DIR / "model_sae.pth"

TEST_IMAGES_DIR = Path("number_plates_image")  # expects blur_images/ and/or clear_images/ inside or images directly
OUT_DIR = Path("test_run")
RESULTS_CSV = OUT_DIR / "results.csv"
OVERALL_CSV = OUT_DIR / "overall_metrics.csv"
ZIP_NAME = OUT_DIR / "test_results.zip"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================= METRICS =================
def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_clamped = np.clip(pred.astype(np.float32), 0.0, 1.0)
    gt_clamped = np.clip(gt.astype(np.float32), 0.0, 1.0)
    mse = float(mean_squared_error(gt_clamped, pred_clamped))
    mae = float(np.mean(np.abs(gt_clamped - pred_clamped)))  # works for HxWxC
    try:
        psnr = float(peak_signal_noise_ratio(gt_clamped, pred_clamped, data_range=1.0))
    except:
        psnr = float("nan")
    try:
        ssim = float(structural_similarity(gt_clamped, pred_clamped, data_range=1.0, multichannel=True))
    except:
        ssim = float("nan")
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}

# ================= UTILITIES =================
def load_state_dict_safely(model: torch.nn.Module, path: Path, device: torch.device):
    raw = torch.load(str(path), map_location=device)
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    new_state = {}
    for k, v in raw.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v
    try:
        model.load_state_dict(new_state, strict=True)
        return {"loaded": True, "strict": True, "missing_keys": [], "unexpected_keys": []}
    except RuntimeError as e:
        res = model.load_state_dict(new_state, strict=False)
        return {"loaded": True, "strict": False, "missing_keys": res.missing_keys, "unexpected_keys": res.unexpected_keys, "error": str(e)}

def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    np_img = t.detach().cpu().numpy()
    np_img = np.transpose(np_img, (1,2,0))
    np_img = np.clip(np_img, 0.0, 1.0)
    return np_img

def pil_loader_cv2(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_numpy_image(img_np: np.ndarray, path: str):
    img_uint8 = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def find_image_pairs(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    blur_dir = input_dir / "blur_images1"
    gt_dir = input_dir / "clear_images1"
    blur_images = sorted([p for p in blur_dir.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")]) if blur_dir.exists() else []
    gt_images_all = sorted([p for p in gt_dir.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")]) if gt_dir.exists() else []
    gt_map = {p.name: p for p in gt_images_all}
    gt_images = [gt_map.get(b.name, None) for b in blur_images]
    return blur_images, gt_images

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ================= MAIN =================
def main():
    ensure_dir(OUT_DIR)
    device = DEVICE

    # Remove old files if they exist
    if RESULTS_CSV.exists():
        RESULTS_CSV.unlink()
    if OVERALL_CSV.exists():
        OVERALL_CSV.unlink()


    # Load models
    cnn = models.CNN().to(device).eval()
    deepcnn = models.DeepCNN().to(device).eval()
    ae = models.SimpleAE().to(device).eval()

    models_map = {
        "CNN": {"model": cnn, "weights": CNN_WEIGHTS, "metrics": []},
        "DeepCNN": {"model": deepcnn, "weights": DEEPCNN_WEIGHTS, "metrics": []},
        "SimpleAE": {"model": ae, "weights": AE_WEIGHTS, "metrics": []},
    }

    for name, entry in models_map.items():
        if entry["weights"].exists():
            info = load_state_dict_safely(entry["model"], entry["weights"], device)
            print(f"Loaded {name}: {info}")
        else:
            print(f"Warning: {name} weights not found, using random init.")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    blur_images, gt_images = find_image_pairs(TEST_IMAGES_DIR)
    model_out_dirs = {}
    for name in models_map.keys():
        d = OUT_DIR / name
        ensure_dir(d)
        model_out_dirs[name] = d

    results = []

    # Process all images
    for idx, bpath in enumerate(blur_images):
        try:
            bimg = pil_loader_cv2(str(bpath))
        except Exception:
            continue
        gtpath = gt_images[idx] if idx < len(gt_images) else None
        gtimg = pil_loader_cv2(str(gtpath)) if gtpath else None
        inp_tensor = transform(bimg).unsqueeze(0).to(device)

        for mname, entry in models_map.items():
            model = entry["model"]
            with torch.no_grad():
                out = model(inp_tensor)
                _, _, h, w = out.shape
                if h != 224 or w != 224:
                    out = F.interpolate(out, size=(224,224), mode='bilinear', align_corners=False)
            out_np = tensor_to_numpy_image(out)
            if out_np.max() > 1.01 or out_np.min() < -0.01:
                out_np = (out_np - out_np.min()) / (out_np.max() - out_np.min() + 1e-12)

            out_path = model_out_dirs[mname] / f"deblur_{bpath.stem}.jpg"
            save_numpy_image(out_np, str(out_path))

            row = {"image": bpath.name, "model": mname, "output_path": str(out_path)}
            if gtimg is not None:
                gt_resized = cv2.resize(gtimg, (224,224), interpolation=cv2.INTER_LINEAR)
                gt_np = gt_resized.astype(np.float32)/255.0
                metrics = compute_metrics(out_np, gt_np)
                row.update(metrics)
                entry["metrics"].append(metrics)
            else:
                row.update({"mse": None, "mae": None, "psnr": None, "ssim": None})
            results.append(row)
        print(f"Processed image {idx+1}/{len(blur_images)}")

    # Save per-image results
    import csv
    fieldnames = ["image", "model", "mse", "mae", "psnr", "ssim", "output_path"]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # Compute overall metrics
    summary = []
    for mname, entry in models_map.items():
        metrics_list = entry["metrics"]
        if not metrics_list:
            continue
        df = pd.DataFrame(metrics_list)
        summary_row = {
            "model": mname,
            "mse_mean": df["mse"].mean(),
            "mse_std": df["mse"].std(),
            "mae_mean": df["mae"].mean(),
            "mae_std": df["mae"].std(),
            "psnr_mean": df["psnr"].mean(),
            "psnr_std": df["psnr"].std(),
            "ssim_mean": df["ssim"].mean(),
            "ssim_std": df["ssim"].std(),
        }
        summary.append(summary_row)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OVERALL_CSV, index=False)
    print(f"Saved overall metrics: {OVERALL_CSV}")

    # Plot metrics comparison
    metrics_names = ["mse_mean", "mae_mean", "psnr_mean", "ssim_mean"]
    for metric in metrics_names:
        plt.figure(figsize=(6,4))
        plt.bar(summary_df["model"], summary_df[metric])
        plt.title(f"{metric} comparison across models")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.savefig(OUT_DIR / f"{metric}_comparison.png")
        plt.close()

    # Zip all results
    with zipfile.ZipFile(ZIP_NAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(RESULTS_CSV, arcname="results.csv")
        zf.write(OVERALL_CSV, arcname="overall_metrics.csv")
        for mname in model_out_dirs:
            for p in (model_out_dirs[mname].glob("*")):
                zf.write(p, arcname=f"{mname}/{p.name}")

    print(f"Done. All results saved in: {ZIP_NAME}")

if __name__ == "__main__":
    main()
