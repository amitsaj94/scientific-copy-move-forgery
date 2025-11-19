import argparse
import torch
import cv2
import numpy as np
from model import HybridForgeryModel, load_checkpoint

IMG_SIZE = 384   # same size used in training


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return img, orig


# ---------------------------------------------------------
# Mask post-processing
# ---------------------------------------------------------
def postprocess_mask(mask, orig_shape):
    mask = torch.sigmoid(mask)[0, 0].detach().cpu().numpy()
    mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
    mask = (mask * 255).astype(np.uint8)
    return mask


# ---------------------------------------------------------
# Overlay heatmap on original image
# ---------------------------------------------------------
def overlay_mask(image, mask):
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return blended


# ---------------------------------------------------------
# Final panel: original | mask | heatmap
# ---------------------------------------------------------
def make_side_by_side(orig, mask, overlay, label, prob):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    h = orig.shape[0]
    mask_rgb = cv2.resize(mask_rgb, (orig.shape[1], h))
    overlay = cv2.resize(overlay, (orig.shape[1], h))

    # Prediction text
    txt = f"{label} ({prob:.3f})"
    cv2.putText(overlay, txt, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    panel = np.hstack([orig, mask_rgb, overlay])
    return panel


# ---------------------------------------------------------
# Main inference function
# ---------------------------------------------------------
def infer_and_visualize(model_path, input_img, output_img="vis_result.png", threshold=0.5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridForgeryModel().to(device)
    model = load_checkpoint(model, model_path, device)
    model.eval()

    # Preprocess
    img, orig = preprocess_image(input_img)
    img = img.to(device)

    with torch.no_grad():
        seg, cls_logit = model(img)

    cls_prob = torch.sigmoid(cls_logit).item()
    label = "FORGED" if cls_prob > threshold else "ORIGINAL"

    mask = postprocess_mask(seg, orig.shape)
    overlay = overlay_mask(orig, mask)
    panel = make_side_by_side(orig, mask, overlay, label, cls_prob)

    cv2.imwrite(output_img, panel)

    print("\n--- VISUALIZATION READY ---")
    print(f"Prediction   : {label}")
    print(f"Confidence   : {cls_prob:.4f}")
    print(f"Saved to     : {output_img}\n")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="vis_result.png")
    parser.add_argument("--thr", type=float, default=0.5)
    args = parser.parse_args()

    infer_and_visualize(args.model, args.input, args.output, args.thr)
