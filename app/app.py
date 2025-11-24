# app.py - Streamlit Demo for Scientific Copy-Move Forgery Detection (clean, memory-optimized)
import os
import io
import gc
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch

# Local imports (model.py should provide HybridForgeryModel & load_checkpoint)
from model import HybridForgeryModel, load_checkpoint
from huggingface_hub import hf_hub_download


# CONFIG / DEVICES

# Force CPU for Streamlit Cloud (no GPU). This reduces memory spikes on hosted runtimes.
DEVICE = torch.device("cpu")
# Limit MKL/OpenMP threads to reduce memory / CPU contention in hosted envs
torch.set_num_threads(1)

IMG_SIZE = 384
HF_REPO = "Amitsaj/image-forgery-checkpoints"   # change if needed
HF_FILENAME = "best_hybrid_stepB_v2.pth"        # file on HF
DEFAULT_LOCAL_CKPT = "checkpoints/best_hybrid_stepB_v2.pth"  # fallback if present locally

st.set_page_config(layout="wide", page_title="Forgery Inspector — Scientific Demo")


# UTIL: Preprocess / Postprocess

def preprocess_pil(pil_img, img_size=IMG_SIZE):
    img = np.array(pil_img.convert("RGB"))
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype("float32") / 255.0
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, img, (orig_h, orig_w)

def postprocess_mask(mask_tensor, orig_shape):
    # mask_tensor might be a torch.Tensor with grad; ensure detach then sigmoid
    with torch.no_grad():
        mask = torch.sigmoid(mask_tensor)[0, 0].cpu().detach().numpy()
    mask_resized = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
    return mask_resized

def apply_colormap(mask_float):
    mask_uint8 = (np.clip(mask_float, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    return heat

def overlay_image(img_rgb, heatmap, alpha=0.5):
    heat_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_rgb, 1 - alpha, heat_rgb, alpha, 0)
    return blended


# CACHED: Download checkpoint from Hugging Face (cached)

@st.cache_data(show_spinner=False)
def cached_hf_ckpt(repo_id: str, filename: str, token_env: str = "HF_TOKEN"):
    """
    Download checkpoint from HF and cache the path.
    Uses HF_TOKEN environment variable if set (recommended for private repos).
    """
    token = os.environ.get(token_env)
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        return path
    except Exception as e:
        # Fallback: if user has a local checkpoint (useful for local dev)
        local = os.path.abspath(DEFAULT_LOCAL_CKPT)
        if os.path.exists(local):
            return local
        raise RuntimeError(f"Failed to download checkpoint from HF and no local fallback found: {e}")


# CACHED: Load model (cached resource)

@st.cache_resource(show_spinner=False)
def get_model(ckpt_path: str):
    """
    Create model instance and load weights from ckpt_path using user-provided load_checkpoint implementation.
    This function is cached by Streamlit so the model is instantiated only once per process.
    """
    model = HybridForgeryModel().to(DEVICE)
    # load_checkpoint should handle torch.load weight issues (weights_only / safe globals) as you implemented
    model = load_checkpoint(model, ckpt_path, DEVICE)
    model.eval()
    return model


# PREDICT (segmentation + classification)

def predict(model, pil_img, threshold=0.5):
    tensor, orig_img, orig_shape = preprocess_pil(pil_img, IMG_SIZE)
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        seg_logits, cls_logit = model(tensor)
        cls_prob = float(torch.sigmoid(cls_logit).item())
        seg_mask = postprocess_mask(seg_logits, orig_shape)
        binary_mask = (seg_mask > threshold).astype(np.uint8)
    # Cleanup
    del tensor, seg_logits, cls_logit
    gc.collect()
    return {
        "orig": orig_img,
        "seg_raw": seg_mask,
        "seg_bin": binary_mask,
        "cls_prob": cls_prob
    }


# SAFE GRAD-CAM (no long-lived autograd graphs)


  """
    Compute Grad-CAM for classifier in a memory-safe way:
      - Run encoder under no_grad to avoid storing intermediate grads for the whole graph.
      - Clone + detach the last encoder activation and enable grad only on that tensor.
      - Backprop from classification score to obtain grads for the cloned activation.
      - Detach everything before converting to numpy.
    """

def grad_cam_classifier(model, pil_img, upsample_to=None):
  
    model.eval()
    tensor, orig_img, orig_shape = preprocess_pil(pil_img, IMG_SIZE)
    tensor = tensor.to(DEVICE)

    # 1) Forward encoder under no_grad to avoid storing big autograd buffers
    with torch.no_grad():
        encoder = model.seg_model.encoder
        features = encoder(tensor)

    # 2) Take deep activation, clone & require grad (isolates graph)
    deep = features[-1].clone().detach().requires_grad_(True)

    # 3) Forward classifier on the detached deep activation (classifier expects encoder-bottleneck shape)
    cls_logit = model.classifier(deep)
    prob = torch.sigmoid(cls_logit)[0, 0]

    # 4) Backprop only with respect to deep
    model.zero_grad(set_to_none=True)
    prob.backward(retain_graph=False)

    # 5) Extract gradients and activations, detach before numpy
    grads = deep.grad[0].detach().cpu()
    activations = deep[0].detach().cpu()
    weights = grads.mean(dim=(1, 2))

    cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)
    cam = torch.clamp(cam, min=0)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)
    cam_np = cam.numpy()

    if upsample_to is None:
        upsample_to = orig_shape[:2]

    cam_resized = cv2.resize(cam_np, (upsample_to[1], upsample_to[0]), interpolation=cv2.INTER_LINEAR)

    # Cleanup
    del tensor, features, deep, cls_logit, grads, activations, weights, cam, cam_np
    gc.collect()

    return cam_resized, float(prob.item())


# UI - Layout

st.title("Forgery Inspector — Scientific Copy-Move Forensics")
st.markdown("Interactive demo for segmentation + classification. Research demo with explainability.")

# Sidebar - model controls + examples
with st.sidebar:
    st.header("Model & Input")

    # Download and cache the checkpoint path (runs fast after first call)
    st.write("Checkpoint source:")
    try:
        ckpt_path_cached = cached_hf_ckpt(HF_REPO, HF_FILENAME)
        st.success("Checkpoint ready.")
    except Exception as e:
        st.error(f"Checkpoint download failed: {e}")
        st.stop()

    # Button to load model (cached by Streamlit)
    if st.button("Load model"):
        try:
            st.session_state["model_obj"] = get_model(ckpt_path_cached)
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    # Example images - use absolute path so Cloud finds them
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")
    examples_list = ["-- none --"]
    if os.path.exists(EXAMPLES_DIR):
        examples_list += sorted(
            [f for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

    examples = st.selectbox("Example images (optional)", examples_list)
    threshold = st.slider("Segmentation threshold", 0.0, 1.0, 0.5, 0.01)
    show_gradcam = st.checkbox("Show Grad-CAM (classifier)", value=True)
    st.markdown("Notes: Model trained at 384×384. Adjust threshold to tune mask sensitivity.")

# Ensure model loaded (if user previously loaded in session or not)
if "model_obj" not in st.session_state:
    # Try auto-load (convenience for users who didn't press Load model)
    try:
        st.session_state["model_obj"] = get_model(ckpt_path_cached)
        st.info("Model auto-loaded.")
    except Exception as e:
        st.error(f"Model not loaded: {e}")
        st.stop()

model = st.session_state["model_obj"]

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    # If example chosen, open its file for processing
    if examples and examples != "-- none --":
        if st.button("Use example"):
            uploaded = open(os.path.join(EXAMPLES_DIR, examples), "rb")

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Uploaded image", width="stretch")

        # Run prediction (seg + cls)
        result = predict(model, pil_img, threshold=threshold)
        orig = result["orig"]
        seg_raw = result["seg_raw"]
        seg_bin = result["seg_bin"]
        cls_prob = result["cls_prob"]

        st.markdown(f"**Classification**: **{'FORGED' if cls_prob>0.5 else 'ORIGINAL'}** — Confidence: `{cls_prob:.4f}`")

        # Visualizations
        heatmap = apply_colormap(seg_raw)
        overlay = overlay_image(orig, heatmap, alpha=0.45)
        mask_vis = (seg_bin * 255).astype(np.uint8)

        st.subheader("Masks & Overlay")
        viz_cols = st.columns(3)
        viz_cols[0].image(orig, caption="Original", width="stretch")
        viz_cols[1].image(mask_vis, caption=f"Binary Mask (thr={threshold})", width="stretch")
        viz_cols[2].image(overlay, caption="Overlay (heatmap)", width="stretch")

        # Download mask button
        if st.button("Download binary mask"):
            is_success, im_buf_arr = cv2.imencode(".png", mask_vis)
            io_buf = io.BytesIO(im_buf_arr.tobytes())
            st.download_button(label="Download mask PNG", data=io_buf, file_name="mask.png", mime="image/png")

        # Grad-CAM (safe)
        if show_gradcam:
            with st.spinner("Computing Grad-CAM..."):
                try:
                    cam, cam_prob = grad_cam_classifier(model, pil_img, upsample_to=orig.shape[:2])
                    cam_heat = apply_colormap(cam)
                    cam_overlay = overlay_image(orig, cam_heat, alpha=0.5)
                    st.subheader("Grad-CAM (classifier)")
                    gc_cols = st.columns(3)
                    gc_cols[0].image(cam, caption="Grad-CAM (float)", width="stretch")
                    gc_cols[1].image(cam_heat, caption="Grad-CAM heatmap", width="stretch")
                    gc_cols[2].image(cam_overlay, caption="Grad-CAM overlay", width="stretch")
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")

        # Clean up large objects
        del pil_img, orig, seg_raw, seg_bin, mask_vis, heatmap, overlay
        gc.collect()
    else:
        st.info("Upload an image to run detection or choose an example.")

with col2:
    st.subheader("Model & Training Insights")
    st.write("Model: HybridForgeryModel (UNet encoder = EfficientNet-B3)")
    st.write("Image size used for training:", IMG_SIZE)

    # Use BASE_DIR to find training plots on Cloud
    loss_plot = os.path.join(BASE_DIR, "training_plots", "loss.png")
    if os.path.exists(loss_plot):
        st.image(loss_plot, caption="Loss curve", width="stretch")
    else:
        st.info("Training plots not found. Save plots to `training_plots/loss.png` to display them.")

    st.markdown("### Sample predictions (from validation)")
    samples_dir = os.path.join(BASE_DIR, "visualizations")
    if os.path.exists(samples_dir):
        sample_files = sorted([os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.lower().endswith((".png", ".jpg"))])
        for s in sample_files[:6]:
            st.image(s, width="stretch")
    else:
        st.info("Place sample prediction images in `visualizations/` to show them here.")

st.markdown("---")
st.markdown("Project: Scientific Copy-Move Forgery Detection — Demo (Option B)")
st.markdown("Built for portfolio / research. Features: segmentation + classification, Grad-CAM, download mask.")

