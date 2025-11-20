# app.py - Streamlit Demo for Scientific Copy-Move Forgery Detection 
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import os
import torch
from model import HybridForgeryModel, load_checkpoint

# -----------------------
# CONFIG
# -----------------------
IMG_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(layout="wide", page_title="Forgery Inspector — Scientific Demo")

# -----------------------
# UTIL: Preprocess / Postprocess
# -----------------------
def preprocess_pil(pil_img, img_size=IMG_SIZE):
    img = np.array(pil_img.convert("RGB"))
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size))
    img_norm = img_resized.astype("float32") / 255.0
    tensor = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).float()
    return tensor, img, (orig_h, orig_w)

def postprocess_mask(mask_tensor, orig_shape):
    mask = torch.sigmoid(mask_tensor)[0,0].cpu().detach().numpy()
    mask_resized = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
    return mask_resized

def apply_colormap(mask_float):
    mask_uint8 = (np.clip(mask_float, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    return heat

def overlay_image(img_rgb, heatmap, alpha=0.5):
    heat_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_rgb, 1-alpha, heat_rgb, alpha, 0)
    return blended

# -----------------------
# LOAD MODEL (cached)
# -----------------------
@st.cache_resource(show_spinner=False)
def get_model():
    model = HybridForgeryModel().to(DEVICE)
    # Load checkpoint from HF
    model = load_checkpoint(
        model=model,
        path=None,
        device=DEVICE,
        from_hf=True,
        hf_repo="Amitsaj/image-forgery-checkpoints",
        hf_filename="best_hybrid_stepB_v2.pth"
    )
    model.eval()
    return model

# -----------------------
# PREDICT (seg + cls)
# -----------------------
def predict(model, pil_img, threshold=0.5):
    tensor, orig_img, orig_shape = preprocess_pil(pil_img, IMG_SIZE)
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        seg_logits, cls_logit = model(tensor)
        cls_prob = torch.sigmoid(cls_logit).item()
        seg_mask = postprocess_mask(seg_logits, orig_shape)
        binary_mask = (seg_mask > threshold).astype(np.uint8)
    return {
        "orig": orig_img,
        "seg_raw": seg_mask,
        "seg_bin": binary_mask,
        "cls_prob": cls_prob
    }

# -----------------------
# GRAD-CAM (classifier explainability)
# -----------------------
def grad_cam_classifier(model, pil_img, upsample_to=None):
    model.eval()
    tensor, orig_img, orig_shape = preprocess_pil(pil_img, IMG_SIZE)
    tensor = tensor.to(DEVICE).requires_grad_(True)

    encoder = model.seg_model.encoder
    features = encoder(tensor)
    deep = features[-1]
    deep.retain_grad()

    cls_logit = model.classifier(deep)
    prob = torch.sigmoid(cls_logit)[0,0]
    cls_logit.backward(retain_graph=True)

    grads = deep.grad[0]
    activations = deep[0]
    weights = grads.mean(dim=(1,2))

    cam = (weights.view(-1,1,1) * activations).sum(dim=0).cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    if upsample_to is None:
        upsample_to = orig_shape[:2]
    cam_resized = cv2.resize(cam, (upsample_to[1], upsample_to[0]))
    return cam_resized, prob.item()

# -----------------------
# UI - Layout
# -----------------------
st.title("Forgery Inspector — Scientific Copy-Move Forensics")
st.markdown("Interactive demo for segmentation + classification. Option B: Research-grade demo with explainability.")

with st.sidebar:
    st.header("Model & Input")
    load_btn = st.button("Load model")
    examples = st.selectbox(
        "Example images (optional)",
        ["-- none --"] + [f for f in os.listdir("examples") if f.lower().endswith((".png",".jpg",".jpeg"))]
        if os.path.exists("examples") else ["-- none --"]
    )
    threshold = st.slider("Segmentation threshold", 0.0, 1.0, 0.5, 0.01)
    show_gradcam = st.checkbox("Show Grad-CAM (classifier)", value=True)
    st.markdown("**Notes:** Model trained at IMG_SIZE=384. Threshold adjusts mask sensitivity.")

# ----------------------- Load model -----------------------
if "model_obj" not in st.session_state:
    if load_btn:
        try:
            st.session_state["model_obj"] = get_model()
            st.success("Model loaded from Hugging Face ✅")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

model = st.session_state.get("model_obj", None)
if model is None:
    st.warning("Please load the model to proceed.")
    st.stop()

# ----------------------- Main app -----------------------
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if examples and examples != "-- none --":
        if st.button("Use example"):
            uploaded = open(os.path.join("examples", examples), "rb")

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

        result = predict(model, pil_img, threshold=threshold)
        orig = result["orig"]
        seg_raw = result["seg_raw"]
        seg_bin = result["seg_bin"]
        cls_prob = result["cls_prob"]

        st.markdown(f"**Classification**: **{'FORGED' if cls_prob>0.5 else 'ORIGINAL'}** — Confidence: `{cls_prob:.4f}`")

        heatmap = apply_colormap(seg_raw)
        overlay = overlay_image(orig, heatmap, alpha=0.45)
        mask_vis = (seg_bin*255).astype(np.uint8)

        st.subheader("Masks & Overlay")
        viz_cols = st.columns(3)
        viz_cols[0].image(orig, caption="Original", use_container_width=True)
        viz_cols[1].image(mask_vis, caption=f"Binary Mask (thr={threshold})", use_container_width=True)
        viz_cols[2].image(overlay, caption="Overlay (heatmap)", use_container_width=True)

        if st.button("Download binary mask"):
            is_success, im_buf_arr = cv2.imencode(".png", mask_vis)
            io_buf = io.BytesIO(im_buf_arr.tobytes())
            st.download_button(label="Download mask PNG", data=io_buf, file_name="mask.png", mime="image/png")

        if show_gradcam:
            with st.spinner("Computing Grad-CAM..."):
                try:
                    cam, cam_prob = grad_cam_classifier(model, pil_img, upsample_to=orig.shape[:2])
                    cam_heat = apply_colormap(cam)
                    cam_overlay = overlay_image(orig, cam_heat, alpha=0.5)
                    st.subheader("Grad-CAM (classifier)")
                    gc_cols = st.columns(3)
                    gc_cols[0].image(cam, caption="Grad-CAM (float)", use_container_width=True)
                    gc_cols[1].image(cam_heat, caption="Grad-CAM heatmap", use_container_width=True)
                    gc_cols[2].image(cam_overlay, caption="Grad-CAM overlay", use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
    else:
        st.info("Upload an image to run detection or choose an example.")

with col2:
    st.subheader("Model & Training Insights")
    st.write("Model: HybridForgeryModel (UNet encoder = EfficientNet-B3)")
    st.write("Image size used for training:", IMG_SIZE)
    if os.path.exists("training_plots/loss.png"):
        st.image("training_plots/loss.png", caption="Loss curve", use_container_width=True)
    else:
        st.info("Training plots not found. Save plots to `training_plots/loss.png` to display them.")

    st.markdown("### Sample predictions (from validation)")
    samples_dir = "visualizations"
    if os.path.exists(samples_dir):
        sample_files = [os.path.join(samples_dir,f) for f in os.listdir(samples_dir) if f.lower().endswith((".png",".jpg"))]
        for s in sample_files[:6]:
            st.image(s, use_container_width=True)
    else:
        st.info("Place sample prediction images in `visualizations/` to show them here.")

st.markdown("---")
st.markdown("Project: Scientific Copy-Move Forgery Detection — Demo (Option B)")
st.markdown("Built for portfolio / research. Features: segmentation + classification, Grad-CAM, download mask.")
