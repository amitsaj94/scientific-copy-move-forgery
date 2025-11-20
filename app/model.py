import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
import os

class HybridForgeryModel(torch.nn.Module):
    """
    Hybrid Segmentation + Classification model:
    U-Net (EfficientNet-B3) + Classification head.
    """

    def __init__(self, encoder_name="efficientnet-b3", pretrained=True):
        super().__init__()

        # Segmentation backbone
        self.seg_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1
        )

        # Classification head from encoder bottleneck
        bottleneck = self.seg_model.encoder.out_channels[-1]

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(bottleneck, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        # Encoder
        enc_feats = self.seg_model.encoder(x)
        deep = enc_feats[-1]

        # Classification
        cls_logit = self.classifier(deep)

        # Segmentation
        dec = self.seg_model.decoder(enc_feats)
        seg = self.seg_model.segmentation_head(dec)

        return seg, cls_logit


def load_checkpoint(model, path, device, from_hf=False, hf_repo=None, hf_filename=None):
    """
    Load the saved training checkpoint.
    Supports:
      - Local .pth files
      - Hugging Face Hub hosted checkpoints
    """
    if from_hf:
        if hf_repo is None or hf_filename is None:
            raise ValueError("hf_repo and hf_filename must be provided for HF download")
        print(f"Downloading checkpoint from HF: {hf_repo}/{hf_filename}")
        path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
        print(f"Downloaded to: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    print(f"Loading checkpoint: {path}")

    # PyTorch 2.6+ safe load: weights_only=False to allow numpy globals
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Determine correct key
    if "model_state" in ckpt:
        key = "model_state"
    elif "model_state_dict" in ckpt:
        key = "model_state_dict"
    else:
        raise KeyError("Checkpoint missing 'model_state' or 'model_state_dict' key")

    model.load_state_dict(ckpt[key])
    print(f"Loaded model — epoch: {ckpt.get('epoch', '?')}")
    return model
