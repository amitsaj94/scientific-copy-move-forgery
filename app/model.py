import torch
import segmentation_models_pytorch as smp


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


def load_checkpoint(model, path, device):
    """
    Load the saved training checkpoint.
    Uses torch.load safely and supports your training .pth file.
    """
    print(f"Loading checkpoint: {path}")

    # FIXED — removed invalid map_snapshot argument
    ckpt = torch.load(path, map_location=device)

    if "model_state" not in ckpt:
        raise KeyError("Checkpoint missing key 'model_state'")

    model.load_state_dict(ckpt["model_state"])

    print(f"Loaded model — epoch: {ckpt.get('epoch', '?')}")
    return model
