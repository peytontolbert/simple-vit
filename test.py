import torch
from model import VisionTransformer

def test_vit_forward_pass():
    batch_size = 2
    in_channels = 3
    image_size = 32
    patch_size = 4
    num_classes = 10

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        emb_dim=128,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
    )

    # Create dummy inputs
    dummy_inputs = torch.randn(batch_size, in_channels, image_size, image_size)
    outputs = model(dummy_inputs)

    assert outputs.shape == (batch_size, num_classes), \
        f"Expected output shape {(batch_size, num_classes)}, got {outputs.shape}"
    print("Forward pass test passed!")

if __name__ == "__main__":
    test_vit_forward_pass()
