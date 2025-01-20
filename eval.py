import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse

from model import VisionTransformer

def evaluate(args):
    # Dataset and DataLoader (CIFAR-10 test set)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Instantiate model
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=10,
        emb_dim=args.emb_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, default="vit_cifar10.pth")
    args = parser.parse_args()

    evaluate(args)
