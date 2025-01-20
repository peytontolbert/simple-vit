import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import logging

from model import VisionTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args):
    # Dataset and DataLoader (CIFAR-10 for example)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Instantiate model
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=10,           # CIFAR-10 classes
        emb_dim=args.emb_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % args.print_freq == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/args.print_freq:.4f}")
                running_loss = 0.0

    # Save the model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="vit_cifar10.pth")
    args = parser.parse_args()

    train(args)
