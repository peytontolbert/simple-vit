import subprocess
import sys

if __name__ == "__main__":
    print("Running training script...")
    train_cmd = [
        sys.executable, "train.py",
        "--image_size", "32",
        "--patch_size", "4",
        "--emb_dim", "128",
        "--depth", "6",
        "--num_heads", "4",
        "--mlp_ratio", "4.0",
        "--dropout", "0.1",
        "--batch_size", "64",
        "--epochs", "2",
        "--lr", "1e-3",
        "--print_freq", "100",
        "--save_path", "vit_cifar10.pth"
    ]
    subprocess.run(train_cmd)

    print("Running evaluation script...")
    eval_cmd = [
        sys.executable, "eval.py",
        "--image_size", "32",
        "--patch_size", "4",
        "--emb_dim", "128",
        "--depth", "6",
        "--num_heads", "4",
        "--mlp_ratio", "4.0",
        "--dropout", "0.1",
        "--batch_size", "64",
        "--checkpoint", "vit_cifar10.pth"
    ]
    subprocess.run(eval_cmd)
