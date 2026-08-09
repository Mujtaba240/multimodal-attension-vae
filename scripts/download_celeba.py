"""Download CelebA to Google Drive. Run once from Colab."""

import os
import subprocess
import sys


def download(data_dir):
    os.makedirs(data_dir, exist_ok=True)

    img_dir = os.path.join(data_dir, "img_align_celeba")
    if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 200000:
        print(f"CelebA already exists at {data_dir} ({len(os.listdir(img_dir))} images)")
        return

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
    import gdown

    files = {
        "img_align_celeba.zip": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
        "list_attr_celeba.txt": "0B7EVK8r0v71pblRyaVFSWGxPY0U",
        "list_eval_partition.txt": "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
    }

    for fname, file_id in files.items():
        output = os.path.join(data_dir, fname)
        if os.path.exists(output):
            print(f"  {fname} exists, skipping")
            continue
        print(f"  Downloading {fname}...")
        gdown.download(id=file_id, output=output, quiet=False)

    zip_path = os.path.join(data_dir, "img_align_celeba.zip")
    if os.path.exists(zip_path) and not os.path.exists(img_dir):
        print("  Extracting images...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print(f"  Done — {len(os.listdir(img_dir))} images extracted")


if __name__ == "__main__":
    default_dir = "/content/drive/MyDrive/Sapienza University/DL&AI_Project/data/celeba"
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    download(data_dir)