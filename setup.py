#!/usr/bin/env python3
"""
CropGuard NE — One-Click Setup Script
======================================
Downloads PlantVillage dataset from Kaggle, verifies structure, and kicks off training.

Prerequisites:
    pip install kaggle tensorflow pillow numpy scikit-learn matplotlib seaborn streamlit anthropic

Kaggle API key:
    1. Go to https://www.kaggle.com/settings → "Create New Token"
    2. Place kaggle.json at ~/.kaggle/kaggle.json  (Linux/Mac)
                          or C:\\Users\\YOU\\.kaggle\\kaggle.json  (Windows)
    3. chmod 600 ~/.kaggle/kaggle.json
"""

import os, sys, subprocess, shutil, zipfile, pathlib

ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
PLOT_DIR  = ROOT / "plots"


def run(cmd: str, check=True):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def check_kaggle_credentials():
    kaggle_json = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌  ~/.kaggle/kaggle.json not found.")
        print("   1. Visit https://www.kaggle.com/settings → 'Create New Token'")
        print("   2. Save the downloaded file to ~/.kaggle/kaggle.json")
        print("   3. Run:  chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    print("✅  Kaggle credentials found.")


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "plantvillage-dataset.zip"

    if (DATA_DIR / "plantvillage dataset").exists():
        print("✅  Dataset already downloaded.")
        return

    print("\n[1/4] Downloading PlantVillage dataset from Kaggle (~1.5 GB)...")
    run(f"kaggle datasets download abdallahalidev/plantvillage-dataset -p {DATA_DIR}")

    print("\n[2/4] Unzipping dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    zip_path.unlink()
    print("✅  Dataset extracted.")


def verify_structure():
    color_dir = DATA_DIR / "plantvillage dataset" / "color"
    if not color_dir.exists():
        print(f"❌  Expected directory not found: {color_dir}")
        print("   Check the unzipped structure manually.")
        sys.exit(1)

    classes = [d for d in color_dir.iterdir() if d.is_dir()]
    total_images = sum(len(list(c.glob("*.jpg"))) + len(list(c.glob("*.JPG"))) + len(list(c.glob("*.png"))) for c in classes)

    print(f"\n✅  Dataset verified:")
    print(f"   Classes : {len(classes)}")
    print(f"   Images  : {total_images:,}")
    if len(classes) != 38:
        print(f"⚠️   Expected 38 classes, found {len(classes)}")


def train():
    MODEL_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    print("\n[3/4] Starting model training (may take 2–4 hours without GPU)...")
    print("      Use Google Colab / Kaggle Notebook with GPU for faster training.\n")
    run(f"python {ROOT}/model/train.py")


def launch_app():
    print("\n[4/4] Launching Streamlit app...")
    run(f"streamlit run {ROOT}/app/app.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CropGuard NE Setup")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-train",    action="store_true", help="Skip model training (use Claude AI fallback)")
    parser.add_argument("--app-only",      action="store_true", help="Just launch the Streamlit app")
    args = parser.parse_args()

    print("="*60)
    print("  CropGuard NE — Setup Script")
    print("="*60)

    if args.app_only:
        launch_app()
        sys.exit(0)

    if not args.skip_download:
        check_kaggle_credentials()
        download_dataset()
        verify_structure()

    if not args.skip_train:
        train()

    launch_app()
