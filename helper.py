import os
import subprocess
import shutil
import sys

_PIP_NAME = {
    "yaml":      "pyyaml",
    "PIL":       "Pillow",
    "sklearn":   "scikit-learn",
    "cv2":       "opencv-python",
    "tqdm":      "tqdm",
    "timm":      "timm",
    "tensorboard": "tensorboard",
}

def install_modules(modules):
    """
    Auto-install missing Python packages using `uv pip install`.

    Each entry in `modules` can be:
      - A plain string  → treated as both the import name and pip name
                          (falls back to _PIP_NAME mapping if import differs from pip)
      - A 2-tuple       → (import_name, pip_package_name)

    Examples:
        install_modules(["numpy", "torch", "yaml", "PIL"])
        install_modules([("yaml", "pyyaml"), ("PIL", "Pillow")])
    """
    for item in modules:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            import_name, pip_name = item
        else:
            import_name = item
            pip_name = _PIP_NAME.get(import_name, import_name)

        try:
            __import__(import_name)
        except ImportError:
            print(f"[helper] Installing '{pip_name}' (import: {import_name}) ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

def create_structure(structure):

    """
    This function creates directories 
    
    """
    dataset_dirs = []  

    for root, sub_dirs, files in structure:
        if not os.path.exists(root):
            os.makedirs(root)
            print(f"Created directory: {root}")
        for sub_dir in sub_dirs:
            path = os.path.join(root, sub_dir)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created sub-directory: {path}")

            
            if sub_dir == "dataset":
                dataset_dirs.append(path)

        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("# Created By LORD \n")
                print(f"Created file: {file_path}")
    
    return dataset_dirs

def copy_dataset(source_path, target_dirs):
    """
    Copies the dataset from the source path to multiple target directories.
    Works on both Windows and macOS/Linux.
    """
    for target_dir in target_dirs:
        if os.path.exists(source_path):
            try:
                if not os.listdir(target_dir):  
                    shutil.copytree(source_path, target_dir, dirs_exist_ok=True)
                    print(f"Copied dataset to: {target_dir}")
                else:
                    print(f"Skipped copying to {target_dir} (already contains files).")
            except Exception as e:
                print(f"Error copying dataset to {target_dir}: {e}")
        else:
            print(f"Source dataset path {source_path} does not exist!")

def download_dataset(link , dataset_dirs):
    import kagglehub

    if input("Do you want to download the dataset? (yes/no): ").strip().lower() == "yes":
        dataset_path = kagglehub.dataset_download(link)
        print(f"Dataset downloaded at: {dataset_path}")
        copy_dataset(dataset_path, dataset_dirs)

if __name__ == "__main__":

    required_modules = [
        "numpy", "torch", "torchvision", "timm",
        "yaml", "PIL", "sklearn", "matplotlib", "tensorboard", "tqdm",
    ]

    project_structure = [
        (".",         ["configs", "checkpoints", "logs", "results", "data"],  []),
        ("results",   ["gradcam"],                                             []),
        ("models",    [],                                                      ["__init__.py"]),
        ("training",  [],                                                      ["__init__.py"]),
    ]

    install_modules(required_modules)
    dataset_dirs = create_structure(project_structure)
    print("Helper setup complete!")

