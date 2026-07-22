#!/usr/bin/env python3

import subprocess
from pathlib import Path
import subprocess
def get_git_root():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], 
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return Path(root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(__file__).resolve().parents[2]
    

PROJECT_ROOT = get_git_root()
REPORTS_DIR = PROJECT_ROOT / "reports"
SCRIPT_PATH = PROJECT_ROOT / "src" / "utils" / "create_image_descriptions.py"

def main():
    pdf_files = sorted(REPORTS_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem  # e.g., "oatly_2022"
        out_dir = REPORTS_DIR / f"{pdf_name}_images"
        if out_dir.exists():
            continue
        print(f"Processing {pdf_path.name} -> {out_dir}")
        cmd = [
            "python",
            str(SCRIPT_PATH),
            "--pdf_path", str(pdf_path),
            "--output_dir", str(out_dir)
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed on {pdf_path.name}: {e}")

if __name__ == "__main__":
    main()
