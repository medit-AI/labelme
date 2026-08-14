"""Build the standalone Labelme executable with PyInstaller."""

import subprocess
import sys
from pathlib import Path

import osam

here: Path = Path(__file__).parent
repo_root: Path = here.parent
labelme_path: Path = repo_root / "labelme"
osam_path: Path = Path(osam.__file__).resolve().parent
# Mirrors osam.types.Blob.path: models already pulled by osam live here, keyed
# by hash, regardless of which model (sam2, sam3, efficientsam, ...) they are.
osam_cache_blobs_path: Path = Path.home() / ".cache" / "osam" / "models" / "blobs"


def main() -> None:
    args: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(here / "pyinstaller_entry.py"),
        "--name=Labelme",
        "--windowed",
        "--noconfirm",
        f"--specpath={repo_root / 'build'}",
        "--add-data="
        f"{osam_path / '_models' / 'yoloworld' / 'clip' / 'bpe_simple_vocab_16e6.txt.gz'}"
        ";osam/_models/yoloworld/clip",
        f"--add-data={labelme_path / '_config' / 'default_config.yaml'}"
        ";labelme/_config",
        f"--add-data={labelme_path / 'icons'};labelme/icons",
        f"--add-data={labelme_path / 'translate'};labelme/translate",
        f"--icon={labelme_path / 'icons' / 'icon.ico'}",
        "--onedir",
    ]

    if osam_cache_blobs_path.is_dir() and any(osam_cache_blobs_path.iterdir()):
        print(f"Bundling already-downloaded osam models from {osam_cache_blobs_path}")
        args.append(f"--add-data={osam_cache_blobs_path};osam_cache_blobs")
    else:
        print(
            f"No cached osam models found at {osam_cache_blobs_path}; "
            "the packaged app will download models on first use."
        )

    subprocess.check_call(args, cwd=repo_root)


if __name__ == "__main__":
    main()
