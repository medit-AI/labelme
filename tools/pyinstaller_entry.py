import shutil
import sys
from pathlib import Path

from labelme.__main__ import main

"""
tools/pyinstaller_entry.py

Added _seed_osam_model_cache(), run only when frozen (i.e. only in the built exe, not in normal dev runs): copies any bundled model blobs into ~/.cache/osam/models/blobs on the target machine (skipping files already present), so osam finds them locally and never needs internet access on a fresh machine.
"""

def _seed_osam_model_cache() -> None:
    # osam.types.Blob.path always resolves under ~/.cache/osam/models/blobs.
    bundled_blobs_dir = Path(sys._MEIPASS) / "osam_cache_blobs"
    if not bundled_blobs_dir.is_dir():
        return

    target_blobs_dir = Path.home() / ".cache" / "osam" / "models" / "blobs"
    for src in bundled_blobs_dir.rglob("*"):
        if src.is_dir():
            continue
        dst = target_blobs_dir / src.relative_to(bundled_blobs_dir)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        _seed_osam_model_cache()
    main()
