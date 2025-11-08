#!/usr/bin/env python3
from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

DOWNLOAD_URL = "https://bprna.cgrb.oregonstate.edu/bpRNA_1m_90.zip"
TARGET_SUBDIR = Path("data/unzipped")
NESTED_SUFFIXES = (
    "_stfiles.zip",
    "_stafiles.zip",
    "_dbnfiles.zip",
    "_bpseqfiles.zip",
    "_boseqfiles.zip",
)

_TEMP_ZIP: Path | None = None


def _cleanup_temp() -> None:
    global _TEMP_ZIP
    if _TEMP_ZIP is not None:
        _TEMP_ZIP.unlink(missing_ok=True)
        _TEMP_ZIP = None


def _register_cleanup(path: Path) -> None:
    global _TEMP_ZIP
    _TEMP_ZIP = path
    atexit.register(_cleanup_temp)

    def _handler(signum, frame):
        _cleanup_temp()
        # exit with 1 on Ctrl+C/TERM
        sys.exit(1)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def ensure_unzip() -> None:
    if shutil.which("unzip") is None:
        sys.stderr.write("Error: 'unzip' is required in PATH.\n")
        sys.exit(1)


def download_zip(url: str) -> Path:
    fd, tmp_path = tempfile.mkstemp(prefix="bpRNA_", suffix=".zip")
    os.close(fd)
    tmp_file = Path(tmp_path)

    try:
        with urlopen(url) as resp, tmp_file.open("wb") as out:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            chunk_size = 1 << 20
            with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))
    except Exception:
        tmp_file.unlink(missing_ok=True)
        raise

    _register_cleanup(tmp_file)
    return tmp_file


def unzip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-o", str(zip_path), "-d", str(dest)], check=True)


def should_extract(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(sfx) for sfx in NESTED_SUFFIXES)


def flatten_if_single_dir(path: Path) -> None:
    items = list(path.iterdir())
    if len(items) == 1 and items[0].is_dir():
        inner = items[0]
        for child in inner.iterdir():
            shutil.move(str(child), path / child.name)
        inner.rmdir()


def main() -> None:
    ensure_unzip()

    repo_root = Path(__file__).resolve().parents[1]
    target_dir = repo_root / TARGET_SUBDIR

    try:
        zip_path = download_zip(DOWNLOAD_URL)
        print(f"Extracting top-level archive to {target_dir} ...")
        unzip(zip_path, target_dir)

        nested = [z for z in target_dir.rglob("*.zip") if should_extract(z.name)]
        if not nested:
            print("No matching nested zips found.")
            return

        for z in tqdm(nested, desc="Extracting nested zips", unit="zip"):
            dest = z.with_name(z.stem)
            unzip(z, dest)
            flatten_if_single_dir(dest)
    finally:
        _cleanup_temp()


if __name__ == "__main__":
    main()