#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Download official Ultralytics YOLO26 pretrained weights for detect, segment, pose and depth tasks.

Weights are fetched from the Ultralytics GitHub release assets (default tag: v8.4.0). The script only uses the
Python standard library, so it works in any environment (including ones without torch/cv2 installed) and supports
resumable downloads, retries and concurrent jobs.

Examples:
    # nano models for all four tasks (default)
    python download_weights.py

    # only pose + depth, nano size
    python download_weights.py -t pose depth

    # all sizes for detect + segment
    python download_weights.py -t detect segment -s all

    # download into ./weights with 4 parallel connections
    python download_weights.py -s n s m l x -j 4 -d weights

    # use a GitHub mirror if the official host is slow/unreachable
    python download_weights.py --base-url https://ghfast.top/https://github.com

    # only print the URLs that would be downloaded
    python download_weights.py -s all --list
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_RELEASE = "v8.4.0"
OFFICIAL_BASE_URL = "https://github.com/ultralytics/assets/releases/download"
TASK_SUFFIX = {
    "detect": "",
    "segment": "-seg",
    "pose": "-pose",
    "depth": "-depth",
}
SIZE_ORDER = "nsmlx"
MIN_BYTES = 1_000_000  # every YOLO26 weight is well above 1 MB
CHUNK_SIZE = 1 << 20  # 1 MiB


def human_size(num_bytes: float) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def build_model_names(tasks: list[str], sizes: list[str], prefix: str = "yolo26") -> list[str]:
    """Return weight filenames for the requested tasks and model sizes."""
    return [f"{prefix}{size}{TASK_SUFFIX[task]}.pt" for task in tasks for size in sizes]


def build_url(name: str, base_url: str, release: str) -> str:
    """Build the release asset URL for one weight file."""
    if release == "latest":
        return f"{base_url.rstrip('/')}/latest/download/{name}"
    return f"{base_url.rstrip('/')}/{release}/{name}"


def download_one(
    name: str,
    base_url: str,
    release: str,
    dest_dir: Path,
    *,
    force: bool = False,
    retries: int = 3,
    timeout: float = 60.0,
    show_progress: bool = True,
) -> tuple[str, str, int]:
    """Download one weight file with resume and retry support.

    Returns:
        A ``(filename, status, bytes)`` tuple where status is ``downloaded``, ``skipped`` or ``failed``.
    """
    dest = dest_dir / name
    if dest.exists() and not force:
        size = dest.stat().st_size
        if size >= MIN_BYTES:
            return name, "skipped", size
        dest.unlink()  # remove truncated/corrupt leftovers

    url = build_url(name, base_url, release)
    part = dest_dir / f"{name}.part"
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            return _download_to(url, part, dest, name, timeout=timeout, show_progress=show_progress)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                last_error = f"HTTP {e.code} {e.reason} ({url})"
                break  # retrying a missing asset is pointless
            last_error = f"HTTP {e.code} {e.reason}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < retries:
            delay = min(2**attempt, 10)
            print(f"  {name}: {last_error}, retrying in {delay}s ({attempt}/{retries})...", file=sys.stderr)
            time.sleep(delay)

    return name, f"failed ({last_error})", part.stat().st_size if part.exists() else 0


def _download_to(url: str, part: Path, dest: Path, name: str, *, timeout: float, show_progress: bool):
    """Stream a URL to ``part`` (resuming if possible), then atomically rename it to ``dest``."""
    start = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "ultralytics-weights-downloader"}
    if start:
        headers["Range"] = f"bytes={start}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if start and response.getcode() != 206:  # server ignored Range, restart from scratch
            start = 0
        content_length = response.headers.get("Content-Length")
        total = start + int(content_length) if content_length else 0
        mode = "ab" if start else "wb"
        done = start
        t0 = time.monotonic()

        with open(part, mode) as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if show_progress and total:
                    elapsed = max(time.monotonic() - t0, 1e-6)
                    speed = (done - start) / elapsed / (1 << 20)
                    pct = done / total * 100
                    sys.stdout.write(
                        f"\r  {name:<24} {pct:5.1f}%  {human_size(done):>9} / {human_size(total):<9} {speed:5.1f} MB/s"
                    )
                    sys.stdout.flush()

    final_size = part.stat().st_size
    if total and final_size != total:
        raise OSError(f"size mismatch: got {final_size} bytes, expected {total}")
    if final_size < MIN_BYTES:
        raise OSError(f"downloaded file is too small ({final_size} bytes)")

    part.replace(dest)  # atomic on the same filesystem
    if show_progress:
        sys.stdout.write("\r" + " " * 96 + "\r")
    return name, "downloaded", final_size


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official Ultralytics YOLO26 pretrained weights (detect / segment / pose / depth).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-t",
        "--tasks",
        nargs="+",
        choices=list(TASK_SUFFIX),
        default=list(TASK_SUFFIX),
        help="tasks to download (default: all four)",
    )
    parser.add_argument(
        "-s",
        "--sizes",
        nargs="+",
        default=["n"],
        help='model sizes: n s m l x, or "all" (default: n)',
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent / "weights",
        help="destination directory (default: <repo>/weights)",
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE, help=f"GitHub release tag (default: {DEFAULT_RELEASE})")
    parser.add_argument(
        "--base-url",
        default=OFFICIAL_BASE_URL,
        help="release download base URL; point it at a mirror if GitHub is slow (default: official)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="number of parallel downloads (default: 1)")
    parser.add_argument("--retries", type=int, default=3, help="download attempts per file (default: 3)")
    parser.add_argument("--timeout", type=float, default=60.0, help="network timeout in seconds (default: 60)")
    parser.add_argument("--list", action="store_true", help="print URLs and exit without downloading")
    args = parser.parse_args(argv)

    sizes = list(SIZE_ORDER) if "all" in args.sizes else args.sizes
    invalid = sorted(set(sizes) - set(SIZE_ORDER))
    if invalid:
        parser.error(f"invalid size(s): {', '.join(invalid)}; choose from {', '.join(SIZE_ORDER)} or 'all'")
    # Keep a stable order and drop duplicates.
    args.sizes = [s for s in SIZE_ORDER if s in sizes]
    args.tasks = list(dict.fromkeys(args.tasks))
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = build_model_names(args.tasks, args.sizes)

    if args.list:
        for name in names:
            print(build_url(name, args.base_url, args.release))
        return 0

    args.dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(names)} weight(s) from {args.base_url} into {args.dir}")
    print(f"  tasks: {', '.join(args.tasks)} | sizes: {', '.join(args.sizes)} | release: {args.release}\n")

    results: list[tuple[str, str, int]] = []
    show_progress = args.jobs == 1  # parallel progress bars would interleave

    if args.jobs == 1:
        for name in names:
            results.append(
                download_one(
                    name,
                    args.base_url,
                    args.release,
                    args.dir,
                    force=args.force,
                    retries=args.retries,
                    timeout=args.timeout,
                    show_progress=show_progress,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    download_one,
                    name,
                    args.base_url,
                    args.release,
                    args.dir,
                    force=args.force,
                    retries=args.retries,
                    timeout=args.timeout,
                    show_progress=False,
                ): name
                for name in names
            }
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda r: names.index(r[0]))
    print("\nSummary:")
    failed = 0
    total_bytes = 0
    for name, status, size in results:
        ok = not status.startswith("failed")
        failed += not ok
        total_bytes += size if ok else 0
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {name:<24} {human_size(size):>9}  {status}")

    print(f"\n{len(results) - failed}/{len(results)} available, {human_size(total_bytes)} on disk")
    if failed:
        print("Some downloads failed. Re-run the same command to resume partial files, or try --base-url <mirror>.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
