"""Extract every video under --input-dir to 2 fps JPEG frames, one directory per video.

<input-dir>/<sub>/<name>.mp4 -> <output-dir>/<sub>/<name>/<name>_000001.jpg, ...

    python -m cigo.data.extract_frames --input-dir videos/ --output-dir $CIGO_DATA_ROOT/<dataset>/video_14400frames_fps2
"""

import argparse
import os
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


def extract(job) -> str:
    """Frames of one video, written to <dir>.part and renamed on success so a killed run leaves no short dir."""
    video, frame_dir, fps, max_frames = job
    if frame_dir.is_dir() and any(frame_dir.iterdir()):
        return ""
    part = frame_dir.with_name(frame_dir.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-vframes",
        str(max_frames),
        str(part / f"{frame_dir.name}_%06d.jpg"),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as err:
        shutil.rmtree(part, ignore_errors=True)
        return f"{video}: {err}"
    shutil.rmtree(frame_dir, ignore_errors=True)
    part.replace(frame_dir)
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=14400, help="frames per video, at most (2 h at 2 fps)")
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    src, dst = Path(args.input_dir), Path(args.output_dir)
    videos = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    jobs = [(v, dst / v.relative_to(src).with_suffix(""), args.fps, args.max_frames) for v in videos]
    with Pool(args.num_workers) as pool:
        errors = [e for e in tqdm(pool.imap_unordered(extract, jobs), total=len(jobs)) if e]
    print(f"{len(jobs) - len(errors)} of {len(jobs)} videos extracted to {dst}")
    for e in errors:
        print(e)


if __name__ == "__main__":
    main()
