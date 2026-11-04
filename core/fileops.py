import os
import subprocess
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp",
    ".tiff", ".tif", ".webp", ".svg",
    ".heic", ".heif", ".avif", ".jxl",
}

KNOWN_NON_IMAGE_EXTENSIONS = {
    ".txt", ".pdf", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".log", ".sql",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".iso",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt",
    ".sh", ".py", ".js", ".ts", ".html", ".css",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
}

MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/avif": ".avif",
    "image/jxl": ".jxl",
    "image/v": ".v",
    "image/vips": ".v",
}


# ISO-BMFF brands for AVIF, HEIC/HEIF and generic HEIF
_FTYP_IMAGE_BRANDS = {
    b"avif", b"avis",
    b"heic", b"heix", b"hevc", b"hevx",
    b"heim", b"heis", b"hevm", b"hevs",
    b"mif1", b"msf1",
}


def _sniff_image_magic(path: Path) -> bool:
    # fallback for Windows, no `file` command
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
    except OSError:
        return False
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if head.startswith(b"\xff\xd8\xff"):
        return True
    if head.startswith((b"GIF87a", b"GIF89a")):
        return True
    if head.startswith(b"BM"):
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return True
    if head[:2] == b"\xff\x0a":  # JXL codestream
        return True
    if head[:12] in (b"\x00\x00\x00\x0cjP  \r\n\x87\n",   # JPEG 2000
                      b"\x00\x00\x00\x0cJXL \r\n\x87\n"):  # JXL container
        return True
    if head[4:8] == b"ftyp":
        return head[8:12] in _FTYP_IMAGE_BRANDS
    return False


def is_image_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return True
    if suffix in KNOWN_NON_IMAGE_EXTENSIONS:
        return False
    try:
        result = subprocess.run(["file", "--mime-type", "-b", str(path)],
                                check=True, capture_output=True)
        mime_type = result.stdout.decode().strip()
        return mime_type.startswith("image/")
    except Exception:
        # no `file` (usually Windows), sniff magic bytes
        return _sniff_image_magic(path)


def is_image_mime(mime: str) -> bool:
    return mime.startswith("image/")


def collect_images(directory: Path, recursive: bool = False) -> list[Path]:
    image_files: list[Path] = []
    if recursive:
        for root, _, files in os.walk(directory):
            image_files.extend(
                fp for file in files if is_image_file(fp := Path(root) / file)
            )
    else:
        image_files.extend(
            file for file in directory.iterdir()
            if file.is_file() and is_image_file(file)
        )
    return image_files


def compute_output_name(
    input_file: Path,
    output_dir: Path,
    input_dir: Path,
    recursive: bool,
    no_suffix: bool,
    is_single_image: bool,
    target_format: str | None,
) -> Path:
    # single image keeps name, recursive mirrors subdirs
    stem = input_file.stem
    ext = input_file.suffix or '.png'
    if target_format:
        ext = "." + target_format.lower()

    subdir = Path()
    if recursive:
        try:
            rel = input_file.parent.relative_to(input_dir)
            subdir = rel
        except ValueError:
            subdir = Path()

    same_dir = output_dir.resolve() == input_dir.resolve()
    apply_suffix = not no_suffix and (not is_single_image or same_dir)
    name = f"{stem}_processed{ext}" if apply_suffix else f"{stem}{ext}"
    return output_dir / subdir / name
