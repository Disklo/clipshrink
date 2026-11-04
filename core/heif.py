import contextlib
import functools
import os
import shutil
import struct
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

from .utils import run_capture, set_last_error

# libheif is CPU-only, ffmpeg has HW HEVC but no HEIF muxer.
# encode with ffmpeg, pull hvcC from the mp4, wrap in minimal ISO-BMFF.

_HW_HEVC_CANDIDATES = {
    "darwin": ("hevc_videotoolbox",),
    "win32": ("hevc_nvenc", "hevc_qsv", "hevc_amf"),
    "cygwin": ("hevc_nvenc", "hevc_qsv", "hevc_amf"),
    "linux": ("hevc_nvenc", "hevc_qsv", "hevc_amf"),
}

_HEVC_MAX_DIM = 4096


@functools.cache
def _hw_state() -> dict[str, bool]:
    return {"disabled": False}


def _hw_disabled() -> bool:
    return _hw_state()["disabled"]


def _disable_hw() -> None:
    _hw_state()["disabled"] = True


def _reset_hw() -> None:
    _hw_state.cache_clear()


def _quality_to_qp(quality: int) -> int:
    return max(0, min(51, round(51 - quality * 51 / 100)))


def _probe_hw_hevc_encoder(encoder: str) -> bool:
    fd, out_name = tempfile.mkstemp(prefix="clipshrink_hevc_probe_", suffix=".mp4")
    os.close(fd)
    try:
        result = run_capture([
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=1",
            "-frames:v", "1", "-c:v", encoder, "-rc", "cqp",
            "-qp_i", "30", "-qp_p", "30", "-f", "mp4", out_name,
        ])
        return result.returncode == 0 and os.path.getsize(out_name) > 0
    except Exception:
        return False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(out_name)


@functools.cache
def _hw_hevc_encoder() -> str:
    if shutil.which("ffmpeg") is None:
        return ""
    for encoder in _HW_HEVC_CANDIDATES.get(sys.platform, ()):
        if _probe_hw_hevc_encoder(encoder):
            return encoder
    return ""


def _ffprobe_dims(path: Path) -> tuple[int, int] | None:
    result = run_capture([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", str(path),
    ])
    if result.returncode != 0:
        return None
    text = (result.stdout or "").strip()
    try:
        width, height = text.split("x")
        return int(width), int(height)
    except ValueError:
        return None


def _iter_boxes(data: bytes, start: int, end: int) -> Iterator[tuple[str, int, int, int]]:
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", data[pos:pos + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            break
        yield typ.decode("latin1"), pos, size, hdr
        pos += size


def _find_box(data: bytes, name: str, start: int, end: int) -> tuple[int, int] | None:
    for nm, pos, size, _hdr in _iter_boxes(data, start, end):
        if nm == name:
            return pos, size
    return None


def _find_stbl(data: bytes, start: int, end: int) -> tuple[int, int] | None:
    for nm, pos, size, hdr in _iter_boxes(data, start, end):
        if nm == "stbl":
            return pos + hdr, pos + size
        if nm in ("trak", "mdia", "minf", "moov"):
            found = _find_stbl(data, pos + hdr, pos + size)
            if found is not None:
                return found
    return None


def _find_hvcc(data: bytes, start: int, end: int) -> bytes | None:
    # sample entry children aren't box-aligned, scan for the signature
    pos = start
    while True:
        pos = data.find(b"hvcC", pos, end)
        if pos < 0:
            return None
        if pos - 4 >= start:
            size = struct.unpack(">I", data[pos - 4:pos])[0]
            if size >= 8 and pos + size <= end:
                return data[pos + 4:pos + size]
        pos += 4


def _mp4_extract(data: bytes) -> tuple[bytes | None, list[tuple[int, int]]]:
    hvcc = None
    mdat_start = mdat_end = None
    stbl = None
    for nm, pos, size, hdr in _iter_boxes(data, 0, len(data)):
        if nm == "mdat":
            mdat_start, mdat_end = pos + hdr, pos + size
        elif nm == "moov":
            # one stbl serves both the hvcC lookup and sample-table walk
            stbl = _find_stbl(data, pos + hdr, pos + size)
            if stbl is not None:
                hvcc = _find_hvcc(data, stbl[0], stbl[1])
    if hvcc is None or mdat_start is None:
        return None, []
    if stbl is None:
        return hvcc, []
    stbl_start, stbl_end = stbl

    sizes = []
    offsets = []
    stsc_entries = []
    for nm, pos, _size, _hdr in _iter_boxes(data, stbl_start, stbl_end):
        if nm == "stsz":
            default = struct.unpack(">I", data[pos + 12:pos + 16])[0]
            count = struct.unpack(">I", data[pos + 16:pos + 20])[0]
            if default:
                sizes = [default] * count
            else:
                sizes = [struct.unpack(">I", data[pos + 20 + 4 * i:pos + 24 + 4 * i])[0]
                         for i in range(count)]
        elif nm == "stco":
            count = struct.unpack(">I", data[pos + 12:pos + 16])[0]
            offsets = [struct.unpack(">I", data[pos + 16 + 4 * i:pos + 20 + 4 * i])[0]
                       for i in range(count)]
        elif nm == "co64":
            count = struct.unpack(">I", data[pos + 12:pos + 16])[0]
            offsets = [struct.unpack(">Q", data[pos + 16 + 8 * i:pos + 24 + 8 * i])[0]
                       for i in range(count)]
        elif nm == "stsc":
            count = struct.unpack(">I", data[pos + 12:pos + 16])[0]
            for i in range(count):
                base = pos + 16 + 12 * i
                stsc_entries.append((struct.unpack(">I", data[base:base + 4])[0],
                                     struct.unpack(">I", data[base + 4:base + 8])[0]))

    if not offsets or not sizes:
        return hvcc, []

    def samples_per_chunk(chunk_index: int) -> int:
        count = stsc_entries[0][1] if stsc_entries else 1
        for first, spc in stsc_entries:
            if chunk_index + 1 >= first:
                count = spc
            else:
                break
        return count

    samples = []
    sample_idx = 0
    for chunk_idx, offset in enumerate(offsets):
        for _ in range(samples_per_chunk(chunk_idx)):
            if sample_idx >= len(sizes):
                break
            size = sizes[sample_idx]
            if offset + size <= len(data) and mdat_start <= offset < mdat_end:
                samples.append((offset, size))
            offset += size
            sample_idx += 1
    return hvcc, samples


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + typ + payload


def _fullbox(typ: bytes, payload: bytes, version: int = 0, flags: int = 0) -> bytes:
    return _box(typ, bytes([version]) + flags.to_bytes(3, "big") + payload)


def _build_heic(width: int, height: int, hvcc: bytes, payload: bytes) -> bytes:
    ftyp = _box(b"ftyp", b"heix" + b"\x00\x00\x00\x00" + b"mif1heixmiaf")

    hdlr = _fullbox(b"hdlr", b"\x00\x00\x00\x00" + b"pict" + b"\x00" * 12 + b"\x00")
    pitm = _fullbox(b"pitm", b"\x00\x01")
    infe = _box(b"infe", b"\x02\x00\x00\x00" + b"\x00\x01" + b"\x00\x00"
                + b"hvc1" + b"heic\x00")
    iinf = _fullbox(b"iinf", b"\x00\x01" + infe)
    hvcc_box = _box(b"hvcC", hvcc)
    colr = _box(b"colr", b"nclx" + b"\x00\x01" + b"\x00\x0d" + b"\x00\x06" + b"\x80")
    ispe = _fullbox(b"ispe", width.to_bytes(4, "big") + height.to_bytes(4, "big"))
    pixi = _fullbox(b"pixi", b"\x03" + b"\x0a\x0a\x0a")
    ipco = _box(b"ipco", hvcc_box + colr + ispe + pixi)
    ipma = _fullbox(b"ipma", b"\x00\x00\x00\x01" + b"\x00\x01" + b"\x04"
                    + b"\x01\x02\x03\x04")
    iprp = _box(b"iprp", ipco + ipma)
    iloc = _fullbox(b"iloc", b"\x44\x40" + b"\x00\x01" + b"\x00\x01" + b"\x00\x00"
                    + b"\x00\x00\x00\x00" + b"\x00\x01" + b"\x00\x00\x00\x00"
                    + len(payload).to_bytes(4, "big"))

    meta_prefix = b"\x00\x00\x00\x00" + hdlr + iloc + iinf + pitm + iprp
    base_offset = len(ftyp) + 8 + len(meta_prefix) + 8
    iloc_fixed = _fullbox(b"iloc", b"\x44\x40" + b"\x00\x01" + b"\x00\x01" + b"\x00\x00"
                          + base_offset.to_bytes(4, "big") + b"\x00\x01"
                          + b"\x00\x00\x00\x00" + len(payload).to_bytes(4, "big"))
    meta = _box(b"meta", b"\x00\x00\x00\x00" + hdlr + iloc_fixed + iinf + pitm + iprp)

    mdat = _box(b"mdat", payload)
    return ftyp + meta + mdat


def heic_hw_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    strip_metadata: bool = False,
) -> bool:
    # false means the caller should use the CPU path
    if _hw_disabled():
        return False
    encoder = _hw_hevc_encoder()
    if not encoder:
        return False
    dims = _ffprobe_dims(input_path)
    if dims is None:
        return False
    width, height = dims
    if width <= 0 or height <= 0 or width > _HEVC_MAX_DIM or height > _HEVC_MAX_DIM:
        return False

    qp = _quality_to_qp(quality)
    fd, mp4_name = tempfile.mkstemp(prefix="clipshrink_heic_", suffix=".mp4")
    os.close(fd)
    try:
        result = run_capture([
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-i", str(input_path), "-an", "-frames:v", "1",
            "-c:v", encoder, "-rc", "cqp",
            "-qp_i", str(qp), "-qp_p", str(qp),
            "-f", "mp4", mp4_name,
        ])
        if result.returncode != 0:
            _disable_hw()
            set_last_error((result.stderr or "").strip() or (result.stdout or "").strip())
            return False
        mp4 = Path(mp4_name).read_bytes()
        hvcc, samples = _mp4_extract(mp4)
        if not hvcc or not samples:
            _disable_hw()
            set_last_error("Failed to parse ffmpeg HEVC output.")
            return False
        offset, size = samples[0]
        payload = mp4[offset:offset + size]
        if not payload:
            set_last_error("Hardware HEVC encode produced no bitstream.")
            return False
        # hardware encoders round odd dims to even for 4:2:0. signal the
        # decoded size so ispe matches what libheif decodes.
        coded = _ffprobe_dims(Path(mp4_name))
        if coded is not None:
            width, height = coded
        output_path.write_bytes(_build_heic(width, height, hvcc, payload))
        # validate the muxed HEIC, the hand-rolled container is the only
        # untested step. sticky disable stops retries in this process.
        if (output_path.stat().st_size == 0
                or _ffprobe_dims(output_path) is None):
            _disable_hw()
            set_last_error("Hardware HEIF muxer produced an unreadable file.")
            with contextlib.suppress(OSError):
                output_path.unlink()
            return False
        return True
    except FileNotFoundError:
        return False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(mp4_name)
