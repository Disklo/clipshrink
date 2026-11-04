import struct
import subprocess
from pathlib import Path

import pytest

from core import heif


def _ok_result():
    return subprocess.CompletedProcess([], 0, "", "")


def _mk_mp4(payload: bytes) -> bytes:
    ftyp = heif._box(b"ftyp", b"isom" + b"\x00\x00\x00\x00" + b"isomiso2mp41")
    mdat = heif._box(b"mdat", payload)
    payload_offset = len(ftyp) + len(mdat) - len(payload)

    hvcc = heif._box(b"hvcC", b"\x01\x01\x60\x00" + b"\x00" * 8)
    visual = b"\x00" * 6 + b"\x00\x01" + b"\x00" * 78
    hev1 = heif._box(b"hev1", visual + hvcc)
    stsd = heif._fullbox(b"stsd", b"\x00\x00\x00\x01" + hev1)
    stts = heif._fullbox(b"stts", b"\x00\x00\x00\x01" + b"\x00\x00\x00\x01" + b"\x00\x00\x00\x01")
    stsc = heif._fullbox(b"stsc", b"\x00\x00\x00\x01" + b"\x00\x00\x00\x01" + b"\x00\x00\x00\x01" + b"\x00\x00\x00\x01")
    stsz = heif._fullbox(b"stsz", len(payload).to_bytes(4, "big") + b"\x00\x00\x00\x01")
    stco = heif._fullbox(b"stco", b"\x00\x00\x00\x01" + payload_offset.to_bytes(4, "big"))
    stbl = heif._box(b"stbl", stsd + stts + stsc + stsz + stco)
    minf = heif._box(b"minf", stbl)
    mdia = heif._box(b"mdia", minf)
    trak = heif._box(b"trak", mdia)
    moov = heif._box(b"moov", trak)
    return ftyp + mdat + moov


def _walk(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", data[pos:pos + 8])
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - pos
            hdr = 8
        else:
            hdr = 8
        if size < hdr or pos + size > end:
            break
        yield typ.decode("latin1"), pos, size, hdr
        pos += size


def _find_all(data: bytes, name: str):
    return [(pos, size) for nm, pos, size, hdr in _walk(data, 0, len(data)) if nm == name]


def test_mp4_extract_gets_hvcc_and_sample():
    payload = b"\xde\xad\xbe\xef" * 25
    mp4 = _mk_mp4(payload)
    hvcc, samples = heif._mp4_extract(mp4)
    assert hvcc is not None
    assert hvcc[:4] == b"\x01\x01\x60\x00"
    assert len(samples) == 1
    offset, size = samples[0]
    assert mp4[offset:offset + size] == payload


def test_mp4_extract_returns_empty_when_no_hvcc():
    mp4 = _mk_mp4(b"x" * 10)
    no_hvcc = mp4.replace(b"hvcC", b"avcC")
    hvcc, samples = heif._mp4_extract(no_hvcc)
    assert hvcc is None
    assert samples == []


def test_mp4_extract_stsz_default_and_explicit():
    payload = b"y" * 40
    ftyp = heif._box(b"ftyp", b"isom" + b"\x00" * 12)
    mdat = heif._box(b"mdat", payload)
    offset = len(ftyp) + len(mdat) - len(payload)
    hvcc_box = heif._box(b"hvcC", b"\x01\x01\x60\x00\x00" * 2)
    stsd = heif._fullbox(b"stsd", b"\x00\x00\x00\x01" + heif._box(b"hev1", b"\x00" * 86 + hvcc_box))
    stsc = heif._fullbox(b"stsc", b"\x00\x00\x00\x01" + b"\x00\x00\x00\x01" + b"\x00\x00\x00\x02" + b"\x00\x00\x00\x01")
    stsz = heif._fullbox(b"stsz", b"\x00\x00\x00\x00" + b"\x00\x00\x00\x02" + b"\x00\x00\x00\x10" + b"\x00\x00\x00\x18")
    stco = heif._fullbox(b"stco", b"\x00\x00\x00\x01" + offset.to_bytes(4, "big"))
    stbl = heif._box(b"stbl", stsd + stsc + stsz + stco)
    minf = heif._box(b"minf", stbl)
    mdia = heif._box(b"mdia", minf)
    trak = heif._box(b"trak", mdia)
    moov = heif._box(b"moov", trak)
    mp4 = ftyp + mdat + moov

    hvcc, samples = heif._mp4_extract(mp4)
    assert hvcc is not None
    assert samples == [(offset, 16), (offset + 16, 24)]


def test_build_heic_layout():
    hvcc = b"\x01\x01\x60\x00" + b"\x00" * 12
    payload = b"z" * 1234
    heic_bytes = heif._build_heic(640, 480, hvcc, payload)

    ftyps = [heic_bytes[pos + 8:pos + size] for nm, pos, size, hdr in _walk(heic_bytes, 0, len(heic_bytes))
             if nm == "ftyp"]
    ftyp_size = len(ftyps[0]) + 8
    assert ftyps[0][:4] == b"heix"
    assert ftyps[0][8:12] == b"mif1"

    meta_boxes = [(pos, size) for nm, pos, size, hdr in _walk(heic_bytes, 0, len(heic_bytes))
                  if nm == "meta"]
    assert len(meta_boxes) == 1
    meta_pos, meta_size = meta_boxes[0]
    meta_children = [nm for nm, p, s, h in _walk(heic_bytes, meta_pos + 12, meta_pos + meta_size)]
    assert meta_children == ["hdlr", "iloc", "iinf", "pitm", "iprp"]

    for nm, pos, _size, _hdr in _walk(heic_bytes, meta_pos + 12, meta_pos + meta_size):
        if nm == "iloc":
            base = struct.unpack(">I", heic_bytes[pos + 20:pos + 24])[0]
            assert base == ftyp_size + meta_size + 8
            assert heic_bytes[base:base + len(payload)] == payload

    iprp_boxes = [(pos, size) for nm, pos, size, hdr in _walk(heic_bytes, meta_pos + 12, meta_pos + meta_size)
                  if nm == "iprp"]
    iprp_pos, iprp_size = iprp_boxes[0]
    ipco = [heic_bytes[p + 8:p + s] for nm, p, s, hdr in _walk(heic_bytes, iprp_pos + 8, iprp_pos + iprp_size)
            if nm == "ipco"]
    assert ipco and ipco[0].find(b"hvcC") >= 0
    assert ipco[0].find(b"ispe") >= 0

    mdat_boxes = [(pos, size) for nm, pos, size, hdr in _walk(heic_bytes, 0, len(heic_bytes))
                  if nm == "mdat"]
    assert len(mdat_boxes) == 1
    mdat_pos, mdat_size = mdat_boxes[0]
    assert heic_bytes[mdat_pos + 8:mdat_pos + mdat_size] == payload


def test_quality_to_qp_is_monotone():
    qps = [heif._quality_to_qp(q) for q in (100, 90, 80, 50, 20, 1)]
    assert qps == sorted(qps)
    assert heif._quality_to_qp(100) == 0
    assert heif._quality_to_qp(1) >= 49
    assert 0 <= heif._quality_to_qp(80) <= 51


def test_heic_hw_save_end_to_end(monkeypatch, tmp_path):
    payload = b"\x01\x23\x45\x67" * 50
    mp4_fixture = _mk_mp4(payload)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "ffmpeg" and "-f" in cmd:
            Path(cmd[-1]).write_bytes(mp4_fixture)
        return _ok_result()

    monkeypatch.setattr(heif, "_hw_hevc_encoder", lambda: "hevc_amf")
    monkeypatch.setattr(heif, "_ffprobe_dims", lambda p: (640, 480))
    monkeypatch.setattr(heif, "run_capture", fake_run)

    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    out = tmp_path / "out.heic"
    assert heif.heic_hw_save(src, out, 80) is True
    assert out.stat().st_size > 0

    encode_cmd = calls[0]
    assert encode_cmd[0] == "ffmpeg"
    assert encode_cmd[encode_cmd.index("-c:v") + 1] == "hevc_amf"
    assert encode_cmd[encode_cmd.index("-qp_i") + 1] == str(heif._quality_to_qp(80))

    heic_bytes = out.read_bytes()
    meta_boxes = [(pos, size) for nm, pos, size, hdr in _walk(heic_bytes, 0, len(heic_bytes))
                  if nm == "meta"]
    meta_pos, meta_size = meta_boxes[0]
    for nm, pos, _size, _hdr in _walk(heic_bytes, meta_pos + 12, meta_pos + meta_size):
        if nm == "iloc":
            base = struct.unpack(">I", heic_bytes[pos + 20:pos + 24])[0]
            assert heic_bytes[base:base + len(payload)] == payload
        if nm == "ispe":
            w, h = struct.unpack(">II", heic_bytes[pos + 12:pos + 20])
            assert (w, h) == (640, 480)


def test_heic_hw_save_signals_decoded_dims_when_cropped(monkeypatch, tmp_path):
    payload = b"\x01\x23\x45\x67" * 50
    mp4_fixture = _mk_mp4(payload)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg" and "-f" in cmd:
            Path(cmd[-1]).write_bytes(mp4_fixture)
        return _ok_result()

    def dims(path):
        return (1890, 1342) if str(path).endswith(".mp4") else (1890, 1341)

    monkeypatch.setattr(heif, "_hw_hevc_encoder", lambda: "hevc_amf")
    monkeypatch.setattr(heif, "_ffprobe_dims", dims)
    monkeypatch.setattr(heif, "run_capture", fake_run)

    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    out = tmp_path / "out.heic"
    assert heif.heic_hw_save(src, out, 80) is True

    heic_bytes = out.read_bytes()
    meta_boxes = [(pos, size) for nm, pos, size, hdr in _walk(heic_bytes, 0, len(heic_bytes))
                  if nm == "meta"]
    meta_pos, meta_size = meta_boxes[0]
    for nm, pos, _size, _hdr in _walk(heic_bytes, meta_pos + 12, meta_pos + meta_size):
        if nm == "ispe":
            w, h = struct.unpack(">II", heic_bytes[pos + 12:pos + 20])
            assert (w, h) == (1890, 1342)


def test_heic_hw_save_returns_false_without_encoder(monkeypatch, tmp_path):
    monkeypatch.setattr(heif, "_hw_hevc_encoder", lambda: "")
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    assert heif.heic_hw_save(src, tmp_path / "out.heic", 80) is False


def test_heic_hw_save_skips_oversized_input(monkeypatch, tmp_path):
    monkeypatch.setattr(heif, "_hw_hevc_encoder", lambda: "hevc_amf")
    monkeypatch.setattr(heif, "_ffprobe_dims", lambda p: (5000, 3000))
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    assert heif.heic_hw_save(src, tmp_path / "out.heic", 80) is False


@pytest.fixture(autouse=True)
def _reset_hw_latch():
    heif._reset_hw()
    yield
    heif._reset_hw()


def test_heic_hw_save_disables_after_encode_failure(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "amf failure")

    monkeypatch.setattr(heif, "_hw_hevc_encoder", lambda: "hevc_amf")
    monkeypatch.setattr(heif, "_ffprobe_dims", lambda p: (640, 480))
    monkeypatch.setattr(heif, "run_capture", fake_run)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    assert heif.heic_hw_save(src, tmp_path / "out.heic", 80) is False
    assert heif._hw_disabled() is True
    assert heif.heic_hw_save(src, tmp_path / "out2.heic", 80) is False
    assert len(calls) == 1


def test_heic_hw_save_disables_when_mp4_unparseable(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"not an mp4 at all")
        return _ok_result()

    monkeypatch.setattr(heif, "_hw_hevc_encoder", lambda: "hevc_amf")
    monkeypatch.setattr(heif, "_ffprobe_dims", lambda p: (640, 480))
    monkeypatch.setattr(heif, "run_capture", fake_run)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    assert heif.heic_hw_save(src, tmp_path / "out.heic", 80) is False
    assert heif._hw_disabled() is True


def test_heic_hw_save_validates_muxed_output(monkeypatch, tmp_path):
    payload = b"\x01\x23\x45\x67" * 50
    mp4_fixture = _mk_mp4(payload)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg" and "-f" in cmd:
            Path(cmd[-1]).write_bytes(mp4_fixture)
        return _ok_result()

    def dims(path):
        return None if str(path).endswith(".heic") else (640, 480)

    monkeypatch.setattr(heif, "_hw_hevc_encoder", lambda: "hevc_amf")
    monkeypatch.setattr(heif, "_ffprobe_dims", dims)
    monkeypatch.setattr(heif, "run_capture", fake_run)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    out = tmp_path / "out.heic"
    assert heif.heic_hw_save(src, out, 80) is False
    assert heif._hw_disabled() is True
    assert not out.exists()
