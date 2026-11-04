from core import fileops


def test_is_image_mime():
    assert fileops.is_image_mime("image/png")
    assert fileops.is_image_mime("image/jpeg")
    assert not fileops.is_image_mime("text/plain")
    assert not fileops.is_image_mime("application/pdf")


def test_mime_to_ext_mapping():
    assert fileops.MIME_TO_EXT["image/png"] == ".png"
    assert fileops.MIME_TO_EXT["image/jpeg"] == ".jpg"
    assert fileops.MIME_TO_EXT["image/webp"] == ".webp"


def test_is_image_file_accepts_known_extensions(tmp_path):
    for name in ("photo.png", "photo.JPG", "photo.webp"):
        f = tmp_path / name
        f.write_bytes(b"placeholder")
        assert fileops.is_image_file(f)


def test_is_image_file_rejects_known_non_image_extensions(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    assert not fileops.is_image_file(f)


def test_is_image_file_short_circuits_known_non_image_extensions(tmp_path):
    for name in ("book.pdf", "song.mp3", "video.mp4", "archive.zip"):
        f = tmp_path / name
        f.write_bytes(b"placeholder")
        assert not fileops.is_image_file(f)


def test_is_image_file_fallback_to_mime(tmp_path):
    f = tmp_path / "extensionless"
    f.write_bytes(b"not really an image")
    result = fileops.is_image_file(f)
    assert result is False


def test_is_image_file_sniffs_when_file_cmd_missing(tmp_path, monkeypatch):
    def no_file_cmd(*args, **kwargs):
        raise FileNotFoundError("file")

    monkeypatch.setattr(fileops.subprocess, "run", no_file_cmd)
    signatures = {
        "mystery1": b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
        "mystery2": b"\xff\xd8\xff\xe0" + b"\x00" * 16,
        "mystery3": b"GIF89a" + b"\x00" * 16,
        "mystery4": b"BM" + b"\x00" * 24,
        "mystery5": b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8,
        "mystery6": b"II*\x00" + b"\x00" * 16,
        "mystery7": b"\xff\x0a" + b"\x00" * 16,
        "mystery8": b"\x00\x00\x00\x0cjP  \r\n\x87\n" + b"\x00" * 8,
        "mystery9": b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00",
        "mystery10": b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00",
    }
    for name, magic in signatures.items():
        f = tmp_path / name
        f.write_bytes(magic)
        assert fileops.is_image_file(f), f"should sniff {name} as image"


def test_is_image_file_sniff_rejects_non_image(tmp_path, monkeypatch):
    def no_file_cmd(*args, **kwargs):
        raise FileNotFoundError("file")

    monkeypatch.setattr(fileops.subprocess, "run", no_file_cmd)
    f = tmp_path / "mystery"
    f.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")
    assert not fileops.is_image_file(f)
    f2 = tmp_path / "mystery2"
    f2.write_bytes(b"just some text")
    assert not fileops.is_image_file(f2)


def test_is_image_file_sniff_unreadable_file_is_false(tmp_path, monkeypatch):
    def no_file_cmd(*args, **kwargs):
        raise FileNotFoundError("file")

    monkeypatch.setattr(fileops.subprocess, "run", no_file_cmd)
    f = tmp_path / "a_directory"
    f.mkdir()
    assert not fileops.is_image_file(f)


def test_is_image_file_sniff_not_consulted_when_file_cmd_works(tmp_path,
                                                                monkeypatch):
    def fake_run(cmd, **kwargs):
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, b"text/plain\n", b"")

    monkeypatch.setattr(fileops.subprocess, "run", fake_run)
    f = tmp_path / "mystery"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    assert not fileops.is_image_file(f)


def _setup(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    img = input_dir / "photo.png"
    img.write_bytes(b"x")
    return img, input_dir, out_dir


def test_compute_output_name_batch_adds_suffix(tmp_path):
    img, input_dir, out_dir = _setup(tmp_path)
    result = fileops.compute_output_name(img, out_dir, input_dir, False, False, False, None)
    assert result == out_dir / "photo_processed.png"


def test_compute_output_name_single_image_no_suffix(tmp_path):
    img, input_dir, out_dir = _setup(tmp_path)
    result = fileops.compute_output_name(img, out_dir, input_dir, False, False, True, None)
    assert result == out_dir / "photo.png"


def test_compute_output_name_no_suffix_flag(tmp_path):
    img, input_dir, out_dir = _setup(tmp_path)
    result = fileops.compute_output_name(img, out_dir, input_dir, False, True, False, None)
    assert result == out_dir / "photo.png"


def test_compute_output_name_same_dir_forces_suffix(tmp_path):
    img, input_dir, _ = _setup(tmp_path)
    result = fileops.compute_output_name(img, input_dir, input_dir, False, False, True, None)
    assert result == input_dir / "photo_processed.png"


def test_compute_output_name_recursive_mirrors_subdir(tmp_path):
    _img, input_dir, out_dir = _setup(tmp_path)
    sub = input_dir / "sub" / "deep"
    sub.mkdir(parents=True)
    nested = sub / "art.jpg"
    nested.write_bytes(b"x")
    result = fileops.compute_output_name(nested, out_dir, input_dir, True, False, False, None)
    assert result == out_dir / "sub" / "deep" / "art_processed.jpg"


def test_compute_output_name_target_format_extension(tmp_path):
    img, input_dir, out_dir = _setup(tmp_path)
    result = fileops.compute_output_name(img, out_dir, input_dir, False, False, True, "JPEG")
    assert result == out_dir / "photo.jpeg"


def test_compute_output_name_extensionless_defaults_to_png(tmp_path):
    _img, input_dir, out_dir = _setup(tmp_path)
    no_ext = input_dir / "photo"
    no_ext.write_bytes(b"x")
    result = fileops.compute_output_name(no_ext, out_dir, input_dir, False, False, True, None)
    assert result == out_dir / "photo.png"
