from pathlib import Path

import pytest

from core import memgate, utils

GiB = 1024 ** 3
MiB = 1024 ** 2


@pytest.mark.parametrize(("text", "expected"), [
    ("536870912", 536870912),
    ("512m", 512 * MiB),
    ("512M", 512 * MiB),
    ("2g", 2 * GiB),
    ("1k", 1024),
    (" 12 g ", 12 * GiB),
])
def test_parse_size_valid(text, expected):
    assert memgate.parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "12x", "1.5g", "-5m", "m"])
def test_parse_size_invalid_raises(text):
    with pytest.raises(ValueError):
        memgate.parse_size(text)


def test_budget_bytes_scales_available(monkeypatch):
    assert memgate.budget_bytes(10 * GiB) == int(10 * GiB * 0.7)


def test_budget_bytes_floor(monkeypatch):
    assert memgate.budget_bytes(10 * MiB) == 512 * MiB


def test_budget_bytes_override(monkeypatch):
    monkeypatch.setenv("CLIPSHRINK_MAX_RAM", "2g")
    assert memgate.budget_bytes(100 * GiB) == 2 * GiB


def test_budget_bytes_override_beats_measurement(monkeypatch):
    monkeypatch.setenv("CLIPSHRINK_MAX_RAM", "512m")
    assert memgate.budget_bytes(10 * GiB) == 512 * MiB


def test_budget_bytes_junk_override_falls_back(monkeypatch):
    monkeypatch.setenv("CLIPSHRINK_MAX_RAM", "banana")
    assert memgate.budget_bytes(10 * GiB) == int(10 * GiB * 0.7)


def test_budget_bytes_unknown_platform(monkeypatch):
    monkeypatch.delenv("CLIPSHRINK_MAX_RAM", raising=False)
    monkeypatch.setattr(memgate, "_mem_available", lambda: None)
    assert memgate.budget_bytes() is None


def test_cap_returns_requested_without_budget(monkeypatch):
    monkeypatch.setattr(memgate, "budget_bytes", lambda: None)
    assert memgate.probe_worker_cap(30000, 21952, 4) == 4


def test_cap_serializes_giant_image(monkeypatch):
    monkeypatch.setattr(memgate, "budget_bytes", lambda: 8 * GiB)
    assert memgate.probe_worker_cap(30000, 21952, 4) == 1


def test_cap_fits_small_images_fully(monkeypatch):
    monkeypatch.setattr(memgate, "budget_bytes", lambda: 8 * GiB)
    assert memgate.probe_worker_cap(5000, 4000, 4) == 4


def test_cap_partial_concurrency(monkeypatch):
    monkeypatch.setattr(memgate, "budget_bytes", lambda: 6 * GiB)
    assert memgate.probe_worker_cap(10000, 10000, 4) == 3


def test_cap_never_exceeds_requested(monkeypatch):
    monkeypatch.setattr(memgate, "budget_bytes", lambda: 64 * GiB)
    assert memgate.probe_worker_cap(100, 100, 2) == 2


def test_cap_never_below_one(monkeypatch):
    monkeypatch.setattr(memgate, "budget_bytes", lambda: 512 * MiB)
    assert memgate.probe_worker_cap(30000, 21952, 4) == 1


def test_cap_requested_one_stays_one():
    assert memgate.probe_worker_cap(30000, 21952, 1) == 1


def test_cap_degenerate_dimensions_return_requested(monkeypatch):
    monkeypatch.setattr(memgate, "budget_bytes", lambda: 8 * GiB)
    assert memgate.probe_worker_cap(0, 100, 4) == 4
    assert memgate.probe_worker_cap(100, 0, 4) == 4


def test_is_tmpfs_true_for_tmp_mount(monkeypatch, tmp_path):
    monkeypatch.setattr(
        memgate, "_mounts_text",
        lambda: "tmpfs /tmp tmpfs rw,nosuid 0 0\n/dev/sda2 /home ext4 rw 0 0\n")
    memgate.is_tmpfs.cache_clear()
    try:
        assert memgate.is_tmpfs("/tmp/clipshrink_probe_x.png") is True
        assert memgate.is_tmpfs("/home/user/file.png") is False
    finally:
        memgate.is_tmpfs.cache_clear()


def test_is_tmpfs_false_without_proc_mounts(monkeypatch, tmp_path):
    monkeypatch.setattr(memgate, "_mounts_text", lambda: "")
    memgate.is_tmpfs.cache_clear()
    try:
        assert memgate.is_tmpfs(str(tmp_path)) is False
    finally:
        memgate.is_tmpfs.cache_clear()


def test_is_tmpfs_longest_mount_point_wins(monkeypatch):
    monkeypatch.setattr(
        memgate, "_mounts_text",
        lambda: "tmpfs / tmpfs rw 0 0\n/dev/sda1 /home/disklo ext4 rw 0 0\n")
    memgate.is_tmpfs.cache_clear()
    try:
        assert memgate.is_tmpfs("/home/disklo/x.png") is False
        assert memgate.is_tmpfs("/etc/fstab") is True
    finally:
        memgate.is_tmpfs.cache_clear()


def test_spill_dir_none_when_temp_is_real_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(memgate, "is_tmpfs", lambda path: False)
    assert memgate.magick_spill_dir(str(tmp_path)) is None


def test_spill_dir_prefers_candidate_on_real_disk(monkeypatch, tmp_path):
    real = tmp_path / "realdisk"
    real.mkdir()
    monkeypatch.setattr(memgate, "is_tmpfs",
                        lambda path: not str(path).startswith(str(real)))
    monkeypatch.setattr(memgate, "_spill_candidates", lambda: [str(real)])
    assert memgate.magick_spill_dir("/tmp/whatever") == str(real)


def test_spill_dir_skips_tmpfs_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(memgate, "is_tmpfs", lambda path: True)
    monkeypatch.setattr(memgate, "_spill_candidates",
                        lambda: [str(tmp_path / "missing"), "/var/tmp"])
    assert memgate.magick_spill_dir("/tmp/whatever") is None


def _jpeg_probe_cmd() -> list[str]:
    return ["magick", "in.png", "-quality", "90", "-sampling-factor", "4:2:0",
            "-define", "jpeg:optimize-coding=true", "-interlace", "Plane",
            "out.jpg"]


def test_env_caps_jpeg_probe(monkeypatch):
    monkeypatch.setattr(memgate, "magick_spill_dir", lambda temp_dir=None: None)
    env = memgate.magick_env(_jpeg_probe_cmd())
    assert env["MAGICK_MEMORY_LIMIT"] == "2GiB"
    assert env["MAGICK_MAP_LIMIT"] == "2GiB"
    assert env["MAGICK_DISK_LIMIT"] == "64GiB"


def test_env_caps_extent_and_identify(monkeypatch):
    monkeypatch.setattr(memgate, "magick_spill_dir", lambda temp_dir=None: None)
    extent = memgate.magick_env(
        ["magick", "in.png", "-define", "jpeg:extent=1000000", "out.jpg"])
    assert extent["MAGICK_MEMORY_LIMIT"] == "2GiB"
    identify = memgate.magick_env(["magick", "identify", "-format", "%w", "in.png"])
    assert identify["MAGICK_MEMORY_LIMIT"] == "2GiB"


def test_env_no_caps_for_geometry_ops(monkeypatch):
    monkeypatch.setattr(memgate, "magick_spill_dir", lambda temp_dir=None: None)
    env = memgate.magick_env(
        ["magick", "in.png", "-resize", "100x100!", "-depth", "8", "out.png"])
    assert "MAGICK_MEMORY_LIMIT" not in env
    assert "MAGICK_MAP_LIMIT" not in env


def test_env_no_caps_for_plain_reencode(monkeypatch):
    monkeypatch.setattr(memgate, "magick_spill_dir", lambda temp_dir=None: None)
    env = memgate.magick_env(["magick", "in.png", "out.png"])
    assert "MAGICK_MEMORY_LIMIT" not in env


def test_env_opt_in_caps_everything(monkeypatch):
    monkeypatch.setenv("CLIPSHRINK_MAGICK_LIMITS", "1")
    monkeypatch.setattr(memgate, "magick_spill_dir", lambda temp_dir=None: None)
    env = memgate.magick_env(
        ["magick", "in.png", "-resize", "100x100!", "out.png"])
    assert env["MAGICK_MEMORY_LIMIT"] == "2GiB"


@pytest.mark.parametrize("value", ["true", "yes"])
def test_env_opt_in_truthy_values(monkeypatch, value):
    monkeypatch.setenv("CLIPSHRINK_MAGICK_LIMITS", value)
    monkeypatch.setattr(memgate, "magick_spill_dir", lambda temp_dir=None: None)
    env = memgate.magick_env(["magick", "in.png", "-resize", "10x10!", "o.png"])
    assert "MAGICK_MEMORY_LIMIT" in env


def test_env_opt_in_off_by_default(monkeypatch):
    monkeypatch.delenv("CLIPSHRINK_MAGICK_LIMITS", raising=False)
    monkeypatch.setattr(memgate, "magick_spill_dir", lambda temp_dir=None: None)
    env = memgate.magick_env(["magick", "in.png", "-resize", "10x10!", "o.png"])
    assert "MAGICK_MEMORY_LIMIT" not in env


def test_env_routes_spill_off_tmpfs(monkeypatch, tmp_path):
    monkeypatch.setattr(memgate, "magick_spill_dir",
                        lambda temp_dir=None: str(tmp_path))
    env = memgate.magick_env(["magick", "in.png", "out.png"])
    assert env["MAGICK_TMPDIR"] == str(tmp_path)


def test_run_capture_injects_magick_env(monkeypatch):
    monkeypatch.setattr(memgate, "magick_env",
                        lambda cmd: {"MAGICK_MEMORY_LIMIT": "2GiB"})
    kwargs = utils._magick_kwargs(["magick", "in.png", "out.jpg"], {})
    assert kwargs["env"]["MAGICK_MEMORY_LIMIT"] == "2GiB"


def test_run_capture_preserves_caller_env(monkeypatch):
    monkeypatch.setattr(memgate, "magick_env",
                        lambda cmd: {"MAGICK_MEMORY_LIMIT": "2GiB"})
    kwargs = utils._magick_kwargs(
        ["magick", "in.png", "out.jpg"], {"env": {"MAGICK_MEMORY_LIMIT": "8GiB"}})
    assert kwargs["env"]["MAGICK_MEMORY_LIMIT"] == "8GiB"


def test_run_capture_untouched_for_non_magick(monkeypatch):
    monkeypatch.setattr(memgate, "magick_env",
                        lambda cmd: {"MAGICK_MEMORY_LIMIT": "2GiB"})
    kwargs = utils._magick_kwargs(["ffmpeg", "-i", "a.mp4"], {})
    assert "env" not in kwargs


def test_parallel_fit_search_cap_neutral():
    from core import search

    def probe_fn(q):
        return (q <= 60, q * 1000, Path(f"/tmp/probe{q}.jpg"))

    best_cap, exec_cap = search._parallel_fit_search(probe_fn, 1, 100, worker_cap=1)
    best_full, exec_full = search._parallel_fit_search(probe_fn, 1, 100)
    assert best_cap == best_full == (60, 60000, Path("/tmp/probe60.jpg"))
    assert exec_cap == exec_full


def test_grid_search_cap_neutral():
    from core import search

    def probe_fn(q):
        return (q <= 55, q * 1000, Path(f"/tmp/probe{q}.jpg"))

    best_cap, exec_cap = search._coarse_grid_search(probe_fn, 1, 100, step=10,
                                                    worker_cap=1)
    best_full, exec_full = search._coarse_grid_search(probe_fn, 1, 100, step=10)
    assert best_cap == best_full
    assert exec_cap == exec_full


def test_simple_search_round_applies_worker_cap(monkeypatch, tmp_path):
    from core import search
    current = tmp_path / "cur.png"
    current.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(search.converters, "get_image_width", lambda p: 640)
    monkeypatch.setattr(search.converters, "get_image_height", lambda p: 480)
    monkeypatch.setattr(search.memgate, "probe_worker_cap",
                        lambda w, h, r: calls.append((w, h, r)) or 1)
    monkeypatch.setattr(search.converters, "get_image_filesize", lambda p: 50)
    round_ = search._simple_search_round(
        probe_encode=lambda src, out, k: True, probe_suffix=".jpg",
        probe_message="m {}", search_step=None, max_size_bytes=100,
        round_collect_candidates=False, round_frontier_ceiling=0)
    result = round_(current, [], 1, 10)
    assert calls == [(640, 480, search._MAX_PARALLEL)]
    assert result.fitted is True


def test_simple_search_round_cap_failure_is_neutral(monkeypatch, tmp_path):
    from core import search
    current = tmp_path / "cur.png"
    current.write_bytes(b"x")

    def boom(p):
        raise RuntimeError("no backend")

    monkeypatch.setattr(search.converters, "get_image_width", boom)
    monkeypatch.setattr(search.converters, "get_image_filesize", lambda p: 50)
    round_ = search._simple_search_round(
        probe_encode=lambda src, out, k: True, probe_suffix=".jpg",
        probe_message="m {}", search_step=None, max_size_bytes=100,
        round_collect_candidates=False, round_frontier_ceiling=0)
    result = round_(current, [], 1, 10)
    assert result.fitted is True


def test_frontier_round_applies_worker_cap(monkeypatch, tmp_path):
    from core import search
    from core.encoders import CodecEncoder
    calls = []
    monkeypatch.setattr(search.converters, "get_image_width", lambda p: 10)
    monkeypatch.setattr(search.converters, "get_image_height", lambda p: 10)
    monkeypatch.setattr(search.memgate, "probe_worker_cap",
                        lambda w, h, r: calls.append((w, h, r)) or 1)
    monkeypatch.setattr(search.converters, "get_image_filesize", lambda p: 50)
    enc = CodecEncoder(
        "jpeg",
        encode=lambda i, o, q, subsampling="4:2:0", strip_metadata=False: True,
        suffix=".jpg", subsampling_supported=True)
    round_ = search._frontier_lossy_round(
        [enc], "4:2:0", False, 100, round_frontier_ceiling=90)
    result = round_(tmp_path / "cur.png", [], 1, 100)
    assert calls == [(10, 10, search._MAX_PARALLEL)]
    assert result.fitted is True


def test_tempdir_none_on_real_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(memgate, "is_tmpfs", lambda path: False)
    memgate.tempdir.cache_clear()
    try:
        assert memgate.tempdir() is None
    finally:
        memgate.tempdir.cache_clear()


def test_tempdir_routes_off_tmpfs(monkeypatch, tmp_path):
    real = tmp_path / "realdisk"
    real.mkdir()
    memgate.is_tmpfs.cache_clear()
    monkeypatch.setattr(memgate, "is_tmpfs",
                        lambda path: not str(path).startswith(str(real)))
    monkeypatch.setattr(memgate, "_spill_candidates", lambda: [str(real)])
    memgate.tempdir.cache_clear()
    try:
        assert memgate.tempdir() == str(real)
    finally:
        memgate.tempdir.cache_clear()


def test_tempdir_survives_spill_dir_errors(monkeypatch):
    def boom(temp_dir=None):
        raise OSError("proc gone")
    monkeypatch.setattr(memgate, "magick_spill_dir", boom)
    memgate.tempdir.cache_clear()
    try:
        assert memgate.tempdir() is None
    finally:
        memgate.tempdir.cache_clear()


def test_pipeline_temp_uses_memgate_dir(monkeypatch, tmp_path):
    from core import pipeline
    monkeypatch.setattr(memgate, "tempdir", lambda: str(tmp_path))
    created = []
    real_mkstemp = pipeline.tempfile.mkstemp

    def fake_mkstemp(*, prefix, suffix, dir):
        created.append(dir)
        return real_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    monkeypatch.setattr(pipeline.tempfile, "mkstemp", fake_mkstemp)
    path = pipeline._make_temp(".png")
    path.unlink()
    assert created == [str(tmp_path)]


def test_search_probe_uses_memgate_dir(monkeypatch, tmp_path):
    from core import search
    monkeypatch.setattr(memgate, "tempdir", lambda: str(tmp_path))
    probe = search._new_probe(".png", [])
    try:
        assert probe.parent == tmp_path
    finally:
        probe.unlink(missing_ok=True)


def test_magick_error_synthesizes_signal_death(monkeypatch):
    import subprocess

    from core import converters
    monkeypatch.setattr(converters, "run_capture", lambda cmd: subprocess.CompletedProcess(cmd, -9, "", ""))
    converters._run_magick(["magick", "in.png", "out.png"])
    assert "signal 9" in converters.get_last_error()


def test_magick_error_synthesizes_exit_code(monkeypatch):
    import subprocess

    from core import converters
    monkeypatch.setattr(converters, "run_capture", lambda cmd: subprocess.CompletedProcess(cmd, 1, "", ""))
    converters._run_magick(["magick", "in.png", "out.png"])
    assert "exited with code 1" in converters.get_last_error()


def test_magick_error_prefers_stderr(monkeypatch):
    import subprocess

    from core import converters
    monkeypatch.setattr(converters, "run_capture",
                        lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    converters._run_magick(["magick", "in.png", "out.png"])
    assert converters.get_last_error() == "boom"


def test_apply_process_tempdir_redirects_off_tmpfs(monkeypatch, tmp_path):
    real = tmp_path / "realdisk"
    real.mkdir()
    memgate.tempdir.cache_clear()
    monkeypatch.setattr(memgate, "is_tmpfs",
                        lambda path: not str(path).startswith(str(real)))
    monkeypatch.setattr(memgate, "tempdir", lambda: str(real))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "old"))
    memgate.apply_process_tempdir()
    import os
    assert os.environ["TMPDIR"] == str(real)


def test_apply_process_tempdir_noop_on_real_disk(monkeypatch, tmp_path):
    memgate.tempdir.cache_clear()
    monkeypatch.setattr(memgate, "is_tmpfs", lambda path: False)
    monkeypatch.setattr(memgate, "tempdir", lambda: None)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    memgate.apply_process_tempdir()
    import os
    assert os.environ["TMPDIR"] == str(tmp_path)
