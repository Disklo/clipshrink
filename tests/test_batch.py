import os
import threading
import time
from argparse import Namespace
from pathlib import Path

import clipshrink
from core import utils


def _make_args(tmp_path, **overrides):
    ns = Namespace(
        input=str(tmp_path / "in"),
        recursive=False,
        output=str(tmp_path / "out"),
        no_suffix=False,
        format=None,
        jobs=None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _fake_images(tmp_path, count):
    in_dir = tmp_path / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(count):
        img = in_dir / f"img{i}.png"
        img.write_bytes(b"fake")
        images.append(img)
    return images


def _make_fake_process_image(images):
    processed = []
    max_concurrent = 0
    active = 0
    lock = threading.Lock()

    def fake_process_image(img, args):
        nonlocal active, max_concurrent
        with lock:
            active += 1
            max_concurrent = max(max_concurrent, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        processed.append((img.name, Path(args.output).name if args.output else None))
        return {"success": True, "final_size": 1, "bypassed": False}

    return fake_process_image, processed, lambda: max_concurrent


def test_folder_mode_parallel_processes_all_images(tmp_path, monkeypatch):
    images = _fake_images(tmp_path, 6)
    monkeypatch.setattr(clipshrink, "collect_images", lambda _d, _r: images)
    fake, processed, peak = _make_fake_process_image(images)
    monkeypatch.setattr(clipshrink, "process_image", fake)

    assert clipshrink._run_folder_mode(_make_args(tmp_path, jobs=4)) == 0

    assert len(processed) == 6
    assert peak() >= 2
    assert {out for _, out in processed} == {
        f"img{i}_processed.png" for i in range(6)
    }


def test_folder_mode_serial_when_jobs_1(tmp_path, monkeypatch):
    images = _fake_images(tmp_path, 6)
    monkeypatch.setattr(clipshrink, "collect_images", lambda _d, _r: images)
    fake, processed, peak = _make_fake_process_image(images)
    monkeypatch.setattr(clipshrink, "process_image", fake)

    assert clipshrink._run_folder_mode(_make_args(tmp_path, jobs=1)) == 0

    assert len(processed) == 6
    assert peak() == 1


def test_folder_mode_counts_failures(tmp_path, monkeypatch, capsys):
    images = _fake_images(tmp_path, 3)
    monkeypatch.setattr(clipshrink, "collect_images", lambda _d, _r: images)
    monkeypatch.setattr(utils, "set_last_error", lambda msg: None)

    def fake_process_image(img, args):
        return {"success": False, "final_size": None, "bypassed": False}

    monkeypatch.setattr(clipshrink, "process_image", fake_process_image)

    assert clipshrink._run_folder_mode(_make_args(tmp_path, jobs=2)) == 1
    out = capsys.readouterr().out
    assert "3 succeeded" not in out
    assert "0 succeeded, 3 failed" in out


def test_folder_mode_abort_stops_batch(tmp_path, monkeypatch, capsys):
    images = _fake_images(tmp_path, 4)
    monkeypatch.setattr(clipshrink, "collect_images", lambda _d, _r: images)

    def fake_process_image(img, args):
        utils._aborted = True
        return {"success": True, "final_size": 1, "bypassed": False}

    monkeypatch.setattr(clipshrink, "process_image", fake_process_image)

    try:
        assert clipshrink._run_folder_mode(_make_args(tmp_path, jobs=2)) == 130
    finally:
        utils._aborted = False
    out = capsys.readouterr().out
    assert "Interrupted" in out


def test_folder_mode_single_image_skips_parallel(tmp_path, monkeypatch):
    images = _fake_images(tmp_path, 1)
    monkeypatch.setattr(clipshrink, "collect_images", lambda _d, _r: images)
    fake, processed, peak = _make_fake_process_image(images)
    monkeypatch.setattr(clipshrink, "process_image", fake)

    assert clipshrink._run_folder_mode(_make_args(tmp_path, jobs=4)) == 0
    assert len(processed) == 1
    assert peak() == 1
    assert processed[0][1] == "img0.png"


def test_folder_mode_limit_tool_threads_sets_env(monkeypatch):
    monkeypatch.delenv("MAGICK_THREAD_LIMIT", raising=False)
    monkeypatch.delenv("VIPS_CONCURRENCY", raising=False)
    clipshrink._limit_tool_threads(4)
    assert "MAGICK_THREAD_LIMIT" in os.environ
    assert "VIPS_CONCURRENCY" in os.environ
    assert int(os.environ["MAGICK_THREAD_LIMIT"]) >= 1
    monkeypatch.setenv("MAGICK_THREAD_LIMIT", "2")
    clipshrink._limit_tool_threads(4)
    assert os.environ["MAGICK_THREAD_LIMIT"] == "2"
