import pytest

from core import clipboard, converters, errors, pipeline


def test_backend_error_is_clipshrink_and_runtime_error():
    assert issubclass(errors.BackendError, errors.ClipshrinkError)
    assert issubclass(errors.BackendError, RuntimeError)


def test_domain_errors_share_the_base():
    assert issubclass(pipeline.ValidationError, errors.ClipshrinkError)
    assert issubclass(clipboard.ClipboardError, errors.ClipshrinkError)
    assert issubclass(errors.ClipshrinkError, Exception)


def test_converter_failure_caught_as_clipshrink_error(monkeypatch, tmp_path):
    def fail(inp, outp, quality, **kwargs):
        raise errors.BackendError("boom")

    monkeypatch.setattr(converters, "jpeg_save", fail)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    out = tmp_path / "out.jpg"
    with pytest.raises(errors.ClipshrinkError):
        converters.jpeg_save(src, out, 80)
