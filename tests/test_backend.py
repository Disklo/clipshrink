from core.backend import BackendInfo, detect_backends


def test_detect_backends_returns_backendinfo():
    info = detect_backends()
    assert isinstance(info, BackendInfo)
    assert isinstance(info.os_type, str)
    assert isinstance(info.vips, bool)
    assert isinstance(info.magick, bool)
    assert isinstance(info.pngquant, bool)
    assert isinstance(info.realesrgan, bool)
    assert isinstance(info.waifu2x, bool)
    assert isinstance(info.jpegli, bool)
    assert isinstance(info.mozjpeg, bool)
    assert isinstance(info.avifenc, bool)
    assert isinstance(info.oxipng, bool)


def test_backendinfo_optional_encoder_defaults_false():
    info = BackendInfo(os_type="X11", vips=False, magick=False, pngquant=False,
                       realesrgan=False, waifu2x=False)
    assert info.jpegli is False
    assert info.mozjpeg is False
    assert info.avifenc is False
    assert info.oxipng is False


def test_check_mozjpeg_rejects_stock_cjpeg(monkeypatch):
    from core import backend

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = "libjpeg-turbo version 3.2.0"

    monkeypatch.setattr(backend.shutil, "which", lambda c: "/usr/bin/cjpeg" if c == "cjpeg" else None)
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **k: FakeResult())
    assert backend._check_mozjpeg() is False


def test_check_mozjpeg_accepts_mozjpeg_cjpeg(monkeypatch):
    from core import backend

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = "cjpeg version 4.1.5 (mozjpeg)"

    monkeypatch.setattr(backend.shutil, "which", lambda c: "/opt/mozjpeg/bin/cjpeg" if c == "cjpeg" else None)
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **k: FakeResult())
    assert backend._check_mozjpeg() is True


def test_reload_pyvips_noop_when_already_imported(monkeypatch):
    import pytest

    from core import backend

    if backend.pyvips is None:
        pytest.skip("pyvips not importable in this environment")

    class _Boom:
        def __getattr__(self, name):
            raise AssertionError("import attempted although pyvips was loaded")

    monkeypatch.setattr(backend, "importlib", _Boom())
    assert backend.reload_pyvips() is True
    assert backend.pyvips is not None


def test_reload_pyvips_reimports_and_shares_binding(monkeypatch):
    import sys
    import types

    from core import backend, contentscan, converters

    fake = types.ModuleType("pyvips")
    monkeypatch.setitem(sys.modules, "pyvips", fake)
    saved = (backend.pyvips, converters.pyvips, contentscan.pyvips)
    backend.pyvips = None
    converters.pyvips = None
    contentscan.pyvips = None
    try:
        assert backend.reload_pyvips() is True
        assert backend.pyvips is fake
        assert converters.pyvips is fake
        assert contentscan.pyvips is fake
    finally:
        backend.pyvips, converters.pyvips, contentscan.pyvips = saved


def test_reload_pyvips_failure_keeps_consumer_bindings(monkeypatch):
    import pytest

    from core import backend, contentscan, converters

    if backend.pyvips is None:
        pytest.skip("pyvips not importable in this environment")

    def boom(name):
        raise OSError("cannot load library")

    monkeypatch.setattr(backend.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(backend.importlib, "import_module", boom)
    saved = (backend.pyvips, converters.pyvips, contentscan.pyvips)
    backend.pyvips = None
    try:
        assert backend.reload_pyvips() is False
        assert converters.pyvips is saved[1]
        assert contentscan.pyvips is saved[2]
    finally:
        backend.pyvips, converters.pyvips, contentscan.pyvips = saved


def test_reload_pyvips_returns_false_when_import_fails(monkeypatch):
    from core import backend, contentscan, converters

    def boom(name):
        raise OSError("cannot load library")

    monkeypatch.setattr(backend.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(backend.importlib, "import_module", boom)
    saved = (backend.pyvips, converters.pyvips, contentscan.pyvips)
    backend.pyvips = None
    converters.pyvips = None
    contentscan.pyvips = None
    try:
        assert backend.reload_pyvips() is False
        assert backend.pyvips is None
        assert converters.pyvips is None
        assert contentscan.pyvips is None
    finally:
        backend.pyvips, converters.pyvips, contentscan.pyvips = saved
