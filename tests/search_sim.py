import os
import re
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

from core import compressors, search, ssim
from core.compressors import CodecEncoder

W0 = 1000
MAX = 100000


class Sim:
    def __init__(self, spec):
        self.max_size = spec["max_size"]
        self.model = spec["model"]
        self.reg = {}
        self.ident = {}
        self.flow = []
        self.probes = []
        self.main = threading.get_ident()
        self.src = Path(spec["src"])
        self.out = Path(spec["out"])
        self.reg[str(self.src)] = (spec["src_w"], spec["src_size"])
        self.ident[str(self.src)] = (spec["src_w"], spec["src_size"])

    def width(self, p):
        return self.reg.get(str(p), (0, 0))[0]

    def height(self, p):
        return max(1, round(self.width(p) * 3 // 4))

    def size(self, p):
        return self.reg.get(str(p), (0, 0))[1]

    def is_main(self):
        return threading.get_ident() == self.main

    def norm(self, p):
        s = str(p)
        if s == str(self.src):
            return "SRC"
        if s == str(self.out):
            return "OUT"
        w, size = self.ident.get(s, (0, 0))
        return f"F{w}x{size}"


def make_encode(sim, name):
    def encode(inp, outp, quality, subsampling="4:2:0", strip_metadata=False, backend=None):
        w = sim.width(inp)
        size = sim.model.lossy(quality, w)
        sim.reg[str(outp)] = (w, size)
        sim.ident[str(outp)] = (w, size)
        outp.write_bytes(b"x")
        ev = ["encode", name, subsampling, quality, size, size <= sim.max_size]
        (sim.probes if not sim.is_main() else sim.flow).append(ev)
        return True
    return encode


def make_failing_encode(sim, name):
    def encode(inp, outp, quality, subsampling="4:2:0", strip_metadata=False, backend=None):
        ev = ["encode", name, subsampling, quality, 0, False]
        (sim.probes if not sim.is_main() else sim.flow).append(ev)
        return False
    return encode


def make_palette(sim, tag, fail=False):
    def reduce(inp, outp, colors, strip_metadata=False):
        if fail:
            ev = [tag, colors, 0, False]
            (sim.probes if not sim.is_main() else sim.flow).append(ev)
            return False
        w = sim.width(inp)
        size = sim.model.palette(colors, w)
        sim.reg[str(outp)] = (w, size)
        sim.ident[str(outp)] = (w, size)
        outp.write_bytes(b"x")
        ev = [tag, colors, size, size <= sim.max_size]
        (sim.probes if not sim.is_main() else sim.flow).append(ev)
        return True
    return reduce


def make_extent(sim):
    def extent(inp, outp, max_bytes, subsampling, strip_metadata):
        w = sim.width(inp)
        size = sim.model.extent(w)
        sim.reg[str(outp)] = (w, size)
        sim.ident[str(outp)] = (w, size)
        outp.write_bytes(b"x")
        sim.flow.append(["extent", size])
        return True
    return extent


def make_resize(sim, tag):
    def resize(inp, outp, percentage, kernel="Lanczos", **kwargs):
        w = sim.width(inp)
        in_size = sim.size(inp)
        w2 = sim.model.new_width(w, percentage)
        size = sim.model.resized(w, w2, in_size)
        sim.reg[str(outp)] = (w2, size)
        sim.ident[str(outp)] = (w2, size)
        outp.write_bytes(b"x")
        sim.flow.append(["resize", tag, percentage, w2, size, sim.norm(inp)])
        return True
    return resize


def make_gif_resize(sim):
    def gif_resize(inp, outp, geometry, kernel="Lanczos", nearest=False, strip_metadata=False):
        pct = int(geometry.rstrip("%")) if geometry.endswith("%") else 50
        w = sim.width(inp)
        in_size = sim.size(inp)
        w2 = sim.model.new_width(w, pct)
        size = sim.model.resized(w, w2, in_size)
        sim.reg[str(outp)] = (w2, size)
        sim.ident[str(outp)] = (w2, size)
        outp.write_bytes(b"x")
        sim.flow.append(["gif_resize", geometry, nearest, w2, size, sim.norm(inp)])
        return True
    return gif_resize


def run_iter_lossy(sim, spec):
    enc = CodecEncoder(
        "fake", make_encode(sim, "fake"),
        extent=make_extent(sim) if spec.get("extent") else None,
        suffix=".jpg",
    )
    return compressors.iterative_lossy_optimize(sim.src, sim.out, sim.max_size, enc,
                                                **spec["params"])


def run_iter_lossy_fail(sim, spec):
    enc = CodecEncoder("dead", make_failing_encode(sim, "dead"), suffix=".jpg")
    return compressors.iterative_lossy_optimize(sim.src, sim.out, sim.max_size, enc,
                                                **spec["params"])


def run_slowest_lossy(sim, spec):
    encoders = {}
    for name, ss in spec["encoders"]:
        encoders[name.upper()] = CodecEncoder(
            name, make_encode(sim, name), subsampling_supported=ss,
            suffix=".jpg" if name == "jpeg" else ".webp",
        )
    return compressors.slowest_lossy_optimize(sim.src, sim.out, sim.max_size, encoders,
                                           **spec["params"])


def run_iter_art(sim, spec):
    return compressors.iterative_art_optimize(
        sim.src, sim.out, sim.max_size,
        pixel_art_mode=spec.get("pixel_art", False),
        start_colors=256, min_colors=128, **spec["params"],
    )


def run_slowest_art(sim, spec):
    return compressors.slowest_art_optimize(
        sim.src, sim.out, sim.max_size,
        pixel_art_mode=spec.get("pixel_art", False),
        start_colors=256, min_colors=2, **spec["params"],
    )


def run_iter_anim(sim, spec):
    return compressors.iterative_animated_gif_optimize(
        sim.src, sim.out, sim.max_size,
        pixel_art_mode=spec.get("pixel_art", False),
        start_colors=256, min_colors=128, **spec["params"],
    )


def run_slowest_anim(sim, spec):
    return compressors.slowest_animated_gif_optimize(
        sim.src, sim.out, sim.max_size,
        pixel_art_mode=spec.get("pixel_art", False),
        start_colors=256, min_colors=2, **spec["params"],
    )


def run_iter_lossy_frontier(sim, spec):
    encoders = {}
    for name, ss in spec["encoders"]:
        encoders[name.upper()] = CodecEncoder(
            name, make_encode(sim, name), subsampling_supported=ss,
            suffix=".jpg" if name == "jpeg" else ".webp",
        )
    return compressors.iterative_lossy_optimize(
        sim.src, sim.out, sim.max_size, encoders,
        **spec["params"], frontier=True, compare_subsampling=True,
        deep_step=75, ssim=True, frontier_ceiling=None,
    )


def run_iter_art_frontier(sim, spec):
    return compressors.iterative_art_optimize(
        sim.src, sim.out, sim.max_size,
        pixel_art_mode=spec.get("pixel_art", False),
        start_colors=256, min_colors=2, **spec["params"],
        frontier=True, ssim=True, frontier_ceiling=None,
    )


def run_iter_anim_frontier(sim, spec):
    return compressors.iterative_animated_gif_optimize(
        sim.src, sim.out, sim.max_size,
        pixel_art_mode=spec.get("pixel_art", False),
        start_colors=256, min_colors=2, **spec["params"],
        frontier=True, ssim=True, frontier_ceiling=None,
    )


RUNNERS = {

    "iter_lossy": run_iter_lossy,
    "iter_lossy_fail": run_iter_lossy_fail,
    "slowest_lossy": run_slowest_lossy,
    "iter_art": run_iter_art,
    "slowest_art": run_slowest_art,
    "iter_anim": run_iter_anim,
    "slowest_anim": run_slowest_anim,
    "iter_lossy_frontier": run_iter_lossy_frontier,
    "iter_art_frontier": run_iter_art_frontier,
    "iter_anim_frontier": run_iter_anim_frontier,
}


def _spec(**kw):
    return kw


def apply_patches(sim, spec, monkeypatch):
    patches = [
        (search, "_MAX_PARALLEL", 4),
        (search, "_probe_worker_cap", lambda current: 1),
        (compressors.utils, "_aborted", False),
        (compressors.utils, "_abort_if_interrupted", lambda: None),
        (compressors.converters, "get_image_width", sim.width),
        (compressors.converters, "get_image_height", sim.height),
        (compressors.converters, "get_image_filesize", sim.size),
        (compressors.converters, "resize_pct", make_resize(sim, "resize_pct")),
        (compressors.converters, "sample_pct", make_resize(sim, "sample_pct")),
        (compressors.converters, "gif_resize", make_gif_resize(sim)),
        (compressors, "color_palette_reduce",
         make_palette(sim, "palette", fail=spec.get("palette_fail", False))),
        (compressors, "animated_gif_reduce",
         make_palette(sim, "anim", fail=spec.get("palette_fail", False))),
        (ssim, "_chained_reference", lambda ref, cands, reg: ref),
        (ssim, "_animated_chained_reference", lambda ref, cands, reg: ref),
        (ssim, "_ssim_vs_reference",
         lambda ref, cand, reg, at_original=False:
             0.9 - (sim.size(cand) / max(sim.max_size, 1)) * 0.1),
        (ssim, "_animated_ssim_vs_reference",
         lambda ref, cand, reg: 0.9 - (sim.size(cand) / max(sim.max_size, 1)) * 0.1),
    ]
    for mod, attr, val in patches:
        monkeypatch.setattr(mod, attr, val)

    def fake_replace(src, dst):
        sim.flow.append(["replace", sim.norm(src), sim.norm(dst)])
        sim.reg[str(dst)] = sim.reg.pop(str(src), (0, 0))
        sim.ident[str(dst)] = sim.ident.pop(str(src), (0, 0))

    def fake_copy(src, dst):
        sim.flow.append(["copy", sim.norm(src), sim.norm(dst)])
        sim.reg[str(dst)] = sim.reg.get(str(src), (0, 0))
        sim.ident[str(dst)] = sim.ident.get(str(src), (0, 0))

    monkeypatch.setattr(os, "replace", fake_replace)
    monkeypatch.setattr(shutil, "copyfile", fake_copy)


def is_probe_line(line):
    s = line.strip()
    return s.startswith("Testing ") or s.startswith("[")


def norm_stdout(text):
    lines = [re.sub(r"probe_\d+", "probe_*", line)
             for line in text.splitlines() if line.strip()]
    probe_lines = sorted(line for line in lines if is_probe_line(line))
    other = [line for line in lines if not is_probe_line(line)]
    return "\n".join(other + probe_lines)


def run_scenario(sim, spec, monkeypatch, capsys):
    apply_patches(sim, spec, monkeypatch)
    result = RUNNERS[spec["mode"]](sim, spec)
    out = capsys.readouterr().out
    return {
        "result": result,
        "flow": sim.flow,
        "probes": sorted(sim.probes),
        "stdout": norm_stdout(out),
    }


def pixel_model(lossy=None, palette=None, resized=None, extent=None, new_width=None):
    model = {
        "lossy": lossy or (lambda q, w: 0),
        "palette": palette or (lambda c, w: 0),
        "resized": resized or (lambda w, w2, s: max(1, round(s * (w2 / w) ** 2))),
        "extent": extent or (lambda w: 0),
        "new_width": new_width or (lambda w, pct: max(1, round(w * pct / 100))),
    }
    return SimpleNamespace(**model)
