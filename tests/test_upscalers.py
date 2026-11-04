from core.upscalers import compute_nn_scale_factor, detect_upscale_needed


def test_detect_upscale_needed_flag():
    assert detect_upscale_needed(100, 100, 100, None, None, None, True)
    assert not detect_upscale_needed(100, 100, 100, None, None, None, False)


def test_detect_upscale_needed_dimensions():
    assert detect_upscale_needed(100, 100, 10, 200, None, None, False)
    assert detect_upscale_needed(100, 100, 10, None, 300, None, False)
    assert not detect_upscale_needed(100, 100, 10, 80, None, None, False)
    assert not detect_upscale_needed(100, 100, 10, 100, None, None, False)


def test_detect_upscale_needed_integer_scaling():
    assert detect_upscale_needed(100, 100, 10, None, None, 150, False)
    assert not detect_upscale_needed(100, 100, 10, None, None, 50, False)


def test_compute_nn_scale_factor_defaults_to_two():
    assert compute_nn_scale_factor(100, 100, None, None, None) == 2


def test_compute_nn_scale_factor_integer_scaling():
    assert compute_nn_scale_factor(100, 100, None, None, 200) == 2
    assert compute_nn_scale_factor(100, 100, None, None, 250) == 3
    assert compute_nn_scale_factor(100, 100, None, None, 300) == 3
    assert compute_nn_scale_factor(100, 100, None, None, 50) == 2


def test_compute_nn_scale_factor_max_dimensions():
    assert compute_nn_scale_factor(100, 100, 400, None, None) == 4
    assert compute_nn_scale_factor(100, 100, None, 500, None) == 5
    assert compute_nn_scale_factor(100, 100, 50, None, None) == 2
