import numpy as np
from lenet1_physical.colors import LAYER_THEMES, encode_layer


def test_known_themes_present():
    assert set(LAYER_THEMES) == {"L1", "L2", "L3", "L4", "L5", "L6"}


def test_zero_activation_emits_black():
    out = encode_layer("L1", np.zeros((1, 28, 28), dtype=np.float32))
    assert out.shape == (1, 28, 28, 3)
    assert (out == 0).all()


def test_max_activation_emits_full_theme_color():
    out = encode_layer("L2", np.ones((4, 24, 24), dtype=np.float32))
    # L2 is blue: (0, 0, 255)
    assert (out[..., 0] == 0).all()
    assert (out[..., 1] == 0).all()
    assert (out[..., 2] == 255).all()


def test_normalization_per_feature_map():
    # Each feature map normalizes by its own max, so a hot fmap and a cold fmap
    # both reach full brightness at their own peak.
    a = np.array([[[2.0, 0.0]], [[0.0, 0.5]]], dtype=np.float32)  # shape (2,1,2)
    out = encode_layer("L6", a)
    # Inside fmap 0: peak=2.0 -> b=255 at idx 0
    assert out[0, 0, 0, 0] == 255
    # Inside fmap 1: peak=0.5 -> b=255 at idx 1
    assert out[1, 0, 1, 0] == 255


def test_brightness_cap_applied():
    out = encode_layer("L2", np.ones((1, 4, 4), dtype=np.float32), brightness_cap=0.3)
    assert int(out[..., 2].max()) == int(round(255 * 0.3))
