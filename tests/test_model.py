import torch
from lenet1_physical.model.lenet import LeNet5


def test_forward_shape():
    m = LeNet5()
    x = torch.zeros(1, 1, 28, 28)
    out = m(x)
    assert out.shape == (1, 10)


def test_layer_shapes_match_spec():
    m = LeNet5()
    x = torch.zeros(1, 1, 28, 28)
    a = m.forward_with_activations(x)
    assert a["L1"].shape == (1, 1, 28, 28)
    assert a["L2"].shape == (1, 4, 24, 24)
    assert a["L3"].shape == (1, 4, 12, 12)
    assert a["L4"].shape == (1, 12, 8, 8)
    assert a["L5"].shape == (1, 12, 4, 4)
    assert a["L6"].shape == (1, 10)


def test_relu_is_applied_so_no_negatives_in_visualizable_layers():
    torch.manual_seed(0)
    m = LeNet5()
    x = torch.randn(1, 1, 28, 28)
    a = m.forward_with_activations(x)
    for k in ("L2", "L3", "L4", "L5"):
        assert (a[k] >= 0).all(), f"{k} has negatives -- visualization expects ReLU'd"


import numpy as np
from pathlib import Path
from lenet1_physical.model.inference import LeNetInference


def test_inference_loads_weights_and_returns_numpy_dict(tmp_path):
    inf = LeNetInference(Path("weights/lenet5.pt"))
    img = np.zeros((28, 28), dtype=np.float32)
    out = inf.run(img)
    assert set(out) == {"L1", "L2", "L3", "L4", "L5", "L6", "prediction"}
    assert isinstance(out["prediction"], int)
    assert 0 <= out["prediction"] <= 9
    assert out["L2"].shape == (4, 24, 24)
    assert out["L2"].dtype == np.float32
