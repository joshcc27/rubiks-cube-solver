from src.export_onnx import check, export
from src.model import PolicyNet


def test_onnx_export_matches_pytorch(tmp_path):
    model = PolicyNet(width=64, blocks=2).eval()
    path = tmp_path / "policy.onnx"
    export(model, path)
    assert not (tmp_path / "policy.onnx.data").exists()  # weights stay inside the one file
    assert check(model, path) < 1e-4
