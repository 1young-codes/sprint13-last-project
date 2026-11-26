"""그래픽 카드 없이 도는 확인. 상품 영역은 진짜 모델 대신 '색이 있는 곳'으로 잡습니다."""
import csv

import numpy as np
from PIL import Image

from evaluation import product_fidelity as pf

H, W = 96, 128


def scene(seed=0, stripes=True, shift=0, color=(0.8, 0.2, 0.2), noise=0.02):
    """회색 바탕 위 무늬 있는 빨간 상품. 실제 사진처럼 약한 잡음을 넣습니다."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 0.5, np.float32)
    product = np.ones((48, 64, 3), np.float32) * np.array(color, np.float32)
    if stripes:
        product[:, ::8] *= 0.6
    img[24:72, 32 + shift:96 + shift] = product
    return np.clip(img + rng.normal(0, noise, img.shape).astype(np.float32), 0, 1)


def fake_mask(image):
    return ((image.max(-1) - image.min(-1)) > 0.15).astype(np.float32)


def run(before, after):
    return pf.score(before, after, fake_mask(before), fake_mask(after))


def test_same_photo_is_perfect():
    s = run(scene(), scene())
    assert s["iou"] == 1.0 and s["structure"] > 0.999 and s["delta_e"] < 1e-3


def test_brightening_keeps_shape_and_structure():
    s = run(scene(), np.clip(scene() * 1.15, 0, 1))
    assert s["iou"] > 0.95 and s["structure"] > 0.95 and 0 < s["delta_e"] < 20


def test_erased_pattern_drops_structure():
    assert run(scene(), scene(seed=1, stripes=False))["structure"] < 0.8


def test_drawn_pattern_drops_structure():
    assert run(scene(stripes=False), scene(seed=1))["structure"] < 0.8


def test_recolor_raises_delta_e():
    assert run(scene(), scene(color=(0.2, 0.2, 0.8)))["delta_e"] > 50


def test_moved_product_drops_iou():
    assert run(scene(), scene(shift=16))["iou"] < 0.7


def test_known_gap_perfectly_smooth_erase():
    # ponytail: SSIM 구조 항은 한쪽이 완전히 매끈하면 1에 가깝게 나와, 잡음 없이 칠해 지운 것은 놓친다.
    # 실제 사진은 잡음이 있어 대부분 잡힌다. 생성 결과에서 문제가 되면 세부(국소 표준편차) 비율 지표를 더한다.
    assert run(scene(), scene(stripes=False, noise=0.0))["structure"] > 0.8


def test_summary_worst_direction():
    rows = [{"iou": 0.9, "structure": 0.99, "delta_e": 2.0}, {"iou": 0.5, "structure": 0.7, "delta_e": 9.0}]
    s = pf.summarize(rows)
    assert s["iou"]["worst"] == 0.5 and s["structure"]["worst"] == 0.7 and s["delta_e"]["worst"] == 9.0


def save(path, array):
    Image.fromarray((array * 255).round().astype(np.uint8)).save(path)


def test_cli_upscaled_result_is_resized_back(tmp_path, monkeypatch):
    original = scene()
    save(tmp_path / "a.png", original)
    Image.open(tmp_path / "a.png").resize((W * 2, H * 2), Image.Resampling.LANCZOS).save(tmp_path / "a_x2.png")
    save(tmp_path / "b.png", scene(color=(0.2, 0.2, 0.8)))
    (tmp_path / "pairs.csv").write_text(
        f"# 원본,결과\n{tmp_path / 'a.png'},{tmp_path / 'a_x2.png'}\n{tmp_path / 'a.png'},{tmp_path / 'b.png'}\n",
        encoding="utf-8")
    monkeypatch.setattr(pf, "load_mask_fn", lambda: fake_mask)

    assert pf.main([str(tmp_path / "pairs.csv"), "--out", str(tmp_path / "out.csv")]) == 0
    rows = list(csv.DictReader(open(tmp_path / "out.csv", encoding="utf-8")))
    assert len(rows) == 2
    assert float(rows[0]["iou"]) > 0.95 and float(rows[0]["delta_e"]) < 3  # 2배로 키웠다 줄여도 거의 그대로
    assert float(rows[1]["delta_e"]) > 50
