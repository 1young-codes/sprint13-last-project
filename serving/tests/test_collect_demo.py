import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_demo import main, to_square  # noqa: E402


def noise(w, h):
    return Image.effect_noise((w, h), 80).convert("RGB")


def test_wide_photo_is_square_and_original_untouched():
    src = noise(918, 512)
    out = to_square(src)
    assert out.size == (918, 918)
    top = (918 - 512) // 2
    pasted = out.crop((0, top, 918, top + 512))
    assert ImageChops.difference(pasted, src).getbbox() is None  # 원본 화소 그대로


def test_small_photo_is_upscaled_to_min_side():
    assert to_square(noise(300, 200)).size == (512, 512)


def test_ready_then_failed_and_stale_usage_removed(tmp_path):
    photo = tmp_path / "in.png"
    noise(918, 512).save(photo)
    out = tmp_path / "demo"

    assert main(["--photo", str(photo), "--usage", str(photo), "--name", "도미",
                 "--description", "설명", "--out", str(out)]) == 0
    r = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert r["status"] == "ready" and r["product_caption"] and r["usage_caption"]
    assert (out / "usage_example.png").exists()

    # 활용 사진 없이 다시 돌리면 지난 사진을 지운다
    main(["--photo", str(photo), "--name", "도미", "--description", "설명", "--out", str(out)])
    assert not (out / "usage_example.png").exists()

    # 없는 사진이면 failed와 이유
    assert main(["--photo", str(tmp_path / "없음.png"), "--name", "도미",
                 "--description", "설명", "--out", str(out)]) == 1
    r = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert r["status"] == "failed" and r["message"]
    assert not (out / "result.json.tmp").exists()
