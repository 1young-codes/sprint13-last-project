"""그래픽 카드와 모델 없이 도는 확인: 설정 파일, 실행 명령 만들기, 글자 오류율, 선정 규칙."""

from pathlib import Path

import pytest
from PIL import Image

from models.generative_ai.upscales.benchmark.metrics import char_error_rate, psnr
from models.generative_ai.upscales.benchmark.report import average, select
from models.generative_ai.upscales.run_model import build_command, collect_outputs
from models.generative_ai.upscales.settings import load_spec, model_names, read_toml, PACKAGE_DIR

EXPECTED_MODELS = {"fidesr", "pixrestore", "seedvr2", "varestorer", "vosr"}


def test_every_model_has_complete_spec():
    assert set(model_names()) == EXPECTED_MODELS
    for name in model_names():
        spec = load_spec(name)
        assert (PACKAGE_DIR / name / "requirements.txt").is_file()
        assert len(spec["code"]["revision"]) == 40
        for key, entry in spec["weights"].items():
            assert len(entry["revision"]) == 40, f"{name}.{key}: 커밋 고정 필요"
        for place, source in spec["links"].items():
            assert not Path(place).is_absolute()
            assert source.partition(":")[0] in spec["weights"]
        assert set(spec["run"]["scales"]) >= {2, 4}


def test_build_command_fills_placeholders(tmp_path):
    image = tmp_path / "photo.png"
    Image.new("RGB", (50, 30)).save(image)
    spec = {"run": {"command": ["x.py", "{input_file}", "{output_dir}/{input_stem}.png", "{out_short}", "{scale}"]}}
    command = build_command("m", spec, tmp_path, tmp_path / "out", 4, [image])
    assert command[1:] == ["x.py", str(image.resolve()), f"{(tmp_path / 'out').resolve()}/photo.png", "120", "4"]
    with pytest.raises(ValueError):
        build_command("m", spec, tmp_path, tmp_path, 4, [image, image])


def test_collect_outputs_matches_names_and_fixes_size(tmp_path):
    source = tmp_path / "photo.png"
    Image.new("RGB", (10, 6)).save(source)
    raw = tmp_path / "raw" / "sub"
    raw.mkdir(parents=True)
    Image.new("RGB", (41, 24)).save(raw / "photo_upscaled.png")
    result = collect_outputs(tmp_path / "raw", tmp_path, [source], 4)
    assert result["photo"]["resized_to"] == [40, 24]
    assert Image.open(tmp_path / "photo.png").size == (40, 24)


def test_char_error_rate_and_psnr():
    assert char_error_rate("YONHAP NEWS", "yonhapnews") == 0
    assert char_error_rate("ABCD", "ABXD") == 0.25
    assert char_error_rate("", "ABC") is None
    same = Image.new("RGB", (4, 4), (10, 20, 30))
    assert psnr(same, same) == float("inf")


def test_selection_gates_then_ranks():
    config = read_toml(PACKAGE_DIR / "benchmark" / "benchmark.toml")
    rows = [
        {"condition": "c", "model": "good", "status": "ok", "lr_psnr": 40, "gt_lpips": 0.1},
        {"condition": "c", "model": "pretty", "status": "ok", "lr_psnr": 38, "gt_lpips": 0.2},
        {"condition": "c", "model": "liar", "status": "ok", "lr_psnr": 20, "gt_lpips": 0.05},
    ]
    selection = {"gates": [{"condition": "c", "metric": "lr_psnr", "min": 30}],
                 "rank": [{"condition": "c", "metric": "gt_lpips", "weight": 1}]}
    choice = select(average(rows), ["good", "pretty", "liar"], selection, {"lpips": True})
    assert "liar" in choice["failed"] and choice["chosen"] == "good"
    assert config["baseline"] and config["selection"]["gates"]


def test_tiles_cover_image_and_join_back(tmp_path):
    from models.generative_ai.upscales.run_model import cut_tiles, join_tiles, tile_starts

    assert tile_starts(512, 512, 64) == [0]
    assert tile_starts(1000, 512, 64) == [0, 244, 488]
    assert tile_starts(2000, 512, 64)[-1] == 1488
    source = tmp_path / "photo.png"
    Image.effect_noise((150, 100), 60).convert("RGB").save(source)
    tiles = {"size": 256, "overlap": 32}
    layouts = cut_tiles([source], 4, tiles, tmp_path / "tiles")
    (tmp_path / "raw").mkdir()
    join_tiles(layouts, tiles, tmp_path / "tiles", tmp_path / "raw")  # 모델 대신 조각을 그대로 되돌림
    joined = Image.open(tmp_path / "raw" / "photo.png")
    expected = Image.open(source).resize((600, 400), Image.BICUBIC)
    assert joined.size == (600, 400) and psnr(joined, expected) == float("inf")
