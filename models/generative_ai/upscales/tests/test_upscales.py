"""그래픽 카드와 모델 없이 도는 확인: 설정 파일, 실행 명령 만들기, 결과 모으기, 조각 나누기."""

from pathlib import Path

import numpy as np
from PIL import Image

from models.generative_ai.upscales.run_model import build_command, collect_outputs
from models.generative_ai.upscales.settings import load_spec, model_names, PACKAGE_DIR

EXPECTED_MODELS = {"pixrestore"}  # 비교 끝에 고른 모델 하나만 남김


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


def test_runner_env_is_not_a_model():
    assert "run" not in load_spec("runner") and load_spec("runner")["environment"]["python"] >= "3.11"
    assert (PACKAGE_DIR / "runner" / "requirements.txt").is_file()


def test_build_command_fills_placeholders(tmp_path):
    spec = {"run": {"command": ["x.py", "-i", "{input_dir}", "-o", "{output_dir}", "--upscale", "{scale}"]}}
    command = build_command("m", spec, tmp_path, tmp_path / "out", 4)
    assert command[1:] == ["x.py", "-i", str(tmp_path.resolve()), "-o", str((tmp_path / "out").resolve()), "--upscale", "4"]


def test_collect_outputs_matches_names_and_fixes_size(tmp_path):
    source = tmp_path / "photo.png"
    Image.new("RGB", (10, 6)).save(source)
    raw = tmp_path / "raw" / "sub"
    raw.mkdir(parents=True)
    Image.new("RGB", (41, 24)).save(raw / "photo_upscaled.png")
    result = collect_outputs(tmp_path / "raw", tmp_path, [source], 4)
    assert result["photo"]["resized_to"] == [40, 24]
    assert Image.open(tmp_path / "photo.png").size == (40, 24)


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
    assert joined.size == (600, 400) and np.array_equal(np.asarray(joined), np.asarray(expected))
