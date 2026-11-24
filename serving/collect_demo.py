"""모델 결과를 화면이 읽는 데모 폴더(/data/outputs/demo)에 모은다.

형식은 docs/result-format.md를 따른다. GPU는 쓰지 않는다.

예:
    python serving/collect_demo.py --photo /data/datasets/demo/domi_original.png \
        --name 참도미 --description "[임시 설명] 한 줄 설명"
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageFilter

MIN_SIDE = 512
# ponytail: 임시 문구. PR #6이 합쳐지면 generate_product_captions 호출로 바꾼다.
PLACEHOLDER_PRODUCT = "[임시 문구] {name} 상품 소개 문구가 들어갈 자리입니다."
PLACEHOLDER_USAGE = "[임시 문구] {name} 활용 예시 문구가 들어갈 자리입니다."


def to_square(img: Image.Image) -> Image.Image:
    """자르지 않고 정사각형으로 맞춘다.

    같은 사진을 꽉 차게 키워 흐리게 깔고, 그 위에 원본 전체를 가운데에 올린다.
    한 변이 MIN_SIDE보다 작으면 키운다.
    """
    img = img.convert("RGB")
    w, h = img.size
    side = max(w, h, MIN_SIDE)

    cover = side / min(w, h)
    bg = img.resize((round(w * cover), round(h * cover)), Image.LANCZOS)
    left, top = (bg.width - side) // 2, (bg.height - side) // 2
    bg = bg.crop((left, top, left + side, top + side))
    bg = bg.filter(ImageFilter.GaussianBlur(side // 20))

    fit = side / max(w, h)
    fg = img if fit == 1 else img.resize((round(w * fit), round(h * fit)), Image.LANCZOS)
    bg.paste(fg, ((side - fg.width) // 2, (side - fg.height) // 2))
    return bg


def write_result(out: Path, data: dict) -> None:
    # 화면이 쓰다 만 파일을 읽지 않도록 임시 파일에 쓴 뒤 바꿔치기한다.
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out / "result.json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--photo", required=True, type=Path, help="실물 사진 경로 (보정 결과, 각도 결과 등)")
    p.add_argument("--usage", type=Path, help="활용 예시 사진 경로 (없으면 화면은 빈 칸)")
    p.add_argument("--name", required=True, help="상품 이름")
    p.add_argument("--description", required=True, help="한 줄 설명")
    p.add_argument("--product-caption", help="상품 소개 문구 (없으면 임시 문장)")
    p.add_argument("--usage-caption", help="활용 예시 문구 (없으면 임시 문장)")
    p.add_argument("--model-id", default="", help="사진을 만든 모델 이름")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("/data/outputs/demo"))
    args = p.parse_args(argv)

    os.umask(0o002)  # 팀원(adteam)이 결과 파일을 고치고 지울 수 있게
    args.out.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "running",
        "message": "",
        "product_name": args.name,
        "description": args.description,
        "product_caption": "",
        "usage_caption": "",
        "model_id": args.model_id,
        "seed": args.seed,
    }
    write_result(args.out, result)

    try:
        to_square(Image.open(args.photo)).save(args.out / "product_photo.png")

        usage_path = args.out / "usage_example.png"
        if args.usage:
            to_square(Image.open(args.usage)).save(usage_path)
        else:
            usage_path.unlink(missing_ok=True)  # 지난 실행의 사진이 새 결과처럼 보이지 않게

        result["product_caption"] = args.product_caption or PLACEHOLDER_PRODUCT.format(name=args.name)
        result["usage_caption"] = args.usage_caption or PLACEHOLDER_USAGE.format(name=args.name)
        result["status"] = "ready"
    except Exception as e:  # 화면에 이유를 보여 준다
        result["status"] = "failed"
        result["message"] = f"결과를 모으지 못했습니다: {e}"
    write_result(args.out, result)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
