"""광고 시안을 보며 다음 편집을 고르는 화면 없는 실행기입니다."""

import argparse
from functools import cache
from hashlib import sha256
import json
from pathlib import Path
import shutil

from PIL import Image, ImageOps

from .art_direction import THEMES
from .creative_brief import ask_stage, required_text
from .layers import load_layout, load_settings, save_layout, to_float
from .scene_ad import make_scene_ad
from .studio_edits import edit_version, foreground_mask
from .studio_formats import FORMATS, adapt_format, fill_margin
from .studio_guides import START_GUIDES, find_actions


START = """Choose how to start one Korean product poster. Select one route from routes.
The first image shows the product. Remaining images are optional composition references.
Use the product facts and the owner's idea. JSON keys: route, instruction (specific direction), reason (Korean)."""

LOOK = """Look at the real poster images. The first image shows the source product. Then see the posters in
version_order, followed by those same posters at phone-feed size from left to right. Any images after the
phone preview are design references. Pick the better actual poster when there are two, and name the single most consequential visible
problem of the chosen version. First check the actual product, count, shape, and whether displayed use is
physically believable, including how the parts are held or supported. Check the required Korean copy exactly.
Product facts are background evidence. Omitted optional product facts such as volume or weight are allowed
unless the poster makes a misleading offer. Then inspect visual focus, visual idea, the scale and hierarchy of
the headline, and how the photo and type relate. Evaluate the poster against the remaining design references.
Use any supplied theme guides. Identify the actual visible line, action, light, or scale that leads the eye
between the image and the headline. Empty margin and legible copy beside a product are insufficient if the
image does not show the message. Classify a missing photographic idea as scene, and a weak reading path as layout.
For ready, name the visible treatment in observation and explain how it makes this particular selling point
more noticeable than in a standard catalog photo with the same words. Look for an actual use moment, a
deliberate crop or angle, directed lighting that singles out a detail, or type that interacts with a specific
shape or motion in the photo. Allow restrained minimal designs when the treatment is visible and purposeful.
A centered product, aligned type and recognizable details can all be present in a catalog image. Classify
scene if the shot could be swapped for a standard product photo without weakening the idea. Classify layout
if the shot shows the idea but the type and spacing flatten it into disconnected information.
Use the phone preview to read it as a shopper seeing the whole poster. Check whether the product name and price can be
spotted and read as related information after the first glance. Report typography if either is visually lost
or the price reads like a detached footnote. A supporting price can be smaller than the headline.
Do not approve based on legibility alone. Give no score. Return JSON with version (from version_order),
finding (one of ready, typography, layout, distraction, product, scene), and observation (a concrete Korean sentence).
Use ready when the poster could serve this campaign as seen."""

CAMPAIGN_CHECK = """Recheck the candidate against the actual design references and the actual source product.
Image order: source product, finished candidate, candidate photograph without type, then reference posters.
Name one visible treatment that makes a reference's message work. Compare the candidate's execution at that
level, allowing for its own category and mood. A centered product, readable type and a visible feature already
clear in the source do not by themselves establish a campaign idea. Compare the source and the candidate
photograph before attributing a product feature to the new lighting, view or crop. If the scene adds no
purposeful treatment to make the feature more noticeable, return scene. If the photograph does show the idea
but scale, typography or placement weakens it, return layout. Allow quiet minimalism when the treatment is
visible and supports the message. Return JSON with finding (ready, scene or layout), observation (a concrete
Korean sentence about the candidate), and reference_observation (a concrete Korean detail from a reference)."""

ACT = """Use the evidence in observation and the actual chosen poster (first image) to pick one action.
The second image is the source photo. Ground any framing or staging change in a specific feature visible there.
Use only a tool in actions and follow its guide. Choose the smallest change that can fix the visible issue.
When the finding is scene because a selling detail does not stand out, name the detail the next photograph must
show more clearly and how it should read. Let the framing step compare the whole and close views. Keep a full
view only when the owner requires it or the message depends on the complete shape, packaging or arrangement.
Read the earlier edits and what remained visible after each comparison. If moving or enlarging the words
left the same photo-and-copy disconnect, choose a different composition action or revise the photography
when available. Do not repeat an earlier edit that left the same issue.
Keep the exact wording. Return JSON with action, arguments (an object with only the listed input keys),
and reason (Korean, briefly saying what will visibly change)."""

FINDINGS = {"ready", "typography", "layout", "distraction", "product", "scene"}


def read_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def phone_preview(posters):
    sheet = Image.new("RGB", (768, 768), "#222222")
    tiles = [ImageOps.contain(poster, (360, 704), Image.Resampling.LANCZOS) for poster in posters]
    if len(tiles) == 1:
        tile = tiles[0]
        sheet.paste(tile, ((sheet.width - tile.width) // 2, (sheet.height - tile.height) // 2))
    else:
        for index, tile in enumerate(tiles):
            sheet.paste(tile, (12 + index * 384 + (360 - tile.width) // 2, (sheet.height - tile.height) // 2))
    return sheet


def run_studio(photo, fields, idea, output, ai, generation=None, *, references=(), draft=None, mask_path=None,
               edit_limit=3, formats=(), notify=None):
    if (not isinstance(fields, dict) or not isinstance(fields.get("product_name"), str)
            or not fields["product_name"].strip() or type(fields.get("price_krw")) is not int or fields["price_krw"] < 0):
        raise ValueError("상품명과 원 단위 가격을 입력하세요")
    if edit_limit < 0 or (draft is None and generation is None) or (mask_path is not None and draft is None):
        raise ValueError("시안, 생성 기능 또는 수정 횟수 설정을 확인하세요")
    if set(formats) - FORMATS.keys():
        raise ValueError("지원하지 않는 게시 규격입니다")
    source_version = None
    pending_review = None
    if draft is not None:
        source_version = load_layout(draft)
        if source_version.get("preview") or source_version.get("fields") != fields:
            raise ValueError("같은 상품의 완성 시안을 사용하세요")
        actual = {item["role"]: item["text"] for item in source_version["texts"]}
        expected = {"product_name": fields["product_name"], "price": f"{fields['price_krw']:,}원"}
        if "headline" in fields:
            expected["headline"] = required_text(fields, "headline", "확정 문구")
        if any(actual.get(role) != text for role, text in expected.items()):
            raise ValueError("시안의 상품명, 가격 또는 확정 문구가 입력과 다릅니다")
        saved_review = source_version.get("pending_review")
        if (isinstance(saved_review, dict) and isinstance(saved_review.get("finding"), str)
                and saved_review["finding"] in FINDINGS and saved_review["finding"] != "ready"
                and isinstance(saved_review.get("observation"), str) and saved_review["observation"].strip()
                and saved_review.get("image_hash") == sha256((draft / "ad.png").read_bytes()).hexdigest()):
            pending_review = saved_review
    output.mkdir(parents=True, exist_ok=False)
    versions = output / "versions"
    versions.mkdir()

    def record(step, **data):
        event = {"step": step, **data}
        with (output / "events.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
        if notify:
            notify(event)

    current = "v01"
    first = versions / current
    sources = [to_float(image) for image in references]
    record("product", product_name=fields["product_name"], price_krw=fields["price_krw"])
    if source_version is not None:
        shutil.copytree(draft, first)
        if mask_path is not None:
            artwork = source_version["artwork"]
            with Image.open(mask_path) as mask:
                mask = ImageOps.fit(mask.convert("L"), (artwork["w"], artwork["h"]), Image.Resampling.NEAREST)
                mask.save(first / "foreground.png")
    else:
        routes = {name: guide for name, guide in START_GUIDES.items() if sources or name != "reference_poster"}
        start = ask_stage(ai, "studio_start", START, {"fields": fields, "idea": idea, "routes": routes},
                          [to_float(photo), *sources])
        route = start.get("route")
        if not isinstance(route, str) or route not in routes:
            raise ValueError("광고를 시작할 작업을 고르지 못했습니다")
        instruction = required_text(start, "instruction", "광고 시작")
        record("start", route=route, reason=required_text(start, "reason", "광고 시작"), instruction=instruction)
        make_scene_ad(photo, fields, f"{idea}\n{routes[route]} {instruction}", generation() | {"ai": ai}, first,
                      references=references)
    record("poster", version=current, image=(first / "ad.png").relative_to(output).as_posix())

    pending = None
    attempts = []
    observation = ""
    last_failure = None
    for turn in range(edit_limit + 1):
        order = [current, pending] if pending else [current]
        posters = [read_image(versions / name / "ad.png") for name in order]
        pictures = {name: to_float(poster) for name, poster in zip(order, posters, strict=True)}
        edition = load_layout(versions / current)
        required_copy = {text["role"]: text["text"] for text in edition["texts"]}
        directions = {}
        for name in order:
            version = edition if name == current else load_layout(versions / name)
            direction = version.get("art_direction_plan")
            if direction and direction.get("theme") in THEMES:
                directions[name] = direction | {"review_guide": THEMES[direction["theme"]]["review"]}
        if turn == 0 and pending_review:
            review = pending_review | {"version": current}
            record("resume_review", version=current, finding=review["finding"], observation=review["observation"])
        else:
            review = ask_stage(ai, "studio_look", LOOK, {"product_facts": fields, "required_copy": required_copy,
                                                       "idea": idea, "version_order": order, "art_directions": directions},
                               [to_float(photo), *pictures.values(), to_float(phone_preview(posters)), *sources])
        visible = review.get("version")
        finding = review.get("finding")
        if visible not in order or not isinstance(finding, str) or finding not in FINDINGS:
            raise ValueError("비교할 시안이나 수정 이유를 확인하지 못했습니다")
        observation = required_text(review, "observation", "시안 검토")
        if finding == "ready" and sources:
            selected = versions / visible
            scene = read_image(selected / load_layout(selected)["artwork"]["file"])
            check = ask_stage(ai, "studio_campaign_check", CAMPAIGN_CHECK, {"required_copy": required_copy},
                              [to_float(photo), pictures[visible], to_float(scene), *sources])
            finding = check.get("finding")
            if finding not in ("ready", "scene", "layout"):
                raise ValueError("참고 포스터와 비교한 결과를 확인하지 못했습니다")
            observation = required_text(check, "observation", "참고 포스터 비교")
            record("campaign_check", version=visible, finding=finding, observation=observation,
                   reference_observation=required_text(check, "reference_observation", "참고 포스터 비교"))
        record("look", versions=order, version=visible, finding=finding, observation=observation)
        if pending:
            attempts[-1].update(preferred_version=visible, remaining_finding=finding, observation=observation)
            record("edit_review", **attempts[-1])
        current, pending = visible, None
        base = versions / current
        layout = load_layout(base)
        available = {"hold", "select"}
        if turn < edit_limit:
            if (base / "foreground.png").is_file():
                foreground_mask(base, layout)
                available.add("edit_text")
                if generation is not None:
                    available.add("edit_region")
            if generation is not None:
                available.add("revise_scene")
        guides = find_actions(finding, available)
        record("guides", version=current, actions=list(guides))
        decision = ask_stage(ai, "studio_action", ACT,
                             {"finding": finding, "observation": observation, "actions": guides, "product": fields,
                              "owner_idea": idea,
                              "design": layout["design"], "copy": [text["text"] for text in layout["texts"]],
                              "fonts": list(load_settings()["fonts"]), "last_edit_error": last_failure,
                              "earlier_edits": attempts}, [pictures[current], to_float(photo)])
        action = decision.get("action")
        arguments = decision.get("arguments")
        if (not isinstance(action, str) or action not in guides or not isinstance(arguments, dict)
                or arguments.keys() != guides[action]["arguments"].keys()):
            last_failure = "제공된 작업과 입력에 맞는 선택이 필요합니다"
            record("invalid_action", version=current, observation=last_failure)
            continue
        reason = required_text(decision, "reason", "작업 선택")
        record("action", version=current, action=action, arguments=arguments, reason=reason)
        if action == "select":
            final = output / "final"
            shutil.copytree(base, final)
            if "pending_review" in layout:
                save_layout({key: value for key, value in layout.items() if key != "pending_review"}, final)
            record("selected", version=current, image="final/ad.png", observation=observation)
            variants = {}
            for name in dict.fromkeys(formats):
                variant = output / "formats" / name.replace(":", "x")
                try:
                    if not (base / "foreground.png").is_file():
                        raise ValueError("크기 변경에 사용할 제품 영역이 없습니다")
                    framing = adapt_format(base, variant, name, ai, fill_margin if generation is not None else None)
                except (ValueError, FileNotFoundError) as error:
                    record("format_failed", format=name, observation=str(error))
                    continue
                variants[name] = variant / "ad.png"
                record("format", format=name, image=variants[name].relative_to(output).as_posix(),
                       **({"framing": framing} if framing else {}))
            status = "complete" if set(variants) == set(formats) else "format_review"
            record(status, version=current, image="final/ad.png")
            return {"status": status, "version": current, "image": final / "ad.png", "formats": variants}
        if action == "hold":
            break
        new_id = f"v{turn + 2:02d}"
        revised = versions / new_id
        try:
            if action == "revise_scene":
                make_scene_ad(photo, fields, idea, generation() | {"ai": ai}, revised, references=references,
                              previous=base, feedback=required_text(arguments, "instruction", "장면 편집"),
                              scene_issue=observation if finding == "scene" else None)
            else:
                edit_version(base, revised, action, arguments, generation)
        except ValueError as error:
            last_failure = str(error)
            record("edit_failed", version=current, action=action, observation=last_failure)
            continue
        last_failure = None
        pending = new_id
        attempts.append({"candidate": new_id, "parent": current, "action": action, "arguments": arguments,
                         "reason": reason})
        record("poster", version=new_id, parent=current, image=(revised / "ad.png").relative_to(output).as_posix())

    image = versions / current / "ad.png"
    if finding != "ready":
        saved = load_layout(image.parent)
        saved["pending_review"] = {"finding": finding, "observation": observation,
                                   "image_hash": sha256(image.read_bytes()).hexdigest()}
        save_layout(saved, image.parent)
    record("needs_review", version=current, image=image.relative_to(output).as_posix(), observation=observation)
    return {"status": "needs_review", "version": current, "image": image}


def main():
    from ..photo_retouch.interpret import Interpreter
    from .make_ad import load_models

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path, help="상품 원본 사진")
    parser.add_argument("fields", type=Path, help="상품명과 가격 등의 JSON")
    parser.add_argument("output", type=Path, help="새 작업 폴더")
    parser.add_argument("--idea", default="", help="사장님의 요청")
    parser.add_argument("--references", type=Path, nargs="*", default=[], help="광고 구도 참고 이미지")
    parser.add_argument("--draft", type=Path, help="이미 생성한 완성 시안에서 시작")
    parser.add_argument("--foreground", type=Path, help="기존 시안에서 별도로 저장한 피사체 마스크")
    parser.add_argument("--typography-only", action="store_true", help="사진 생성 없이 기존 시안 글자만 편집")
    parser.add_argument("--edit-limit", type=int, default=3, help="수정 시도 횟수")
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=[], help="채택 시안에 적용할 추가 규격")
    parser.add_argument("--low-vram", action="store_true", help="가중치를 나눠 GPU로 옮김")
    args = parser.parse_args()
    if args.typography_only and args.draft is None:
        parser.error("--typography-only에는 완성 시안이 필요합니다")

    @cache
    def generation():
        return load_models(low_vram=args.low_vram)

    run_studio(read_image(args.photo), json.loads(args.fields.read_text(encoding="utf-8")), args.idea, args.output,
               Interpreter(), None if args.typography_only else generation, references=[read_image(p) for p in args.references],
               draft=args.draft, mask_path=args.foreground, edit_limit=args.edit_limit, formats=args.formats,
               notify=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))


if __name__ == "__main__":
    main()
