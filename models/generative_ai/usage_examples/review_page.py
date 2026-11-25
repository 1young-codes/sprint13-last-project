"""사람 평가용 폴더를 만듭니다: index.html 하나와 이미지들. 인터넷 없이 파일을 바로 열어 씁니다.

평가 기준은 하나입니다: AI 같은 질감을 얼마나 걷어냈나(얼마나 실제 사진 같나). 음식마다 세 장에 1~3위를 매기고
CSV로 내려받습니다. summarize()가 평가자들의 CSV를 합쳐 조건별 평균 순위를 냅니다.
"""

import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .settings import conditions

CSV_FIELDS = ["평가자", "번호", "사진", "음식", "조건", "로라", "순위"]

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>활용 예시 로라 비교 평가</title>
<style>
:root {{ color-scheme: light dark; --line: #8884; --muted: #888; }}
body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 1400px; padding: 24px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
.bar {{ position: sticky; top: 0; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
       padding: 12px 0; background: Canvas; border-bottom: 1px solid var(--line); z-index: 1; }}
.bar input {{ font: inherit; padding: 6px 8px; }}
button {{ font: inherit; padding: 6px 14px; cursor: pointer; }}
.muted {{ color: var(--muted); }}
section {{ border-bottom: 1px solid var(--line); padding: 20px 0; }}
section.done h2::after {{ content: " ✓"; color: #2a2; }}
h2 {{ font-size: 17px; margin: 0 0 8px; }}
.source {{ display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px; }}
.source img {{ width: 120px; border-radius: 6px; cursor: zoom-in; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
figure {{ margin: 0; }}
figure img {{ width: 100%; border-radius: 6px; cursor: zoom-in; display: block; }}
figcaption {{ display: flex; justify-content: space-between; align-items: center; margin-top: 6px; gap: 8px; }}
figcaption b {{ font-weight: 600; }}
select {{ font: inherit; padding: 4px; }}
details {{ font-size: 13px; }}
#zoom {{ position: fixed; inset: 0; background: #000e; display: none; align-items: center; justify-content: center;
        cursor: zoom-out; z-index: 2; }}
#zoom img {{ max-width: 96vw; max-height: 94vh; }}
#zoom.open {{ display: flex; }}
</style></head><body>
<h1>활용 예시 로라 비교 평가</h1>
<p class="muted">기준: AI 같은 질감을 얼마나 걷어냈나(얼마나 실제 사진 같나). 음식마다 세 장에 1위(가장 실제 사진 같음)부터 3위까지 매기세요. 사진을 누르면 크게 보입니다.</p>
<div class="bar"><label>평가자 이름 <input id="who" placeholder="이름"></label>
<span id="count" class="muted"></span><button id="save">CSV 저장</button></div>
{sections}
<div id="zoom"><img alt=""></div>
<script>
const DATA = {data};
const key = "usage-review:" + location.pathname;
const saved = JSON.parse(localStorage.getItem(key) || "{{}}");
const who = document.getElementById("who");
who.value = saved.who || "";
function store() {{
  const ranks = {{}};
  document.querySelectorAll("select").forEach(s => {{ if (s.value) ranks[s.name] = s.value; }});
  localStorage.setItem(key, JSON.stringify({{who: who.value, ranks}}));
  let done = 0;
  document.querySelectorAll("section").forEach(sec => {{
    const values = [...sec.querySelectorAll("select")].map(s => s.value);
    const ok = values.every(Boolean) && new Set(values).size === values.length;
    sec.classList.toggle("done", ok); if (ok) done++;
  }});
  document.getElementById("count").textContent = `${{done}} / ${{DATA.length}} 완료`;
}}
document.querySelectorAll("select").forEach(s => {{
  s.value = (saved.ranks || {{}})[s.name] || "";
  s.addEventListener("change", () => {{
    // 같은 음식에서 같은 순위를 두 장에 주면 먼저 준 쪽을 비웁니다
    s.closest("section").querySelectorAll("select").forEach(o => {{ if (o !== s && o.value === s.value) o.value = ""; }});
    store();
  }});
}});
who.addEventListener("input", store);
store();
const zoom = document.getElementById("zoom");
document.querySelectorAll("img[data-zoom]").forEach(img => img.addEventListener("click", () => {{
  zoom.querySelector("img").src = img.dataset.zoom; zoom.classList.add("open"); }}));
zoom.addEventListener("click", () => zoom.classList.remove("open"));
document.addEventListener("keydown", e => {{ if (e.key === "Escape") zoom.classList.remove("open"); }});
document.getElementById("save").addEventListener("click", () => {{
  if (!who.value.trim()) {{ who.focus(); alert("평가자 이름을 적어 주세요"); return; }}
  const rows = [{fields}];
  DATA.forEach((item, i) => item.images.forEach(im => {{
    const s = document.querySelector(`select[name="${{item.id}}|${{im.condition}}"]`);
    rows.push([who.value.trim(), i + 1, item.id, item.name, im.label, im.lora, s.value]);
  }}));
  const text = "\\ufeff" + rows.map(r => r.map(v => `"${{String(v).replaceAll('"', '""')}}"`).join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {{type: "text/csv"}}));
  a.download = `순위_${{who.value.trim()}}.csv`; a.click();
}});
</script></body></html>
"""


def thumbnail(source: Path, target: Path, long_side: int) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((long_side, long_side))
        image.save(target, quality=88)


def build(plans: list[dict], images_dir: Path, photos_dir: Path, out: Path) -> Path:
    """plans: plans.json 항목 (id, file, name_ko, option). images_dir/<조건>/<id>.png를 out에 복사해 페이지를 만듭니다."""
    chosen = conditions()
    for folder in ("images", "photos"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    data, sections = [], []
    for number, plan in enumerate(plans, 1):
        thumbnail(photos_dir / plan["file"], out / "photos" / f"{plan['id']}.jpg", 800)
        figures, images = [], []
        for condition in chosen:
            name = f"{plan['id']}_{condition['name']}.png"
            shutil.copy(images_dir / condition["name"] / f"{plan['id']}.png", out / "images" / name)
            lora = condition.get("display_name") or condition.get("file", "로라 없음")
            images.append({"condition": condition["name"], "label": condition["label"], "lora": lora})
            options = "".join(f'<option value="{r}">{r}위</option>' for r in (1, 2, 3))
            figures.append(
                f'<figure><img src="images/{name}" data-zoom="images/{name}" alt="{html.escape(condition["label"])}" '
                f'loading="lazy"><figcaption><span><b>{html.escape(condition["label"])}</b> '
                f'<span class="muted">{html.escape(lora)}</span></span>'
                f'<select name="{plan["id"]}|{condition["name"]}" aria-label="{html.escape(condition["label"])} 순위">'
                f'<option value="">순위</option>{options}</select></figcaption></figure>')
        option = plan["option"]
        title = f"{number}. {plan['name_ko']} → {option['name_ko']}"
        data.append({"id": plan["id"], "name": f"{plan['name_ko']} → {option['name_ko']}", "images": images})
        sections.append(
            f'<section><h2>{html.escape(title)}</h2><div class="source">'
            f'<img src="photos/{plan["id"]}.jpg" data-zoom="photos/{plan["id"]}.jpg" alt="원본 사진">'
            f'<div><div class="muted">원본 사진 · {html.escape(plan["category_label"])} · '
            f'{"가로" if plan["orientation"] == "landscape" else "세로"}</div>'
            f'<details><summary>생성 프롬프트</summary>{html.escape(option["prompt"])}</details></div></div>'
            f'<div class="grid">{"".join(figures)}</div></section>')
    page = PAGE.format(sections="\n".join(sections), data=json.dumps(data, ensure_ascii=False),
                       fields=json.dumps(CSV_FIELDS, ensure_ascii=False)[1:-1])
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"


def summarize(csv_files: list[Path]) -> dict:
    """평가자 CSV들 → 조건별 평균 순위(낮을수록 실제 사진 같음)와 1위 횟수, 평가자별 평균."""
    ranks, firsts, by_person = defaultdict(list), defaultdict(int), defaultdict(lambda: defaultdict(list))
    for path in csv_files:
        with open(path, encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if not row["순위"]:
                    continue
                rank, label = int(row["순위"]), f"{row['조건']} ({row['로라']})"
                ranks[label].append(rank)
                firsts[label] += rank == 1
                by_person[row["평가자"]][label].append(rank)
    mean = lambda values: round(sum(values) / len(values), 2)  # noqa: E731
    return {"평균 순위": {k: mean(v) for k, v in sorted(ranks.items(), key=lambda kv: mean(kv[1]))},
            "1위 횟수": dict(firsts), "매긴 수": {k: len(v) for k, v in ranks.items()},
            "평가자별 평균 순위": {p: {k: mean(v) for k, v in d.items()} for p, d in by_person.items()}}
