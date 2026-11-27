"""글자 레이어를 HTML로 그려 투명 PNG로 캡처합니다. 줄마다 파일 하나입니다.

브라우저(Chromium)가 한글을 폰트 그대로 그리므로 글자가 깨지지 않습니다.
처음 한 번 설치: pip install playwright && playwright install --with-deps chromium
"""

import html
import re
import urllib.request
from pathlib import Path

from .layers import PROJECT_ROOT, load_settings

FONT_HOST = "http://fonts.local/"  # 브라우저가 이 주소로 폰트를 요청하면 내려받아 둔 파일로 답합니다
LINE_HEIGHT = 1.15

FIT_SCRIPT = """() => {
  const failed = [];
  for (const el of document.querySelectorAll('.t')) {
    const w = el.clientWidth, h = el.clientHeight;
    const minSize = Number(el.dataset.minFontSize || 6);
    const text = document.createRange();
    text.selectNodeContents(el.firstElementChild);
    const fits = () => text.getBoundingClientRect().width <= w && el.scrollHeight <= h &&
      Array.from(text.getClientRects()).filter(rect => rect.width > 0).length <= Number(el.dataset.maxLines);
    let size = Math.min(Math.floor(h * 0.9), Number(el.dataset.maxFontSize));
    el.style.fontSize = size + 'px';
    while (!fits() && size > minSize) {
      size -= 1;
      el.style.fontSize = size + 'px';
    }
    if (el.dataset.minFontSize && (size < minSize || !fits())) failed.push(el.id);
  }
  return failed;
}"""


def minimum_font_size(canvas_width: int) -> int:
    return max(6, round(canvas_width * 0.02))


def font_file(name: str) -> Path:
    """고정 폰트 파일. 없으면 settings.toml의 주소에서 한 번 내려받습니다."""
    settings = load_settings()
    font = settings["fonts"][name]
    path = PROJECT_ROOT / settings["paths"]["font_dir"] / font["file"]
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(font["url"], path)
    return path


def page_html(texts: list[dict], size: tuple[int, int]) -> str:
    fonts = load_settings()["fonts"]
    faces = "".join(f"@font-face {{ font-family: '{font['file']}'; src: url('{FONT_HOST}{font['file']}'); "
                    f"font-weight: 100 900; }}" for font in {fonts[t['font']]['file']: fonts[t['font']]
                                                            for t in texts}.values())
    boxes = []
    for i, t in enumerate(texts):
        align = t.get("align", "center")
        justify = {"left": "flex-start", "center": "center", "right": "flex-end"}[align]
        max_lines = t.get("max_lines", 1)
        break_after = t.get("break_after")
        flow = "nowrap" if max_lines == 1 else "normal"
        copy = html.escape(t["text"])
        if max_lines == 2 and type(break_after) is int:
            words = list(re.finditer(r"\S+", t["text"]))
            if 0 < break_after < len(words):
                end, start = words[break_after - 1].end(), words[break_after].start()
                copy = html.escape(t["text"][:end]) + "<br>" + html.escape(t["text"][start:])
                flow = "nowrap"
        boxes.append(
            f"<div class='t' id='t{i}' data-max-font-size='{t.get('max_font_size', t['h'])}' "
            f"data-min-font-size='{t.get('min_font_size', '')}' data-max-lines='{max_lines}' "
            f"style='left:{t['x']}px; top:{t['y']}px; width:{t['w']}px; height:{t['h']}px; "
            f"font-family:\"{fonts[t['font']]['file']}\"; font-weight:{fonts[t['font']]['weight']}; color:{t['color']}; "
            f"justify-content:{justify}; text-align:{align}; white-space:{flow}'>"
            f"<span>{copy}</span></div>")
    return (f"<html><head><style>{faces} html, body {{ margin: 0; background: transparent; width: {size[0]}px; "
            f"height: {size[1]}px; }} .t {{ position: absolute; display: flex; align-items: center; "
            f"line-height: {LINE_HEIGHT}; overflow: hidden; }} .t span {{ max-width: 100%; word-break: keep-all; }}"
            f".t[data-max-lines='2'] span {{ text-wrap: balance; }}"
            f"</style></head><body>{''.join(boxes)}</body></html>")


def render_texts(texts: list[dict], size: tuple[int, int], folder: Path) -> None:
    """texts의 각 줄을 folder/<file>로 저장합니다. 줄마다 x, y, w, h, text, font, color, file이 있어야 합니다."""
    from playwright.sync_api import sync_playwright

    if not texts:
        return
    paths = {font_file(t["font"]).name: font_file(t["font"]) for t in texts}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": size[0], "height": size[1]})
        page.route(FONT_HOST + "*", lambda route: route.fulfill(
            body=paths[route.request.url.removeprefix(FONT_HOST)].read_bytes(),
            headers={"Access-Control-Allow-Origin": "*", "Content-Type": "font/ttf"}))
        page.set_content(page_html(texts, size))
        page.evaluate("document.fonts.ready.then(() => true)")
        failed = page.evaluate(FIT_SCRIPT)
        if failed:
            roles = ", ".join(texts[int(element[1:])]["role"] for element in failed)
            raise ValueError(f"글자 크기와 줄 수를 유지할 공간이 부족합니다 ({roles})")
        for i, text in enumerate(texts):
            (folder / text["file"]).write_bytes(page.locator(f"#t{i}").screenshot(omit_background=True))
        browser.close()
