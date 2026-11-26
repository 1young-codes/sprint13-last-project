"""글자 레이어를 HTML로 그려 투명 PNG로 캡처합니다. 줄마다 파일 하나입니다.

브라우저(Chromium)가 한글을 폰트 그대로 그리므로 글자가 깨지지 않습니다.
처음 한 번 설치: pip install playwright && playwright install --with-deps chromium
"""

import html
import urllib.request
from pathlib import Path

from .layers import PROJECT_ROOT, load_settings

FONT_HOST = "http://fonts.local/"  # 브라우저가 이 주소로 폰트를 요청하면 내려받아 둔 파일로 답합니다

FIT_SCRIPT = """() => {
  for (const el of document.querySelectorAll('.t')) {
    const w = el.clientWidth, h = el.clientHeight;
    const text = document.createRange();
    text.selectNodeContents(el);
    let size = Math.min(Math.floor(h * 0.9), Number(el.dataset.maxFontSize));
    el.style.fontSize = size + 'px';
    while ((text.getBoundingClientRect().width > w || el.scrollHeight > h) && size > 6) {
      size -= 1;
      el.style.fontSize = size + 'px';
    }
  }
}"""


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
    boxes = "".join(
        f"<div class='t' id='t{i}' data-max-font-size='{t.get('max_font_size', t['h'])}' "
        f"style='left:{t['x']}px; top:{t['y']}px; width:{t['w']}px; height:{t['h']}px; "
        f"font-family:\"{fonts[t['font']]['file']}\"; font-weight:{fonts[t['font']]['weight']}; color:{t['color']}'>"
        f"{html.escape(t['text'])}</div>" for i, t in enumerate(texts))
    return (f"<html><head><style>{faces} html, body {{ margin: 0; background: transparent; width: {size[0]}px; "
            f"height: {size[1]}px; }} .t {{ position: absolute; display: flex; align-items: center; "
            f"justify-content: center; white-space: nowrap; line-height: 1.15; overflow: hidden; }}</style></head>"
            f"<body>{boxes}</body></html>")


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
        page.evaluate(FIT_SCRIPT)
        for i, text in enumerate(texts):
            (folder / text["file"]).write_bytes(page.locator(f"#t{i}").screenshot(omit_background=True))
        browser.close()
