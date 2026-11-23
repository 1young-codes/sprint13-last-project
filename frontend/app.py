import json
from pathlib import Path

import streamlit as st
from PIL import Image

RESULT_DIR = Path("/data/outputs/demo")

SAMPLE = {
    "status": "sample",
    "message": "",
    "product_name": "도미",
    "description": "한 줄 설명이 여기에 나옵니다",
    "product_caption": "상품 소개 문구가 여기에 나옵니다.",
    "usage_caption": "활용 예시 문구가 여기에 나옵니다.",
}


def load_result():
    path = RESULT_DIR / "result.json"
    if not path.exists():
        return SAMPLE
    try:
        return {**SAMPLE, **json.loads(path.read_text(encoding="utf-8"))}
    except json.JSONDecodeError:
        return {**SAMPLE, "status": "failed", "message": "result.json을 읽을 수 없습니다."}


def load_image(name):
    path = RESULT_DIR / name
    if path.exists():
        return Image.open(path)
    return Image.new("RGB", (512, 512), (217, 221, 230))


def card(title, image_name, caption, note):
    with st.container(border=True):
        st.subheader(title)
        st.image(load_image(image_name))
        st.write(caption)
        st.caption(note)


st.set_page_config(page_title="우리 가게 SNS", layout="wide")
st.title("게시글 미리보기")

result = load_result()
status = result["status"]

if status == "running":
    st.info("게시글을 만드는 중입니다.")
elif status == "failed":
    st.error(result["message"] or "게시글을 만들지 못했습니다.")
else:
    if status == "sample":
        st.warning("아직 실제 결과가 없어 샘플을 보여줍니다.")
    st.write(f"**{result['product_name']}** · {result['description']}")
    left, right = st.columns(2)
    with left:
        card("실물 소개", "product_photo.png", result["product_caption"], "원본 제품을 보존한 사진")
    with right:
        card("활용 예시", "usage_example.png", result["usage_caption"], "연출된 장면")

if st.button("새로 고침"):
    st.rerun()
