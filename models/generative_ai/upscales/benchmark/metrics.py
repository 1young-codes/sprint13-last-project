"""업스케일 결과 한 장의 지표를 잽니다. 무거운 지표 모델은 처음 쓸 때 한 번만 불러옵니다."""

from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: Image.Image, b: Image.Image) -> float:
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return float(10 * np.log10(255**2 / mse)) if mse else float("inf")


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def char_error_rate(reference: str, hypothesis: str) -> float | None:
    """글자 오류율. 공백과 대소문자는 무시합니다. 기준 글자가 없으면 None."""
    ref, hyp = ("".join(t.split()).upper() for t in (reference, hypothesis))
    return edit_distance(ref, hyp) / len(ref) if ref else None


def detail_gain(result: Image.Image, base: Image.Image) -> float:
    """결과가 바이큐빅보다 고주파(라플라시안 분산)를 몇 배 가졌는지. 1보다 크게 높으면 디테일을 지어냈을 수 있습니다."""

    def laplacian_var(image):
        g = np.asarray(image.convert("L"), np.float64)
        lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
        return lap.var()

    return float(laplacian_var(result) / max(laplacian_var(base), 1e-9))


class Scorer:
    def __init__(self, settings: dict, models_dir: Path):
        import pyiqa
        import torch

        self.torch = torch
        self.settings = settings
        self.models_dir = models_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.full_reference = {n: pyiqa.create_metric(n, device=self.device) for n in settings["full_reference"]}
        self.no_reference = {n: pyiqa.create_metric(n, device=self.device) for n in settings["no_reference"]}
        self.lower_better = {n: m.lower_better for n, m in (self.full_reference | self.no_reference).items()}
        self._face = self._ocr = None

    def tensor(self, image: Image.Image):
        array = np.asarray(image.convert("RGB"), np.float32) / 255
        return self.torch.from_numpy(array).permute(2, 0, 1)[None].to(self.device)

    def compare(self, image: Image.Image, reference: Image.Image, prefix: str) -> dict:
        with self.torch.no_grad():
            return {f"{prefix}{n}": float(m(self.tensor(image), self.tensor(reference))) for n, m in self.full_reference.items()}

    def face_embedding(self, image: Image.Image):
        if self._face is None:
            from insightface.app import FaceAnalysis

            self._face = FaceAnalysis(name=self.settings["face_model"], root=str(self.models_dir / "insightface"),
                                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            size = self.settings["face_detect_size"]
            self._face.prepare(ctx_id=0, det_size=(size, size))
        faces = self._face.get(np.asarray(image.convert("RGB"))[:, :, ::-1].copy())
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.normed_embedding

    def read_text(self, image: Image.Image) -> str:
        if self._ocr is None:
            import easyocr

            folder = self.models_dir / "easyocr"
            folder.mkdir(parents=True, exist_ok=True)
            self._ocr = easyocr.Reader(self.settings["ocr_languages"], model_storage_directory=str(folder), gpu=self.device == "cuda")
        return " ".join(self._ocr.readtext(np.asarray(image.convert("RGB")), detail=0))

    def score(self, low: Image.Image, result: Image.Image, truth: Image.Image | None, regions: dict) -> dict:
        """low: 모델 입력, result: 모델 결과, truth: 정답(없으면 None), regions: 입력 기준 영역 상자."""
        base = low.resize(result.size, Image.BICUBIC)
        back = result.resize(low.size, Image.BICUBIC)
        scores = self.compare(back, low, "lr_")
        if truth is not None:
            scores |= self.compare(result, truth, "gt_")
        with self.torch.no_grad():
            scores |= {n: float(m(self.tensor(result))) for n, m in self.no_reference.items()}
        reference = truth or base  # 얼굴과 글자는 정답이 없으면 바이큐빅과 비교 (입력에 있던 것만)
        ref_face, out_face = self.face_embedding(reference), self.face_embedding(result)
        if ref_face is None:
            scores["face_similarity"] = None  # 얼굴이 없는 사진
        else:  # 원본엔 있던 얼굴을 결과에서 못 찾으면 0점
            scores["face_similarity"] = 0.0 if out_face is None else float(np.dot(ref_face, out_face))
        scores["ocr_cer"] = char_error_rate(self.read_text(reference), self.read_text(result))
        scores["detail_gain"] = detail_gain(result, base)
        for name, box in regions.items():
            scores[f"region_{name}_lr_psnr"] = psnr(back.crop(box), low.crop(box))
        return scores
