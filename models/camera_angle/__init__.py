"""카메라 앵글 바꾸기. 사용법은 README.md를 보세요."""

import os

# torch 2.14는 일부 행렬 곱을 triton 커널로 바꿔 실행하는데, 이 커널은 처음 쓸 때 C 컴파일러로 빌드됩니다.
# 팀 가상 머신에는 컴파일러가 없어 실패하므로 끄고, torch 기본 연산(결과는 같음)을 씁니다.
# torch를 불러오기 전에 설정해야 해서 패키지 맨 앞에 둡니다. 따로 지정한 값이 있으면 그 값을 따릅니다.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
