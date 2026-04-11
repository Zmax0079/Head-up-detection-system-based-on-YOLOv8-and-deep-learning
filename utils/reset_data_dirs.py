import shutil
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
ROOT = None
for candidate in [THIS_FILE.parent, *THIS_FILE.parents]:
    if (candidate / "config.py").exists():
        ROOT = candidate
        break
if ROOT is None:
    ROOT = THIS_FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import PROCESSED_SINGLE_DIR, SPLIT_DIR, SEQ_DIR, WEIGHTS_DIR, REPORTS_DIR, FIGURES_DIR


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    targets = [
        PROCESSED_SINGLE_DIR,
        SPLIT_DIR,
        SEQ_DIR,
        WEIGHTS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]

    print("将清空以下目录：")
    for t in targets:
        print(f"- {t}")

    for t in targets:
        reset_dir(t)

    print("✅ 已清空旧训练数据/结果目录")


if __name__ == "__main__":
    main()