import os
import cv2
import shutil

from config import TEMP_CROPS_DIR, UP_DIR, DOWN_DIR


PROGRESS_FILE = os.path.join(TEMP_CROPS_DIR, "_progress.txt")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 0


def save_progress(index):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write(str(index))


def main():
    files = [f for f in os.listdir(TEMP_CROPS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()

    start_index = load_progress()
    print(f"Found {len(files)} images in temp_crops.")
    print(f"Resume from index: {start_index}")

    for idx in range(start_index, len(files)):
        fname = files[idx]
        path = os.path.join(TEMP_CROPS_DIR, fname)
        img = cv2.imread(path)

        if img is None:
            continue

        display = img.copy()
        cv2.putText(display, f"{idx+1}/{len(files)} : {fname}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, "[U] up   [D] down   [S] skip   [Q] quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Fast Classify", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("u"):
            shutil.move(path, os.path.join(UP_DIR, fname))
            print(f"[UP]   {fname}")
        elif key == ord("d"):
            shutil.move(path, os.path.join(DOWN_DIR, fname))
            print(f"[DOWN] {fname}")
        elif key == ord("s"):
            print(f"[SKIP] {fname}")
        elif key == ord("q"):
            save_progress(idx)
            print("Quit. Progress saved.")
            break

        save_progress(idx + 1)

    cv2.destroyAllWindows()
    print("Classification session ended.")


if __name__ == "__main__":
    main()