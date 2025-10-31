#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
목적: CT/MRI 4-UP 이미지 뷰어 및 QC 도구, 안좋은 데이터셋은 삭제하기

사용법: 
python data_cert.py --root /path/to/dataset --start_case 0 --start_index 0
python data_cert.py --root ./skull_strip/no_bleed_ct_mri --start_case 0 --start_index 0 --save-csv ./exclusions_no_bleed_yk.csv
python data_cert.py --root ./skull_strip/bleed_ct_mri --start_case 0 --start_index 0 --save-csv ./exclusions_bleed_yk.csv

"""

import argparse
import os
import re
import sys
import csv
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

# 이미지 로딩: PIL(일반 이미지), pydicom(DICOM), cv2(회색/정규화 편의)
from PIL import Image

try:
    import pydicom
    HAS_PYDICOM = True
except Exception:
    HAS_PYDICOM = False

# --------- 유틸 ---------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
DICOM_EXTS = {".dcm"}

def is_dicom(path: Path) -> bool:
    if path.suffix.lower() in DICOM_EXTS:
        return True
    # 확장자 없거나 기타일 때 DICOM 마커 확인(빠르게)
    try:
        with open(path, "rb") as f:
            preamble = f.read(0x100)
        return b"DICM" in preamble
    except Exception:
        return False

def load_image_any(path: Path) -> np.ndarray:
    """
    이미지를 [H, W] 또는 [H, W, 3] 의 float32 [0,1] 범위로 반환.
    DICOM이면 rescale 적용, uint16/mono → [0,1] 정규화.
    """
    if HAS_PYDICOM and is_dicom(path):
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        # RescaleSlope/Intercept 적용
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + inter
        # robust normalize (퍼센타일 1~99)
        lo = np.percentile(arr, 1)
        hi = np.percentile(arr, 99)
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = arr - arr.min()
            if arr.max() > 0:
                arr = arr / arr.max()
        return np.clip(arr, 0, 1)
    else:
        # 일반 이미지
        img = Image.open(str(path))
        img = img.convert("L")  # 의료영상 가시성 위해 기본 그레이스케일
        arr = np.array(img).astype(np.float32)
        if arr.max() > 0:
            arr /= 255.0
        return arr

def sorted_images_in(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file()]
    files = [p for p in files if (p.suffix.lower() in IMG_EXTS) or is_dicom(p)]
    # 숫자 인식 정렬(파일명에 포함된 숫자 기준)
    def sort_key(p: Path):
        nums = re.findall(r"\d+", p.stem)
        return (p.stem.lower(), tuple(int(n) for n in nums)) if nums else (p.stem.lower(), ())
    return sorted(files, key=sort_key)

def build_matches(
    ct_list: List[Path],
    b0_list: List[Path],
    b1000_list: List[Path],
    adc_list: List[Path],
    mode: str = "stem"
) -> List[Tuple[Path, Path, Path, Path]]:
    """
    네 폴더의 파일을 1:1 매칭한 튜플 리스트로 반환.
    mode='stem' 이면 파일명 stem 교집합 기준 매칭, 실패 시 index fallback.
    mode='index' 이면 바로 index 매칭(최소 길이).
    """
    if mode not in {"stem", "index"}:
        mode = "stem"

    if mode == "stem":
        dicts = []
        for L in (ct_list, b0_list, b1000_list, adc_list):
            d = {}
            for p in L:
                d[p.stem] = p
            dicts.append(d)
        common = set(dicts[0].keys()) & set(dicts[1].keys()) & set(dicts[2].keys()) & set(dicts[3].keys())
        if len(common) > 0:
            stems = sorted(list(common))
            return [(dicts[0][s], dicts[1][s], dicts[2][s], dicts[3][s]) for s in stems]
        # fallback
    # index 방식
    n = min(len(ct_list), len(b0_list), len(b1000_list), len(adc_list))
    return [(ct_list[i], b0_list[i], b1000_list[i], adc_list[i]) for i in range(n)]

# --------- 데이터 수집 ---------
def collect_cases(root: Path) -> List[Path]:
    """
    root 하위의 CNUH_X_* 폴더들을 수집
    """
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("CNUH_")])

def scan_case(case_dir: Path, match_mode: str) -> List[Tuple[Path, Path, Path, Path]]:
    ct = sorted_images_in(case_dir / "ct_axial")
    b0 = sorted_images_in(case_dir / "mri_b0_axial")
    b1000 = sorted_images_in(case_dir / "mri_b1000_axial")
    adc = sorted_images_in(case_dir / "adc")
    pairs = build_matches(ct, b0, b1000, adc, mode=match_mode)
    return pairs

# --------- 뷰어 ---------
HELP_TEXT = (
    "Keyboard shortcuts\n"
    "A / D : previous / next\n"
    "E     : toggle exclude / include current set\n"
    "S     : save exclusion list to CSV (and copy files if --exclude-dir)\n"
    "H/?   : toggle help\n"
    "Q     : quit (auto-save once before exit)\n"
)

class QCViewer:
    def __init__(
        self,
        root: Path,
        save_csv: Path,
        exclude_dir: Optional[Path],
        match_mode: str = "stem",
        start_case: int = 1,
        start_index: int = 0
    ):
        self.root = root
        self.save_csv_path = save_csv
        self.exclude_dir = exclude_dir
        self.match_mode = match_mode


        self.cases = collect_cases(root)
        if not self.cases:
            print(f"[ERROR] No 'CNUH_X_*' folders under: {root}")
            sys.exit(1)

        # case마다 매칭 리스트
        self.case_pairs: Dict[str, List[Tuple[Path, Path, Path, Path]]] = {}
        for c in self.cases:
            pairs = scan_case(c, match_mode=self.match_mode)
            if not pairs:
                print(f"[WARN] No matched images in case: {c.name}")
            self.case_pairs[c.name] = pairs

        # 전역 인덱스: (case_idx, within_case_idx)
        self.case_idx = start_case
        self.item_idx = start_index

        # 제외 표시 저장: {(case_name, item_idx): True}
        self.excluded = {}
        self.show_help = True  # 도움말 박스 표시 여부

        # UI
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))
        # 여백 조금 남기기 (상단 진행상태, 하단 도움말 박스 공간)
        self.fig.subplots_adjust(top=0.90, bottom=0.07, left=0.03, right=0.98)

        # 하단 오른쪽에 도움말 텍스트(이미지 위가 아니라 figure 좌표에, 항상 최상단 zorder)
        self.help_txt = self.fig.text(
            0.99, 0.01, HELP_TEXT,
            ha="right", va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
            visible=self.show_help,
            zorder=1000,
        )

        self.fig.canvas.manager.set_window_title("4-UP QC Viewer")
        # self.fig.tight_layout()
        self.show_help = True

        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.refresh()

    def current_case(self) -> str:
        return self.cases[self.case_idx].name

    def current_pairs(self) -> List[Tuple[Path, Path, Path, Path]]:
        return self.case_pairs.get(self.current_case(), [])

    def current_tuple(self) -> Optional[Tuple[Path, Path, Path, Path]]:
        pairs = self.current_pairs()
        if not pairs:
            return None
        if self.item_idx < 0 or self.item_idx >= len(pairs):
            return None
        return pairs[self.item_idx]

    def draw_4up(self, paths: Tuple[Path, Path, Path, Path]):
        (ax_ct, ax_b0), (ax_b1000, ax_adc) = self.axes
        for ax in [ax_ct, ax_b0, ax_b1000, ax_adc]:
            ax.clear()

        # 폴더 라벨만 (파일명 제거)
        labels = [
            ("ct_axial",      paths[0]),
            ("mri_b0_axial",  paths[1]),
            ("mri_b1000_axial", paths[2]),
            ("adc",           paths[3]),
        ]
        axes_list = [ax_ct, ax_b0, ax_b1000, ax_adc]

        for ax, (folder_name, p) in zip(axes_list, labels):
            try:
                img = load_image_any(p)
            except Exception as e:
                img = None
                print(f"[ERROR] Failed to load {p}: {e}")

            ax.clear()
            if img is None:
                ax.text(0.5, 0.5, "Load failed", ha="center", va="center")
                ax.axis("off")
            else:
                # draw image
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(img)
                ax.axis("off")

            # 좌상단에 폴더명만 작은 배지로 표시
            ax.text(
                0.01, 0.02, folder_name,
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none")
            )


        case_name = self.current_case()
        idx = self.item_idx
        total = len(self.current_pairs())
        is_ex = self.excluded.get((case_name, idx), False)
        state = "EXCLUDED" if is_ex else "OK"

        # 진행 상태 한 번만 상단에
        sup = f"[{case_name}]  {idx+1}/{total}  |  State: {state}"
        self.fig.suptitle(sup, fontsize=12)

        # 제외 상태면 빨간 테두리(유지)
        if is_ex:
            for ax in axes_list:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3.0)

        # 도움말 텍스트(하단 박스) 가시성은 on_key에서 제어
        self.help_txt.set_visible(self.show_help)
        self.fig.canvas.draw_idle()


    def move(self, delta: int):
        pairs = self.current_pairs()
        # 현재 케이스가 비어있으면 바로 다음/이전 케이스로 이동
        if not pairs:
            self.next_case_with_items(+1 if delta > 0 else -1, at=("start" if delta > 0 else "end"))
            return

        new_idx = self.item_idx + delta
        if new_idx < 0:
            # 이전 케이스의 마지막으로
            self.next_case_with_items(-1, at="end")
        elif new_idx >= len(pairs):
            # 다음 케이스의 처음으로
            self.next_case_with_items(+1, at="start")
        else:
            self.item_idx = new_idx
            self.refresh()

    def next_case_with_items(self, delta_case: int, at: str = "start"):
        n = len(self.cases)
        for step in range(1, n + 1):
            new_idx = (self.case_idx + delta_case * step) % n
            pairs = self.case_pairs.get(self.cases[new_idx].name, [])
            if pairs:
                self.case_idx = new_idx
                self.item_idx = 0 if at == "start" else len(pairs) - 1
                self.refresh()
                return
        # 모두 비어있다면 변화 없음

    def toggle_exclude(self):
        case_name = self.current_case()
        idx = self.item_idx
        key = (case_name, idx)
        self.excluded[key] = not self.excluded.get(key, False)
        self.refresh()

    def save_csv(self):
        # 현재 표시 상태 기준으로 제외된 세트만 수집
        rows = []
        for case_name, pairs in self.case_pairs.items():
            for i, t in enumerate(pairs):
                if self.excluded.get((case_name, i), False):
                    ct, b0, b1000, adc = t
                    rows.append([case_name, i, str(ct), str(b0), str(b1000), str(adc)])

        out = self.save_csv_path  # ← 경로 속성
        out.parent.mkdir(parents=True, exist_ok=True)
        file_exists = out.exists()

        # 존재하면 append, 없으면 새로 만들고 헤더 씀
        with open(out, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["case", "index", "ct_axial", "mri_b0_axial", "mri_b1000_axial", "adc"])
            w.writerows(rows)

        print(f"[INFO] {'Appended to' if file_exists else 'Created'} CSV: {out} (+{len(rows)} rows)")

        # 선택적 복사 (이번에 기록한 rows만 복사)
        if self.exclude_dir and rows:
            for _, _, ct, b0, b1000, adc in rows:
                for p in [ct, b0, b1000, adc]:
                    src = Path(p)
                    # root 기준 상대 경로 유지하여 복사
                    try:
                        rel = src.relative_to(self.root)
                    except ValueError:
                        rel = src.name  # root 밖이면 파일명만
                    dst = self.exclude_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"[WARN] Copy failed: {src} -> {dst} ({e})")
            print(f"[INFO] Copied excluded files under: {self.exclude_dir}")


    def refresh(self):
        cur = self.current_tuple()
        (ax_ct, ax_b0), (ax_b1000, ax_adc) = self.axes
        if cur is None:
            self.axes.text(0.5, 0.5, "No images", ha="center", va="center")
            self.axes.axis("off")
            self.fig.suptitle(f"[{self.current_case()}]  No matched images", fontsize=12)
            self.help_txt.set_visible(self.show_help)
        self.draw_4up(cur)

    def on_key(self, event):
        if event.key in ("right", "d"):
            self.move(+1)
        elif event.key in ("left", "a"):
            self.move(-1)
        elif event.key in ("e",):
            self.toggle_exclude()
        elif event.key in ("s",):
            self.save_csv()
        elif event.key in ("h", "?"):
            self.show_help = not self.show_help
            self.help_txt.set_visible(self.show_help)  # ← 추가
            self.refresh()
        elif event.key in ("q",):
            # 종료 전 자동 저장 1회
            self.save_csv()
            plt.close(self.fig)

def parse_args():
    ap = argparse.ArgumentParser(description="4-UP QC Viewer for CT/MRI alignment check")
    ap.add_argument("--root", type=Path, required=True, help="Root directory containing CNUH_X_* folders")
    ap.add_argument("--save-csv", type=Path, default=Path("exclusions.csv"), help="CSV to save excluded sets")
    ap.add_argument("--exclude-dir", type=Path, default=None, help="Copy excluded files under this directory on save")
    ap.add_argument("--match-mode", type=str, default="stem", choices=["stem", "index"],
                    help="How to match files across the 4 folders")
    ap.add_argument("--start_case", type=int, default=0, help="Start from this case index (0-based)")
    ap.add_argument("--start_index", type=int, default=0, help="Start from this index (0-based)")
    return ap.parse_args()

def main():
    args = parse_args()
    viewer = QCViewer(
        root=args.root,
        save_csv=args.save_csv,
        exclude_dir=args.exclude_dir,
        match_mode=args.match_mode,
        start_case=args.start_case,
        start_index=args.start_index
    )
    print(HELP_TEXT)
    plt.show()

if __name__ == "__main__":
    main()
