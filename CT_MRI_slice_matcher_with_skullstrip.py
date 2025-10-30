#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

사용법:
python CT_MRI_slice_matcher_with_skullstrip.py \
  --ct_root CT/ct_microbleed_0 \
  --mr_root MRI/mri_microbleed_0/250218_0 \
  --out_root ./skull_strip/no_bleed_ct_mri \
  --skull_strip_method hdbet \
  --mr_norm dicom \
  --b0_ww_adjust 1.8

python CT_MRI_slice_matcher_with_skullstrip.py \
  --ct_root CT/ct_microbleed_1 \
  --mr_root MRI/mri_microbleed_1/250217_1 \
  --out_root ./skull_strip/bleed_ct_mri \
  --skull_strip_method hdbet \
  --mr_norm dicom \
  --b0_ww_adjust 1.8

threshold 방법 사용 예:

python CT_MRI_slice_matcher_with_skullstrip.py \
  --ct_root CT/ct_microbleed_0 \
  --mr_root MRI/mri_microbleed_0/250218_0 \
  --out_root ./skull_strip/no_bleed_ct_mri \
  --skull_strip_method threshold \
  --mr_norm dicom \
  --b0_ww_adjust 1.8

  python CT_MRI_slice_matcher_with_skullstrip.py \
  --ct_root CT/ct_microbleed_1 \
  --mr_root MRI/mri_microbleed_1/250217_1 \
  --out_root ./skull_strip/bleed_ct_mri \
  --skull_strip_method threshold \
  --mr_norm dicom \
  --b0_ww_adjust 1.8

python CT_MRI_slice_matcher_with_skullstrip.py \
  --ct_root CT/ct_microbleed_0 \
  --mr_root MRI/mri_microbleed_0/250218_0 \
  --out_root ./skull_strip/no_bleed_ct_mri \
  --skull_strip_method threshold \
  --ct_skull_threshold_low 15 \
  --ct_skull_threshold_high 45 \
  --ct_skull_erosion 8 \
  --mr_norm dicom \
  --b0_ww_adjust 1.8

python CT_MRI_slice_matcher_with_skullstrip.py \
    --ct_root CT/ct_microbleed_1 \
    --mr_root MRI/mri_microbleed_1/250217_1 \
    --out_root ./skull_strip/bleed_ct_mri \
    --skull_strip_method threshold \
    --ct_skull_threshold_low 15 \
    --ct_skull_threshold_high 45 \
    --ct_skull_erosion 8 \
    --mr_norm dicom \
    --b0_ww_adjust 1.8
  
python CT_MRI_slice_matcher_with_skullstrip.py \
  --ct_root CT/ct_microbleed_0 \
  --mr_root MRI/mri_microbleed_0/250218_0 \
  --out_root ./skull_strip/no_bleed_ct_mri \
  --skull_strip_method freesurfer \
  --mr_norm dicom \
  --b0_ww_adjust 1.8
  
"""



"""
CT_MRI_slice_matcher_with_skullstrip.py

개선사항:
- 정합 전 skull stripping 추가
- CT와 MRI 모두에서 skull 제거 후 정합
- 정합 후 원본 이미지를 리샘플하되, brain mask 적용
- 기존의 데이터 입/출력 및 정합 로직 유지

"""

import os, sys, re, json, csv, argparse, logging
from typing import List, Dict, Any, Tuple, Optional, NamedTuple

import numpy as np
import SimpleITK as sitk

# ---------------- image writer backend ----------------
_SAVE_BACKEND = None
try:
    import cv2
    _SAVE_BACKEND = "cv2"
except Exception:
    try:
        import imageio
        _SAVE_BACKEND = "imageio"
    except Exception:
        _SAVE_BACKEND = None

# ---------------- pydicom ----------------
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut as _apply_voi_lut
    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False

# ---------------- HD-BET (optional) ----------------
_HAS_HDBET = False
_HDBET_ERROR = None

# Add HD-BET to path if it exists locally
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
possible_hdbet_paths = [
    os.path.join(current_dir, "HD-BET"),
    os.path.join(os.path.dirname(current_dir), "HD-BET"),
    os.path.expanduser("~/HD-BET"),
    "./HD-BET",
    "../HD-BET",
]

for path in possible_hdbet_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        
try:
    # Try multiple import methods
    try:
        from HD_BET.hd_bet_prediction import hdbet_predict, get_hdbet_predictor
        _HAS_HDBET = True
    except ImportError as e1:
        try:
            from hd_bet.hd_bet_prediction import hdbet_predict, get_hdbet_predictor
            _HAS_HDBET = True
        except ImportError as e2:
            try:
                import HD_BET
                from HD_BET.hd_bet_prediction import hdbet_predict, get_hdbet_predictor
                _HAS_HDBET = True
            except ImportError as e3:
                _HDBET_ERROR = f"Import failed: {e3}"
except Exception as e:
    _HDBET_ERROR = str(e)


# ============================== DEFAULTS ==============================
DEFAULTS = {
    "ct_root": "CT/ct_microbleed_0",
    "mr_root": "MRI/mri_microbleed_0/250218_0",
    "out_root": "./no_bleed_ct_mri_final",
    "ct_window_center": 40.0,
    "ct_window_width": 80.0,
    "loglevel": "INFO",
    "start_from": "",
    "only": "",
    "skip": "",
    "mr_norm": "robust",
    "mr_p_low": 1.0,
    "mr_p_high": 99.0,
    "b0_wc_adjust": 0.0,
    "b0_ww_adjust": 1.0,
    "b1000_wc_adjust": 0.0,
    "b1000_ww_adjust": 1.0,
    "adc_wc_adjust": 0.0,
    "adc_ww_adjust": 1.0,
    # Skull stripping 관련
    "skull_strip_method": "threshold",  # hdbet, threshold, none
    "skull_strip_save_mask": True,
}


# ============================== UTILS (기존 유지) ==============================
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def save_gray(path: str, img_uint8: np.ndarray):
    """Save grayscale image as PNG"""
    if _SAVE_BACKEND == "cv2":
        if not cv2.imwrite(path, img_uint8):
            raise RuntimeError(f"Failed to write image: {path}")
    elif _SAVE_BACKEND == "imageio":
        imageio.imwrite(path, img_uint8)
    else:
        raise RuntimeError("No image backend (cv2/imageio) available.")

def percentile_norm(arr: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo = np.percentile(arr, p_low); hi = np.percentile(arr, p_high)
    if hi <= lo: hi = lo + 1e-5
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo + 1e-12) * 255.0).round().astype(np.uint8)

def robust_percentile_norm(arr: np.ndarray, p_low=2.0, p_high=98.0) -> np.ndarray:
    mask = arr > 0
    if np.sum(mask) < arr.size * 0.1:
        mask = np.ones_like(arr, dtype=bool)
    if np.sum(mask) == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    masked_arr = arr[mask]
    lo = np.percentile(masked_arr, p_low)
    hi = np.percentile(masked_arr, p_high)
    if hi <= lo: 
        hi = lo + np.std(masked_arr) + 1e-5
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo + 1e-12) * 255.0).round().astype(np.uint8)

def ct_window_to_uint8(arr_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    if width <= 0: width = 1.0
    lo, hi = center - width/2.0, center + width/2.0
    arr = np.clip(arr_hu, lo, hi)
    return ((arr - lo) / (hi - lo + 1e-12) * 255.0).round().astype(np.uint8)

def ct_window_to_uint8_stack(arr_z_y_x: np.ndarray, c: float, w: float) -> np.ndarray:
    out = np.empty_like(arr_z_y_x, dtype=np.uint8)
    for k in range(arr_z_y_x.shape[0]): out[k] = ct_window_to_uint8(arr_z_y_x[k], c, w)
    return out

def log_image_info(tag: str, img: sitk.Image):
    logging.info(f"[{tag}] size={img.GetSize()}, spacing={img.GetSpacing()}, origin={img.GetOrigin()}")


# ============================== SKULL STRIPPING ==============================

def skull_strip_freesurfer(img: sitk.Image, modality: str = "ct") -> Tuple[sitk.Image, sitk.Image]:
    """
    FreeSurfer mri_synthstrip을 사용한 skull stripping
    Returns: (brain_img, mask_img)
    """
    import subprocess
    import tempfile
    
    logging.info("Using FreeSurfer SynthStrip for skull stripping...")
    
    # FreeSurfer 설치 확인
    try:
        result = subprocess.run(['mri_synthstrip', '--help'], 
                              capture_output=True, timeout=5)
        if result.returncode != 0:
            raise RuntimeError("mri_synthstrip not found")
    except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
        raise RuntimeError(
            "FreeSurfer mri_synthstrip not available.\n"
            "Install FreeSurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall\n"
            "Or use --skull_strip_method threshold"
        )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.nii.gz")
        output_path = os.path.join(tmpdir, "output.nii.gz")
        mask_path = os.path.join(tmpdir, "mask.nii.gz")
        
        # NIfTI로 저장
        sitk.WriteImage(img, input_path)
        
        # mri_synthstrip 실행
        try:
            cmd = [
                'mri_synthstrip',
                '-i', input_path,
                '-o', output_path,
                '-m', mask_path,
            ]
            
            if modality.lower() == "ct":
                # CT의 경우 추가 옵션 (있다면)
                pass
            
            logging.info(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logging.error(f"FreeSurfer error: {result.stderr}")
                raise RuntimeError(f"mri_synthstrip failed: {result.stderr}")
            
            # 결과 로드
            brain_img = sitk.ReadImage(output_path)
            mask_img = sitk.ReadImage(mask_path)
            
            # 마스크 통계
            mask_arr = sitk.GetArrayFromImage(mask_img).astype(bool)
            coverage = mask_arr.sum() / mask_arr.size * 100
            logging.info(f"  FreeSurfer mask coverage: {coverage:.1f}%")
            
            return brain_img, mask_img
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("FreeSurfer mri_synthstrip timed out (>5min)")
        except Exception as e:
            logging.error(f"FreeSurfer execution failed: {e}")
            raise


def skull_strip_threshold_ct(ct_img: sitk.Image, 
                              lower_threshold: float = 10.0,
                              upper_threshold: float = 60.0,
                              erosion_size: int = 5) -> Tuple[sitk.Image, sitk.Image]:
    """
    개선된 임계값 기반 CT skull stripping
    뇌 조직: 약 20-40 HU (회백질), 30-40 HU (백질)
    
    Args:
        lower_threshold: 최소 HU (기본 10)
        upper_threshold: 최대 HU (기본 60) - 더 낮추면 더 공격적
        erosion_size: erosion 크기 (기본 5) - 더 크면 더 공격적
    
    Returns: (brain_img, mask_img)
    """
    logging.info("Using threshold-based skull stripping for CT...")
    
    arr = sitk.GetArrayFromImage(ct_img)
    logging.info(f"  CT HU range: {arr.min():.1f} ~ {arr.max():.1f}")
    
    # 1. Brain tissue 임계값
    # Skull은 보통 300+ HU이므로 제외됨
    brain_mask = (arr >= lower_threshold) & (arr <= upper_threshold)
    
    logging.info(f"  Threshold: {lower_threshold} ~ {upper_threshold} HU")
    logging.info(f"  Initial mask coverage: {brain_mask.sum() / brain_mask.size * 100:.1f}%")
    
    # 2. SimpleITK 이미지로 변환
    mask_img = sitk.GetImageFromArray(brain_mask.astype(np.uint8))
    mask_img.CopyInformation(ct_img)
    
    # 3. Fill holes first (뇌 내부 홀 채우기)
    mask_img = sitk.BinaryMorphologicalClosing(mask_img, [15, 15, 5])
    
    # 4. Remove small objects
    mask_img = sitk.BinaryMorphologicalOpening(mask_img, [3, 3, 1])
    
    # 5. 가장 큰 연결 성분만 유지 (뇌)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_img = cc_filter.Execute(mask_img)
    
    num_components = cc_filter.GetObjectCount()
    logging.info(f"  Found {num_components} connected components")
    
    if num_components > 0:
        stats_filter = sitk.LabelShapeStatisticsImageFilter()
        stats_filter.Execute(cc_img)
        labels = stats_filter.GetLabels()
        
        if labels:
            # 크기와 위치를 고려하여 뇌 선택
            img_size = cc_img.GetSize()
            img_center = [img_size[0]//2, img_size[1]//2, img_size[2]//2]
            
            best_label = None
            best_score = -1
            
            for label in labels:
                size = stats_filter.GetNumberOfPixels(label)
                centroid = stats_filter.GetCentroid(label)
                
                # 중앙으로부터의 거리
                dist = np.sqrt(sum((centroid[i] - img_center[i])**2 for i in range(3)))
                
                # 점수: 크기 우선, 중앙 근접도 가산
                size_score = size
                center_score = 1000 / (dist + 1)
                total_score = size_score + center_score * 1000
                
                if total_score > best_score:
                    best_score = total_score
                    best_label = label
            
            if best_label:
                mask_img = (cc_img == best_label)
                logging.info(f"  Kept component (label {best_label})")
    
    # 6. Aggressive erosion to remove skull edges
    logging.info(f"  Applying erosion with size {erosion_size}")
    mask_img = sitk.BinaryErode(mask_img, [erosion_size, erosion_size, max(2, erosion_size//2)])
    
    # 7. Smooth boundaries
    mask_img = sitk.BinaryMorphologicalClosing(mask_img, [8, 8, 3])
    
    # 8. Dilate back slightly (but not to original size)
    dilate_size = max(2, erosion_size - 2)
    mask_img = sitk.BinaryDilate(mask_img, [dilate_size, dilate_size, 1])
    
    # 9. Final smoothing
    mask_img = sitk.BinaryMorphologicalOpening(mask_img, [2, 2, 1])
    
    # 10. 최종 마스크 확인
    final_mask_arr = sitk.GetArrayFromImage(mask_img).astype(bool)
    logging.info(f"  Final mask coverage: {final_mask_arr.sum() / final_mask_arr.size * 100:.1f}%")
    
    # 11. 마스크 적용
    brain_arr = arr.copy()
    brain_arr[~final_mask_arr] = -1024  # 배경을 공기 HU로
    
    brain_img = sitk.GetImageFromArray(brain_arr)
    brain_img.CopyInformation(ct_img)
    
    return brain_img, mask_img


def skull_strip_threshold_mri(mri_img: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
    """
    개선된 임계값 기반 MRI skull stripping
    Returns: (brain_img, mask_img)
    """
    logging.info("Using threshold-based skull stripping for MRI...")
    
    arr = sitk.GetArrayFromImage(mri_img)
    logging.info(f"  MRI intensity range: {arr.min():.1f} ~ {arr.max():.1f}")
    
    # 1. Otsu thresholding으로 초기 분리
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mask_img = otsu_filter.Execute(mri_img)
    
    threshold_value = otsu_filter.GetThreshold()
    logging.info(f"  Otsu threshold: {threshold_value:.1f}")
    
    mask_arr = sitk.GetArrayFromImage(mask_img).astype(bool)
    logging.info(f"  Initial mask coverage: {mask_arr.sum() / mask_arr.size * 100:.1f}%")
    
    # 2. Fill holes
    mask_img = sitk.BinaryMorphologicalClosing(mask_img, [15, 15, 5])
    
    # 3. Remove small objects
    mask_img = sitk.BinaryMorphologicalOpening(mask_img, [3, 3, 1])
    
    # 4. 가장 큰 연결 성분 유지
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_img = cc_filter.Execute(mask_img)
    
    num_components = cc_filter.GetObjectCount()
    logging.info(f"  Found {num_components} connected components")
    
    if num_components > 0:
        stats_filter = sitk.LabelShapeStatisticsImageFilter()
        stats_filter.Execute(cc_img)
        labels = stats_filter.GetLabels()
        
        if labels:
            # 중앙에 있고 적절한 크기의 component 선택
            img_size = cc_img.GetSize()
            img_center = [img_size[0]//2, img_size[1]//2, img_size[2]//2]
            
            best_label = None
            best_score = -1
            
            for label in labels:
                size = stats_filter.GetNumberOfPixels(label)
                centroid = stats_filter.GetCentroid(label)
                
                # 중앙으로부터의 거리
                dist = np.sqrt(sum((centroid[i] - img_center[i])**2 for i in range(3)))
                
                # 점수: 크기 우선, 중앙 근접도 가산
                size_score = size
                center_score = 1000 / (dist + 1)
                total_score = size_score + center_score * 1000
                
                if total_score > best_score:
                    best_score = total_score
                    best_label = label
            
            if best_label:
                mask_img = (cc_img == best_label)
                logging.info(f"  Kept component (label {best_label})")
    
    # 5. Aggressive erosion to remove skull/scalp
    mask_img = sitk.BinaryErode(mask_img, [6, 6, 2])
    
    # 6. Smooth boundaries
    mask_img = sitk.BinaryMorphologicalClosing(mask_img, [8, 8, 3])
    
    # 7. Dilate back slightly
    mask_img = sitk.BinaryDilate(mask_img, [4, 4, 1])
    
    # 8. Final smoothing
    mask_img = sitk.BinaryMorphologicalOpening(mask_img, [2, 2, 1])
    
    # 9. 최종 마스크 확인
    final_mask_arr = sitk.GetArrayFromImage(mask_img).astype(bool)
    logging.info(f"  Final mask coverage: {final_mask_arr.sum() / final_mask_arr.size * 100:.1f}%")
    
    # 10. 마스크 적용
    brain_arr = arr.copy()
    brain_arr[~final_mask_arr] = 0
    
    brain_img = sitk.GetImageFromArray(brain_arr)
    brain_img.CopyInformation(mri_img)
    
    return brain_img, mask_img


def skull_strip_hdbet(img: sitk.Image, device: int = 0, mode: str = "fast") -> Tuple[sitk.Image, sitk.Image]:
    """
    HD-BET을 사용한 skull stripping
    Returns: (brain_img, mask_img)
    """
    if not _HAS_HDBET:
        error_msg = "HD-BET not available.\n"
        if _HDBET_ERROR:
            error_msg += f"Error: {_HDBET_ERROR}\n"
        error_msg += "\nTroubleshooting:\n"
        error_msg += "1. Check installation: python -c \"import HD_BET; print(HD_BET.__file__)\"\n"
        error_msg += "2. Try: pip install --upgrade HD-BET\n"
        error_msg += "3. Or install from source:\n"
        error_msg += "   git clone https://github.com/MIC-DKFZ/HD-BET.git\n"
        error_msg += "   cd HD-BET && pip install -e .\n"
        error_msg += "4. Restart your Python kernel/terminal after installation\n"
        error_msg += "\nAlternatively, use --skull_strip_method threshold"
        raise RuntimeError(error_msg)
    
    logging.info(f"Using HD-BET for skull stripping...")
    
    import tempfile
    import torch
    
    # device 설정
    if device >= 0 and torch.cuda.is_available():
        torch_device = torch.device(f'cuda:{device}')
        logging.info(f"  Using GPU device: {device}")
    else:
        torch_device = torch.device('cpu')
        logging.info(f"  Using CPU")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.nii.gz")
        output_path = os.path.join(tmpdir, "output.nii.gz")
        
        # 저장
        sitk.WriteImage(img, input_path)
        
        # HD-BET 실행
        try:
            logging.info(f"  Initializing HD-BET predictor...")
            
            # predictor 생성
            use_tta = (mode == "accurate")  # accurate 모드면 TTA 사용
            predictor = get_hdbet_predictor(
                use_tta=use_tta,
                device=torch_device,
                verbose=False
            )
            
            logging.info(f"  Running hdbet_predict on {input_path}")
            
            # hdbet_predict 함수 호출
            hdbet_predict(
                input_path,
                output_path,
                predictor,
                keep_brain_mask=True,  # 마스크 저장
                compute_brain_extracted_image=True  # brain 이미지 생성
            )
            
            # 결과 파일 확인
            output_dir = os.path.dirname(output_path)
            generated_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
            logging.info(f"  Generated files: {generated_files}")
            
            # 출력 파일명 패턴
            # output.nii.gz (brain extracted image)
            # output_bet.nii.gz (brain mask)
            brain_file = output_path
            mask_file = output_path[:-7] + '_bet.nii.gz'
            
            # brain 이미지 로드
            if not os.path.exists(brain_file):
                logging.error(f"HD-BET output not found: {brain_file}")
                logging.error(f"Available files: {generated_files}")
                raise RuntimeError("HD-BET did not produce expected output")
            
            logging.info(f"  Found brain image: {os.path.basename(brain_file)}")
            brain_img = sitk.ReadImage(brain_file)
            
            # mask 이미지 로드
            if os.path.exists(mask_file):
                logging.info(f"  Found mask image: {os.path.basename(mask_file)}")
                mask_img = sitk.ReadImage(mask_file)
            else:
                # 마스크가 없으면 brain 이미지로부터 생성
                logging.warning("HD-BET mask not found, creating from brain output")
                brain_arr = sitk.GetArrayFromImage(brain_img)
                orig_arr = sitk.GetArrayFromImage(img)
                
                # brain 이미지가 이미 마스킹되어 있는 경우 (값이 0인 부분이 마스크 외부)
                mask_arr = (brain_arr != 0).astype(np.uint8)
                
                # 만약 brain과 원본이 거의 같다면 (마스킹이 안된 경우), threshold 사용
                if np.allclose(brain_arr, orig_arr, rtol=0.01):
                    logging.warning("  Brain image not masked, using threshold to create mask")
                    mask_arr = (orig_arr > orig_arr.mean() * 0.1).astype(np.uint8)
                
                mask_img = sitk.GetImageFromArray(mask_arr)
                mask_img.CopyInformation(img)
            
            # 마스크 통계
            mask_arr = sitk.GetArrayFromImage(mask_img).astype(bool)
            coverage = mask_arr.sum() / mask_arr.size * 100
            logging.info(f"  HD-BET mask coverage: {coverage:.1f}%")
            
            return brain_img, mask_img
            
        except Exception as e:
            logging.error(f"HD-BET execution failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise


def apply_skull_stripping(img: sitk.Image, 
                          method: str = "threshold",
                          modality: str = "ct",
                          **kwargs) -> Tuple[sitk.Image, sitk.Image]:
    """
    통합 skull stripping 함수
    
    Args:
        img: 입력 이미지
        method: 'freesurfer', 'hdbet', 'threshold', 'none'
        modality: 'ct' or 'mri'
        **kwargs: threshold parameters (lower_threshold, upper_threshold, erosion_size)
    
    Returns:
        (brain_img, mask_img)
    """
    if method == "none":
        # Skull stripping 안함 - 전체를 1로 채운 마스크 반환
        mask_arr = np.ones(sitk.GetArrayFromImage(img).shape, dtype=np.uint8)
        mask_img = sitk.GetImageFromArray(mask_arr)
        mask_img.CopyInformation(img)
        return img, mask_img
    
    elif method == "freesurfer":
        return skull_strip_freesurfer(img, modality)
    
    elif method == "hdbet":
        return skull_strip_hdbet(img)
    
    elif method == "threshold":
        if modality.lower() == "ct":
            return skull_strip_threshold_ct(img, **kwargs)
        else:  # mri
            return skull_strip_threshold_mri(img)
    
    else:
        raise ValueError(f"Unknown skull strip method: {method}")


# ============================== SERIES HELPERS (기존 유지) ==============================

def list_series(root: str) -> List[str]:
    reader = sitk.ImageSeriesReader()
    sids = reader.GetGDCMSeriesIDs(root)
    return list(sids) if sids is not None else []

def series_files(root: str, sid: str) -> List[str]:
    reader = sitk.ImageSeriesReader()
    return reader.GetGDCMSeriesFileNames(root, sid)

def read_series_as_image(files: List[str]) -> sitk.Image:
    r = sitk.ImageSeriesReader(); r.SetFileNames(files); return r.Execute()

def list_immediate_patient_dirs(root: str) -> List[str]:
    return sorted([os.path.join(root, d) for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))])

def pair_patients(ct_root: str, mr_root: str) -> List[Tuple[str,str,str]]:
    ct_dirs = list_immediate_patient_dirs(ct_root)
    mr_dirs = list_immediate_patient_dirs(mr_root)
    ct_map = {os.path.basename(os.path.normpath(d)): d for d in ct_dirs}
    mr_map = {os.path.basename(os.path.normpath(d)): d for d in mr_dirs}
    shared = sorted(set(ct_map.keys()) & set(mr_map.keys()))
    return [(pid, ct_map[pid], mr_map[pid]) for pid in shared]

def pid_key(pid: str):
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', pid))


# ============================== DICOM META / DWI DETECTION (기존 유지) ==============================
BVAL_TAGS = [(0x0018,0x9087), (0x0019,0x100C), (0x0043,0x1039)]
B0_KEYS     = ["B=0","B0","B 0","D=0","D0","TRACE 0","TRACE B0","NO DIFF","BASELINE","APPARENT 0"]
B1000_KEYS  = ["B=1000","B1000","B 1000","1000","HIGH B","TRACE 1000"]
ADC_KEYS    = ["ADC","ADC MAP","ADCMAP","APPARENT DIFFUSION COEFFICIENT","ADC-MAP"]

def _get_first_float(v):
    try:
        if isinstance(v, (list, tuple)):
            return float(v[0])
        s = str(v)
        if "\\" in s:
            return float(s.split("\\")[0])
        return float(s)
    except:
        return None

def read_series_meta(file_path: str) -> Dict[str,Any]:
    meta={"SeriesDescription":"","ProtocolName":"","ImageType":"","bvals":[],
          "PhotometricInterpretation": "",
          "WindowCenter": None, "WindowWidth": None,
          "RescaleSlope": None, "RescaleIntercept": None,
          "HasVOILUT": False,
    }
    if not _HAS_PYDICOM: return meta
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=False, force=True)
        def _get(name:str):
            try:
                v = getattr(ds,name,"")
                if isinstance(v,(list,tuple)): return "\\".join([str(x) for x in v])
                return str(v) if v is not None else ""
            except: return ""
        meta["SeriesDescription"] = _get("SeriesDescription")
        meta["ProtocolName"]     = _get("ProtocolName")
        meta["ImageType"]        = _get("ImageType")

        try: meta["PhotometricInterpretation"] = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        except: pass
        try: meta["RescaleSlope"] = float(getattr(ds, "RescaleSlope", 1.0))
        except: meta["RescaleSlope"] = 1.0
        try: meta["RescaleIntercept"] = float(getattr(ds, "RescaleIntercept", 0.0))
        except: meta["RescaleIntercept"] = 0.0
        try:
            wc = getattr(ds, "WindowCenter", None)
            ww = getattr(ds, "WindowWidth", None)
            meta["WindowCenter"] = _get_first_float(wc)
            meta["WindowWidth"]  = _get_first_float(ww)
        except: pass
        try:
            vl = getattr(ds, "VOILUTSequence", None)
            meta["HasVOILUT"] = bool(vl and len(vl)>0)
        except: meta["HasVOILUT"] = False

        b=[]
        for tag in BVAL_TAGS:
            try:
                if tag in ds:
                    v = ds[tag].value
                    if isinstance(v,(list,tuple)):
                        for x in v:
                            try: b.append(float(x))
                            except: pass
                    else:
                        try: b.append(float(str(v)))
                        except: pass
            except: pass
        try:
            sfg = getattr(ds, "SharedFunctionalGroupsSequence", None)
            if sfg and len(sfg)>0:
                mrseq = getattr(sfg[0], "MRDiffusionSequence", None)
                if mrseq and len(mrseq)>0:
                    v = getattr(mrseq[0], "DiffusionBValue", None)
                    if v is not None: b.append(float(v))
            pfg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
            if pfg:
                for pf in pfg:
                    mrseq = getattr(pf, "MRDiffusionSequence", None)
                    if mrseq and len(mrseq)>0:
                        v = getattr(mrseq[0], "DiffusionBValue", None)
                        if v is not None: b.append(float(v))
        except: pass
        meta["bvals"] = [float(x) for x in b if np.isfinite(x) and x>=0]
    except Exception as e:
        logging.warning(f"pydicom read failed: {file_path}: {e}")
    return meta


def classify_dwi(meta: Dict[str,Any]) -> Dict[str,bool]:
    desc=(meta.get("SeriesDescription") or "").upper()
    prot=(meta.get("ProtocolName") or "").upper()
    it  =(meta.get("ImageType") or "").upper()
    text=f"{desc} | {prot} | {it}"
    bvals=meta.get("bvals",[])
    minb, maxb = (min(bvals), max(bvals)) if bvals else (None,None)
    is_adc = any(k in text for k in ADC_KEYS) or ("ADC" in it)
    is_b0  = (minb is not None and minb<=50) or any(k in text for k in B0_KEYS)
    is_b1k = (maxb is not None and 700<=maxb<=2000) or any(k in text for k in B1000_KEYS)
    return {"is_b0":is_b0,"is_b1000":is_b1k,"is_adc":is_adc}


def detect_dwi_series_recursive(mr_root: str, max_depth:int=2) -> Dict[str,Dict[str,Any]]:
    out={}
    root_norm=os.path.normpath(mr_root); base=root_norm.count(os.sep)
    for r, dnames, _ in os.walk(root_norm):
        if os.path.normpath(r).count(os.sep)-base > max_depth:
            dnames[:]=[]; continue
        sids = list_series(r)
        for sid in sids:
            files = series_files(r, sid)
            if not files: continue
            meta  = read_series_meta(files[0])
            flags = classify_dwi(meta)
            txt=(meta.get("SeriesDescription","")+" "+meta.get("ProtocolName","")).upper()
            sc = 0.01*len(files) + 10*int(flags["is_b0"]) + 10*int(flags["is_b1000"]) + 10*int(flags["is_adc"]) + (2 if any(k in txt for k in ["AX","AXIAL"]) else 0)
            def better(key):
                prev=out.get(key)
                if prev is None: return True
                pmeta=prev["meta"]; pflags=classify_dwi(pmeta)
                ptxt=(pmeta.get("SeriesDescription","")+" "+pmeta.get("ProtocolName","")).upper()
                psc = 0.01*len(prev["files"]) + 10*int(pflags["is_b0"]) + 10*int(pflags["is_b1000"]) + 10*int(pflags["is_adc"]) + (2 if any(k in ptxt for k in ["AX","AXIAL"]) else 0)
                return sc>psc
            if flags["is_b0"] and better("b0"): out["b0"]={"files":files,"meta":meta,"root":r}
            if flags["is_b1000"] and better("b1000"): out["b1000"]={"files":files,"meta":meta,"root":r}
            if flags["is_adc"] and better("adc"): out["adc"]={"files":files,"meta":meta,"root":r}
    
    logging.info("[DWI DEBUG] Series candidates:")
    logging.info("  dir                         sid      files  bmin  bmax   b0  b1000  adc  desc")
    for r, dnames, _ in os.walk(root_norm):
        sids = list_series(r)
        for sid in sids:
            files=series_files(r,sid)
            if not files: continue
            meta = read_series_meta(files[0]); flags=classify_dwi(meta)
            bvals=meta.get("bvals",[]); bmin=f"{min(bvals):.0f}" if bvals else "-"; bmax=f"{max(bvals):.0f}" if bvals else "-"
            desc=(meta.get("SeriesDescription") or "").strip()
            logging.info(f"  {os.path.relpath(r,root_norm)[:26]:26s}  {sid[:8]:8s}  {len(files):5d}  {bmin:>4s}  {bmax:>4s}   "
                         f"{int(flags['is_b0']):1d}    {int(flags['is_b1000']):1d}     {int(flags['is_adc']):1d}  {desc[:60]}")
    return out


# ============================== REGISTRATION & CT-GRID RESAMPLE (기존 유지) ==============================

def orient_to_LPS(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, 'LPS')

def rigid_register_CT_fixed(ct_LPS: sitk.Image, mr_b0_LPS: sitk.Image) -> sitk.Transform:
    """
    수정: skull-stripped 이미지로 정합
    """
    fixed, moving = sitk.Cast(ct_LPS, sitk.sitkFloat32), sitk.Cast(mr_b0_LPS, sitk.sitkFloat32)
    tx0 = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                            sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.2)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(1.0, 200, 1e-6, 10)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4,2,1]); R.SetSmoothingSigmasPerLevel([2,1,0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn(); R.SetInitialTransform(tx0, False)
    logging.info("Running rigid registration (MI) on skull-stripped images...")
    tx = R.Execute(fixed, moving)
    logging.info(f"Registration done. Metric={R.GetMetricValue():.6f}")
    return tx

def resample_to_ct_grid(mr_LPS: sitk.Image, ct_LPS: sitk.Image, tx_mr_to_ct: sitk.Transform) -> sitk.Image:
    return sitk.Resample(mr_LPS, ct_LPS, tx_mr_to_ct, sitk.sitkLinear, 0.0, mr_LPS.GetPixelID())


# ============================== SLICE-LEVEL META COLLECTOR (기존 유지) ==============================

class SliceMeta(NamedTuple):
    z_ip_mm: float
    wc: float | None
    ww: float | None
    slope: float
    intercept: float
    photo: str
    slice_idx: int

def collect_slice_metas(files: list[str]) -> list[SliceMeta]:
    metas = []
    if not _HAS_PYDICOM:
        return []
    for idx, fp in enumerate(files):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            z_ip = None
            try:
                ipp = getattr(ds, "ImagePositionPatient", None)
                if ipp and len(ipp) >= 3:
                    z_ip = float(ipp[2])
            except:
                z_ip = None
            def _first_float(v):
                try:
                    if v is None: return None
                    if isinstance(v,(list,tuple)): return float(v[0])
                    s = str(v); 
                    if "\\" in s: return float(s.split("\\")[0])
                    return float(s)
                except:
                    return None
            wc = _first_float(getattr(ds,"WindowCenter", None))
            ww = _first_float(getattr(ds,"WindowWidth", None))
            slope = float(getattr(ds,"RescaleSlope", 1.0))
            intercept = float(getattr(ds,"RescaleIntercept", 0.0))
            photo = str(getattr(ds,"PhotometricInterpretation","")).upper()
            metas.append(SliceMeta(z_ip_mm=z_ip if z_ip is not None else np.nan,
                                   wc=wc, ww=ww, slope=slope, intercept=intercept, 
                                   photo=photo, slice_idx=idx))
        except Exception as e:
            logging.warning(f"[slice meta] read fail: {fp}: {e}")
            metas.append(SliceMeta(z_ip_mm=np.nan, wc=None, ww=None, slope=1.0, 
                                   intercept=0.0, photo="", slice_idx=idx))
    return metas

def nearest_meta_for_z(z_target_mm: float, orig_img_LPS: sitk.Image, metas: list[SliceMeta]) -> SliceMeta:
    size = orig_img_LPS.GetSize()
    cx, cy = size[0]//2, size[1]//2
    zs = [float(orig_img_LPS.TransformIndexToPhysicalPoint([cx,cy,k])[2]) for k in range(size[2])]

    K = min(len(zs), len(metas))
    if K == 0:
        return SliceMeta(np.nan, None, None, 1.0, 0.0, "", 0)

    if len(zs) != len(metas):
        logging.warning(f"Slice count mismatch: MRI={len(zs)}, DICOM files={len(metas)}. Using simple nearest matching.")
        if len(metas) == 0:
            return SliceMeta(np.nan, None, None, 1.0, 0.0, "", 0)
        
        best_idx = 0
        best_dist = float('inf')
        for i, meta in enumerate(metas):
            if not np.isnan(meta.z_ip_mm):
                dist = abs(meta.z_ip_mm - z_target_mm)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        return metas[best_idx] if best_idx < len(metas) else metas[0]

    metas_k = metas[:K]
    order = np.argsort(zs[:K])
    metas_by_z = [metas_k[i] for i in order]
    zs_sorted  = [zs[i] for i in order]
    idx = int(np.argmin([abs(z - z_target_mm) for z in zs_sorted]))
    return metas_by_z[idx]


# ============================== IMPROVED INTENSITY MAPPING (기존 유지) ==============================

def _apply_window_with_adjustment(arr: np.ndarray, 
                                  wc: Optional[float], 
                                  ww: Optional[float],
                                  wc_adjust: float = 0.0,
                                  ww_multiplier: float = 1.0) -> Optional[np.ndarray]:
    if wc is None or ww is None or not np.isfinite(wc) or not np.isfinite(ww) or ww <= 0:
        return None
    if ww > 10000 or abs(wc) > 50000:
        return None
    adjusted_wc = wc + wc_adjust
    adjusted_ww = ww * ww_multiplier
    if adjusted_ww <= 0:
        adjusted_ww = ww
    lo, hi = adjusted_wc - adjusted_ww/2.0, adjusted_wc + adjusted_ww/2.0
    arrc = np.clip(arr, lo, hi)
    return ((arrc - lo) / (hi - lo + 1e-12) * 255.0).round().astype(np.uint8)

def improved_mri_normalize_stack(arr_z_y_x: np.ndarray,
                                meta: Dict[str,Any],
                                norm_mode: str = "robust",
                                p_low: float = 1.0,
                                p_high: float = 99.0,
                                wc_adjust: float = 0.0,
                                ww_multiplier: float = 1.0) -> np.ndarray:
    slope = float(meta.get("RescaleSlope", 1.0))
    intercept = float(meta.get("RescaleIntercept", 0.0))
    
    if slope != 1.0 or intercept != 0.0:
        arr_rescaled = arr_z_y_x.astype(np.float32) * slope + intercept
    else:
        arr_rescaled = arr_z_y_x.astype(np.float32)
    
    if norm_mode == "percentile":
        out = np.empty_like(arr_z_y_x, dtype=np.uint8)
        for k in range(arr_z_y_x.shape[0]):
            out[k] = percentile_norm(arr_rescaled[k], p_low, p_high)
        arr8 = out
    elif norm_mode == "robust":
        arr8 = robust_percentile_norm(arr_rescaled, p_low, p_high)
    else:  # dicom mode
        wc = meta.get("WindowCenter", None)
        ww = meta.get("WindowWidth", None)
        arr8 = _apply_window_with_adjustment(arr_rescaled, wc, ww, wc_adjust, ww_multiplier)
        if arr8 is None:
            logging.info(f"WC/WW failed (wc={wc}, ww={ww}), using robust fallback")
            arr8 = robust_percentile_norm(arr_rescaled, p_low, p_high)

    photo = str(meta.get("PhotometricInterpretation", "")).upper()
    if photo == "MONOCHROME1":
        arr8 = (255 - arr8).astype(np.uint8)
    return arr8

def improved_mri_normalize_stack_slicewise(arr_z_y_x: np.ndarray,
                                          z_list_mm: list[float],
                                          orig_img_LPS: sitk.Image,
                                          slice_metas: list[SliceMeta],
                                          norm_mode: str = "robust",
                                          fallback_p_low: float = 1.0,
                                          fallback_p_high: float = 99.0,
                                          wc_adjust: float = 0.0,
                                          ww_multiplier: float = 1.0) -> np.ndarray:
    out = np.empty_like(arr_z_y_x, dtype=np.uint8)

    if norm_mode == "robust":
        mask = arr_z_y_x > 0
        if np.sum(mask) > arr_z_y_x.size * 0.1:
            masked_arr = arr_z_y_x[mask]
            lo = np.percentile(masked_arr, fallback_p_low)
            hi = np.percentile(masked_arr, fallback_p_high)
            if hi <= lo: 
                hi = lo + np.std(masked_arr) + 1e-5
        else:
            lo = np.percentile(arr_z_y_x, fallback_p_low)
            hi = np.percentile(arr_z_y_x, fallback_p_high)
            if hi <= lo: hi = lo + 1e-5
    else:
        lo = np.percentile(arr_z_y_x, fallback_p_low)
        hi = np.percentile(arr_z_y_x, fallback_p_high)
        if hi <= lo: hi = lo + 1e-5

    for k, zmm in enumerate(z_list_mm):
        meta = nearest_meta_for_z(zmm, orig_img_LPS, slice_metas)
        sl = arr_z_y_x[k].astype(np.float32)
        sl = sl * float(meta.slope) + float(meta.intercept)

        if norm_mode == "percentile":
            sl_lo = np.percentile(sl, fallback_p_low)
            sl_hi = np.percentile(sl, fallback_p_high)
            if sl_hi <= sl_lo: sl_hi = sl_lo + 1e-5
            slc = np.clip(sl, sl_lo, sl_hi)
            u8 = ((slc - sl_lo) / (sl_hi - sl_lo + 1e-12) * 255.0).round().astype(np.uint8)
        elif norm_mode == "dicom":
            if (meta.wc is not None) and (meta.ww is not None) and np.isfinite(meta.ww) and meta.ww > 0:
                if meta.ww <= 10000 and abs(meta.wc) <= 50000:
                    adjusted_wc = float(meta.wc) + wc_adjust
                    adjusted_ww = float(meta.ww) * ww_multiplier
                    if adjusted_ww > 0:
                        lo_w = adjusted_wc - adjusted_ww/2.0
                        hi_w = adjusted_wc + adjusted_ww/2.0
                        slc = np.clip(sl, lo_w, hi_w)
                        u8 = ((slc - lo_w) / (hi_w - lo_w + 1e-12) * 255.0).round().astype(np.uint8)
                    else:
                        slc = np.clip(sl, lo, hi)
                        u8 = ((slc - lo) / (hi - lo + 1e-12) * 255.0).round().astype(np.uint8)
                else:
                    slc = np.clip(sl, lo, hi)
                    u8 = ((slc - lo) / (hi - lo + 1e-12) * 255.0).round().astype(np.uint8)
            else:
                slc = np.clip(sl, lo, hi)
                u8 = ((slc - lo) / (hi - lo + 1e-12) * 255.0).round().astype(np.uint8)
        else:  # robust mode
            slc = np.clip(sl, lo, hi)
            u8 = ((slc - lo) / (hi - lo + 1e-12) * 255.0).round().astype(np.uint8)

        if str(meta.photo).upper() == "MONOCHROME1":
            u8 = (255 - u8).astype(np.uint8)

        out[k] = u8
    return out


# ============================== SAVE SLICES (CT GRID) - 수정 ==============================

def save_ct_axial(ct_LPS: sitk.Image, 
                 ct_mask: sitk.Image,
                 out_dir: str, 
                 wc: float, 
                 ww: float) -> List[Dict[str,Any]]:
    """
    수정: 마스크 적용하여 PNG로 저장
    """
    ensure_dir(out_dir)
    arr = sitk.GetArrayFromImage(ct_LPS)
    mask_arr = sitk.GetArrayFromImage(ct_mask).astype(bool)
    
    logging.info(f"  CT array shape: {arr.shape}, mask shape: {mask_arr.shape}")
    logging.info(f"  Mask coverage: {mask_arr.sum() / mask_arr.size * 100:.1f}%")
    
    # 마스크 적용 (배경을 HU -1000으로)
    arr_masked = arr.copy()
    arr_masked[~mask_arr] = -1000
    
    # Window/Level 적용하여 uint8로 변환
    arr8 = ct_window_to_uint8_stack(arr_masked, wc, ww)
    
    # 마스크 다시 적용 (배경을 검은색으로)
    for k in range(arr8.shape[0]):
        arr8[k][~mask_arr[k]] = 0
    
    # 첫 번째 슬라이스 확인
    logging.info(f"  First slice - masked pixels: {(~mask_arr[arr8.shape[0]//2]).sum()} / {mask_arr[0].size}")
    
    size, spacing = ct_LPS.GetSize(), ct_LPS.GetSpacing()
    cx, cy = size[0]//2, size[1]//2
    info=[]
    for k in range(size[2]):
        z = float(ct_LPS.TransformIndexToPhysicalPoint([cx,cy,k])[2])
        fn = f"{k:04d}.png"  # PNG로 변경
        save_gray(os.path.join(out_dir, fn), arr8[k])
        info.append({"slice_idx":k, "z_mm":z, "filename":fn, "spacing_z":float(spacing[2])})
    logging.info(f"  [CT] saved {len(info)} slices (skull-stripped) → {out_dir}")
    return info

def save_mr_on_ct_grid_improved(mr_reg_on_ct: sitk.Image,
                               ct_mask_on_ct: sitk.Image,
                               out_dir: str,
                               tag: str,
                               meta: Dict[str,Any],
                               norm_mode: str = "robust",
                               p_low: float = 1.0,
                               p_high: float = 99.0,
                               wc_adjust: float = 0.0,
                               ww_multiplier: float = 1.0,
                               *,
                               orig_img_LPS_for_meta: sitk.Image | None = None,
                               orig_files_for_meta: list[str] | None = None) -> List[Dict[str,Any]]:
    """
    수정: CT 마스크 적용하여 PNG로 저장
    """
    ensure_dir(out_dir)
    arr = sitk.GetArrayFromImage(mr_reg_on_ct)
    mask_arr = sitk.GetArrayFromImage(ct_mask_on_ct).astype(bool)
    
    logging.info(f"  [{tag}] array shape: {arr.shape}, mask shape: {mask_arr.shape}")
    logging.info(f"  [{tag}] Mask coverage: {mask_arr.sum() / mask_arr.size * 100:.1f}%")
    
    size, spacing = mr_reg_on_ct.GetSize(), mr_reg_on_ct.GetSpacing()
    cx, cy = size[0]//2, size[1]//2
    z_list = [float(mr_reg_on_ct.TransformIndexToPhysicalPoint([cx,cy,k])[2]) for k in range(size[2])]

    # 정규화
    if norm_mode in ["dicom", "robust", "percentile"]:
        if norm_mode != "percentile" and (orig_img_LPS_for_meta is None or orig_files_for_meta is None or len(orig_files_for_meta)==0):
            logging.warning(f"[{tag}] slice-wise meta not provided. Using robust volume normalization.")
            arr8 = improved_mri_normalize_stack(arr, meta, "robust", p_low, p_high, wc_adjust, ww_multiplier)
        elif norm_mode == "percentile":
            arr8 = improved_mri_normalize_stack(arr, meta, "percentile", p_low, p_high, wc_adjust, ww_multiplier)
        else:
            slice_metas = collect_slice_metas(orig_files_for_meta)
            arr8 = improved_mri_normalize_stack_slicewise(arr, z_list, orig_img_LPS_for_meta, slice_metas,
                                                        norm_mode, p_low, p_high, wc_adjust, ww_multiplier)
    else:
        arr8 = improved_mri_normalize_stack(arr, meta, "robust", p_low, p_high, wc_adjust, ww_multiplier)

    # 마스크 적용 - 슬라이스별로 배경을 검은색(0)으로
    for k in range(arr8.shape[0]):
        arr8[k][~mask_arr[k]] = 0
    
    logging.info(f"  [{tag}] First slice - masked pixels: {(~mask_arr[arr8.shape[0]//2]).sum()} / {mask_arr[0].size}")

    info=[]
    for k in range(size[2]):
        z = z_list[k]
        fn = f"{k:04d}.png"  # PNG로 변경
        save_gray(os.path.join(out_dir, fn), arr8[k])
        info.append({"slice_idx":k, "z_mm":z, "filename":fn, "spacing_z":float(spacing[2])})
    
    logging.info(f"  [{tag}] saved {len(info)} slices (skull-stripped, CT grid, norm={norm_mode})")
    logging.info(f"    intensity range: {np.min(arr):.1f} ~ {np.max(arr):.1f} → uint8")
    if norm_mode == "dicom":
        wc, ww = meta.get("WindowCenter"), meta.get("WindowWidth")
        adj_wc = wc + wc_adjust if wc is not None else None
        adj_ww = ww * ww_multiplier if ww is not None else None
        logging.info(f"    DICOM window: WC={wc}→{adj_wc}, WW={ww}→{adj_ww:.1f}")
    
    return info


# ============================== MAIN (수정) ==============================

def main():
    sitk.ProcessObject_GlobalWarningDisplayOff()

    ap = argparse.ArgumentParser(description="CT↔DWI with Skull Stripping")
    ap.add_argument("--ct_root"); ap.add_argument("--mr_root"); ap.add_argument("--out_root")
    ap.add_argument("--ct_window_center", type=float); ap.add_argument("--ct_window_width", type=float)
    ap.add_argument("--loglevel"); ap.add_argument("--start_from"); ap.add_argument("--only"); ap.add_argument("--skip")
    ap.add_argument("--mr_norm", choices=["dicom","percentile","robust"])
    ap.add_argument("--mr_p_low", type=float); ap.add_argument("--mr_p_high", type=float)
    ap.add_argument("--b0_wc_adjust", type=float, default=0.0)
    ap.add_argument("--b0_ww_adjust", type=float, default=1.0)
    ap.add_argument("--b1000_wc_adjust", type=float, default=0.0)
    ap.add_argument("--b1000_ww_adjust", type=float, default=1.0)
    ap.add_argument("--adc_wc_adjust", type=float, default=0.0)
    ap.add_argument("--adc_ww_adjust", type=float, default=1.0)
    
    # Skull stripping 옵션
    ap.add_argument("--skull_strip_method", choices=["freesurfer", "hdbet", "threshold", "none"], default="threshold",
                   help="Skull stripping method (default: threshold, freesurfer recommended if installed)")
    ap.add_argument("--skull_strip_save_mask", action="store_true", default=True,
                   help="Save brain masks as NIfTI files")
    
    # Threshold 방법 세부 조정 옵션
    ap.add_argument("--ct_skull_threshold_low", type=float, default=10.0,
                   help="CT skull stripping lower HU threshold (default: 10)")
    ap.add_argument("--ct_skull_threshold_high", type=float, default=60.0,
                   help="CT skull stripping upper HU threshold (default: 60, lower=more aggressive)")
    ap.add_argument("--ct_skull_erosion", type=int, default=5,
                   help="CT skull stripping erosion size (default: 5, higher=more aggressive)")
    
    args = ap.parse_args()

    cfg = DEFAULTS.copy()
    if args.ct_root: cfg["ct_root"]=args.ct_root
    if args.mr_root: cfg["mr_root"]=args.mr_root
    if args.out_root: cfg["out_root"]=args.out_root
    if args.ct_window_center is not None: cfg["ct_window_center"]=args.ct_window_center
    if args.ct_window_width is not None: cfg["ct_window_width"]=args.ct_window_width
    if args.loglevel: cfg["loglevel"]=args.loglevel
    if args.start_from: cfg["start_from"]=args.start_from
    if args.only: cfg["only"]=args.only
    if args.skip: cfg["skip"]=args.skip
    if args.mr_norm: cfg["mr_norm"]=args.mr_norm
    if args.mr_p_low is not None: cfg["mr_p_low"]=args.mr_p_low
    if args.mr_p_high is not None: cfg["mr_p_high"]=args.mr_p_high
    cfg["b0_wc_adjust"] = args.b0_wc_adjust
    cfg["b0_ww_adjust"] = args.b0_ww_adjust
    cfg["b1000_wc_adjust"] = args.b1000_wc_adjust
    cfg["b1000_ww_adjust"] = args.b1000_ww_adjust
    cfg["adc_wc_adjust"] = args.adc_wc_adjust
    cfg["adc_ww_adjust"] = args.adc_ww_adjust
    cfg["skull_strip_method"] = args.skull_strip_method
    cfg["skull_strip_save_mask"] = args.skull_strip_save_mask
    cfg["ct_skull_threshold_low"] = args.ct_skull_threshold_low
    cfg["ct_skull_threshold_high"] = args.ct_skull_threshold_high
    cfg["ct_skull_erosion"] = args.ct_skull_erosion

    logging.basicConfig(level=getattr(logging, cfg["loglevel"].upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")

    logging.info(f"=== CT-DWI Matcher with Skull Stripping ===")
    logging.info(f"Skull stripping method: {cfg['skull_strip_method']}")
    logging.info(f"MRI normalization mode: {cfg['mr_norm']}")

    pairs = pair_patients(cfg["ct_root"], cfg["mr_root"])
    def pid_key(pid:str): return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)',pid))
    pairs.sort(key=lambda t: pid_key(t[0]))
    only_set = set(s.strip() for s in cfg["only"].split(",") if s.strip())
    skip_set = set(s.strip() for s in cfg["skip"].split(",") if s.strip())
    if only_set: pairs=[t for t in pairs if t[0] in only_set]
    if skip_set: pairs=[t for t in pairs if t[0] not in skip_set]
    if cfg["start_from"]: pairs=[t for t in pairs if pid_key(t[0])>=pid_key(cfg["start_from"])]
    if not pairs:
        logging.error("No patients after filtering."); return

    for pid, ct_dir, mr_dir in pairs:
        logging.info(f"\n===== [{pid}] START =====")

        out_dir = os.path.join(cfg["out_root"], pid)
        dir_ct  = os.path.join(out_dir, "ct_axial")
        dir_b0  = os.path.join(out_dir, "mri_b0_axial")
        dir_b1k = os.path.join(out_dir, "mri_b1000_axial")
        dir_adc = os.path.join(out_dir, "adc")
        dir_masks = os.path.join(out_dir, "masks")
        for d in [out_dir, dir_ct, dir_b0, dir_b1k, dir_adc, dir_masks]:
            ensure_dir(d)

        # ---- CT 로드 ----
        ct_sids = list_series(ct_dir)
        if not ct_sids:
            logging.warning(f"[{pid}] No CT series in {ct_dir}"); continue
        best_sid, best_sc = None, -1e9
        for sid in ct_sids:
            files = series_files(ct_dir, sid)
            if not files: continue
            sc=0.01*len(files)
            if _HAS_PYDICOM:
                try:
                    ds=pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
                    desc=(str(getattr(ds,"SeriesDescription","")) or "").upper()
                    for kw in ["AX","AXIAL","HEAD","BRAIN"]:
                        if kw in desc: sc += 5.0
                except: pass
            if sc>best_sc: best_sid, best_sc=sid, sc
        ct_files=series_files(ct_dir, best_sid)
        ct_img=read_series_as_image(ct_files)
        ct_LPS=sitk.DICOMOrient(ct_img,'LPS')
        log_image_info(f"{pid} CT(LPS)", ct_LPS)

        # ---- MR DWI 감지 ----
        dwi = detect_dwi_series_recursive(mr_dir, max_depth=2)
        if not dwi:
            logging.error(f"[{pid}] No DWI series found in {mr_dir}"); continue

        if "b0" not in dwi:
            logging.error(f"[{pid}] No b=0 detected. Skip."); continue

        # ---- MR 로드 ----
        b0_img = read_series_as_image(dwi["b0"]["files"])
        b0_LPS = sitk.DICOMOrient(b0_img,'LPS')
        
        b1k_LPS = None
        adc_LPS = None
        if "b1000" in dwi:
            b1k_img=read_series_as_image(dwi["b1000"]["files"])
            b1k_LPS=sitk.DICOMOrient(b1k_img,'LPS')
        if "adc" in dwi:
            adc_img=read_series_as_image(dwi["adc"]["files"])
            adc_LPS=sitk.DICOMOrient(adc_img,'LPS')

        # ---- Skull Stripping ----
        logging.info(f"[{pid}] Skull stripping CT...")
        ct_brain, ct_mask = apply_skull_stripping(
            ct_LPS, 
            cfg["skull_strip_method"], 
            "ct",
            lower_threshold=cfg.get("ct_skull_threshold_low", 10.0),
            upper_threshold=cfg.get("ct_skull_threshold_high", 60.0),
            erosion_size=cfg.get("ct_skull_erosion", 5)
        )
        
        logging.info(f"[{pid}] Skull stripping MRI b0...")
        b0_brain, b0_mask = apply_skull_stripping(b0_LPS, cfg["skull_strip_method"], "mri")
        
        # 마스크 저장
        if cfg["skull_strip_save_mask"]:
            sitk.WriteImage(ct_mask, os.path.join(dir_masks, "ct_brain_mask.nii.gz"))
            sitk.WriteImage(b0_mask, os.path.join(dir_masks, "b0_brain_mask.nii.gz"))
            logging.info(f"  Saved brain masks to {dir_masks}")

        # ---- b0 ↔ CT 정합 (skull-stripped 이미지로) ----
        tx = rigid_register_CT_fixed(ct_brain, b0_brain)

        # ---- CT 저장 (마스크 적용) ----
        ct_info = save_ct_axial(ct_LPS, ct_mask, dir_ct, 
                               cfg["ct_window_center"], cfg["ct_window_width"])

        # ---- MR을 CT 격자로 리샘플 & 저장 (CT 마스크 적용) ----
        # b0
        b0_on_ct = resample_to_ct_grid(b0_LPS, ct_LPS, tx)
        save_mr_on_ct_grid_improved(
            b0_on_ct, ct_mask, dir_b0, "mri_b0", dwi["b0"]["meta"],
            norm_mode=cfg["mr_norm"], p_low=cfg["mr_p_low"], p_high=cfg["mr_p_high"],
            wc_adjust=cfg["b0_wc_adjust"], ww_multiplier=cfg["b0_ww_adjust"],
            orig_img_LPS_for_meta=b0_LPS,
            orig_files_for_meta=dwi["b0"]["files"],
        )

        # b1000
        if b1k_LPS is not None:
            b1k_on_ct = resample_to_ct_grid(b1k_LPS, ct_LPS, tx)
            save_mr_on_ct_grid_improved(
                b1k_on_ct, ct_mask, dir_b1k, "mri_b1000", dwi["b1000"]["meta"],
                norm_mode=cfg["mr_norm"], p_low=cfg["mr_p_low"], p_high=cfg["mr_p_high"],
                wc_adjust=cfg["b1000_wc_adjust"], ww_multiplier=cfg["b1000_ww_adjust"],
                orig_img_LPS_for_meta=b1k_LPS,
                orig_files_for_meta=dwi["b1000"]["files"],
            )
        
        # ADC
        if adc_LPS is not None:
            adc_on_ct = resample_to_ct_grid(adc_LPS, ct_LPS, tx)
            save_mr_on_ct_grid_improved(
                adc_on_ct, ct_mask, dir_adc, "adc", dwi["adc"]["meta"],
                norm_mode=cfg["mr_norm"], p_low=cfg["mr_p_low"], p_high=cfg["mr_p_high"],
                wc_adjust=cfg["adc_wc_adjust"], ww_multiplier=cfg["adc_ww_adjust"],
                orig_img_LPS_for_meta=adc_LPS,
                orig_files_for_meta=dwi["adc"]["files"],
            )

        # ---- 매칭 CSV ----
        def write_identity_pairs(path:str, n:int, ct_img:sitk.Image):
            cx,cy = ct_img.GetSize()[0]//2, ct_img.GetSize()[1]//2
            with open(path,"w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=["pair_idx","ct_idx","mr_idx","z_mm"])
                w.writeheader()
                for k in range(n):
                    z=float(ct_img.TransformIndexToPhysicalPoint([cx,cy,k])[2])
                    w.writerow({"pair_idx":k,"ct_idx":k,"mr_idx":k,"z_mm":z})

        n_ct = ct_LPS.GetSize()[2]
        write_identity_pairs(os.path.join(out_dir,"matched_pairs_b0.csv"), n_ct, ct_LPS)
        if b1k_LPS is not None: write_identity_pairs(os.path.join(out_dir,"matched_pairs_b1000.csv"), n_ct, ct_LPS)
        if adc_LPS is not None: write_identity_pairs(os.path.join(out_dir,"matched_pairs_adc.csv"),   n_ct, ct_LPS)

        # ---- 메타 ----
        meta = {
            "patient": pid,
            "paths": {"ct_dir": ct_dir, "mr_dir": mr_dir, "out_dir": out_dir},
            "counts_equal": {"ct": n_ct,
                             "b0": n_ct,
                             "b1000": n_ct if b1k_LPS is not None else 0,
                             "adc": n_ct if adc_LPS is not None else 0},
            "skull_stripping": {
                "method": cfg["skull_strip_method"],
                "applied": cfg["skull_strip_method"] != "none",
                "masks_saved": cfg["skull_strip_save_mask"],
            },
            "mr_norm": cfg["mr_norm"],
            "mr_percentiles": [cfg["mr_p_low"], cfg["mr_p_high"]],
            "window_adjustments": {
                "b0": {"wc_adjust": cfg["b0_wc_adjust"], "ww_adjust": cfg["b0_ww_adjust"]},
                "b1000": {"wc_adjust": cfg["b1000_wc_adjust"], "ww_adjust": cfg["b1000_ww_adjust"]},
                "adc": {"wc_adjust": cfg["adc_wc_adjust"], "ww_adjust": cfg["adc_ww_adjust"]},
            },
        }
        with open(os.path.join(out_dir,"run_meta.json"),"w") as f: 
            json.dump(meta,f,indent=2,default=str)

        logging.info(f"===== [{pid}] DONE (skull-stripped, 1-1 saved: {n_ct} per folder) =====")

    logging.info("All done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
        sys.exit(1)