# CT2MRI
심뇌혈관질환을 위한 초광역 연합 의료 AI 기술 개발 - CT2MRI

## Environments
#### 한번에 설치 : 
`pip install -r requirements.txt`  


#### 따로 설치 : 
```
nnunetv2==2.6.2
freesurfer-stats==1.2.1
monai==1.4.0
HD_BET==2.0.1
pydicom==3.0.1
simpleitk==2.5.2
```

#### uv를 활용한 설치:
`uv sync`


## Dataset Preprocessing
### 1. HD-BET 모델 다운로드
데이터 전처리를 시작하기 전에 HD-BET 사전 학습 파라미터를 다운로드합니다.  

`python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"`  

### 2. CT-MRI 정합 및 전처리
`CT_MRI_slice_matcher_with_skullstrip.py`를 사용하여 DICOM 형식의 CT와 MRI 이미지를 정합하고 skull stripping을 수행합니다.

```
python CT_MRI_slice_matcher_with_skullstrip.py \
  --ct_root CT/ct_microbleed_0 \
  --mr_root MRI/mri_microbleed_0/250218_0 \
  --out_root ./skull_strip/no_bleed_ct_mri \
  --skull_strip_method hdbet \
  --mr_norm dicom \
  --b0_ww_adjust 1.8
```


### 3. 데이터 품질 검사 (QC)
`data_cert.py`를 사용하여 CT와 MRI 이미지를 시각적으로 확인하고, 저품질 이미지(두뇌 상단/하단, 뼈가 많은 슬라이스 등)를 제외 목록으로 저장합니다.
```
python data_cert.py \
  --root ./skull_strip/no_bleed_ct_mri \
  --save-csv exclusions.csv \
  --start_case 0 \
  --start_index 0
```


**키보드 단축키:**
- `A/D/방향키`: 이전/다음 이미지
- `E`: 현재 이미지 제외/포함 토글
- `S`: 제외 목록을 CSV로 저장
- `H/?`: 도움말 표시/숨김
- `Q`: 종료 (자동 저장)


### 4. 최종 데이터셋 생성
`data_postprocessing.ipynb`를 실행하여 QC에서 제외된 이미지를 필터링하고 최종 학습 데이터셋을 생성합니다.



## Model
### 가중치 다운로드
1. **Google Drive에서 다운로드** (권장)  
   [🔗 Google Drive 링크](https://drive.google.com/drive/folders/1j3K79B5kJi28BA2bQ83YJW7bhGdqSs_f?usp=sharing)에서 다운로드 후 `diffusion_model/` 폴더에 배치

2. **공식 저장소에서 다운로드**  
   [Latent Diffusion 공식 저장소](https://github.com/CompVis/latent-diffusion)의 Model Zoo에서:
   - `kl-f4.zip`: Autoencoder 모델
   - `celeba.zip`: LDM 모델


### 학습 및 적용
의료 데이터에 Latent Diffusion Model을 적용하려면 `diffusion_model/latent_diffusion.ipynb`를 실행하세요.



## Project Structure
```
CT2MRI/
├── CT_MRI_slice_matcher_with_skullstrip.py  # 데이터 전처리 및 정합
├── data_cert.py                               # 품질 검사 도구
├── data_postprocessing.ipynb                  # 최종 데이터셋 생성
├── diffusion_model/
│   └── latent_diffusion.ipynb                # LDM 학습/추론
└── requirements.txt                           # 필수 패키지 목록
```



## References
- [HD-BET: Automated Brain Extraction](https://github.com/MIC-DKFZ/HD-BET)
- [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)