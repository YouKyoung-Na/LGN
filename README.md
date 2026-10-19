# Learning from Good Neighbors: Slice-Aware CT-to-Diffusion MRI Synthesis
MICCAI 2026

## Environments
#### Install all at once:
`pip install -r requirements.txt`  


#### Install individually:
```
nnunetv2==2.6.2
freesurfer-stats==1.2.1
monai==1.4.0
HD_BET==2.0.1
pydicom==3.0.1
simpleitk==2.5.2
```

#### Installation using uv:
After running `pip install uv`,


`uv sync`


## Dataset Preprocessing
### 1. Download HD-BET Model
Download the HD-BET pre-trained parameters before starting data preprocessing.


`python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"`  

### 2. CT-MRI Registration and Preprocessing
Register CT and MRI images in DICOM format and perform skull stripping using `CT_MRI_slice_matcher_with_skullstrip.py`

```
python CT_MRI_slice_matcher_with_skullstrip.py \
  --ct_root CT/ct_microbleed_0 \
  --mr_root MRI/mri_microbleed_0/250218_0 \
  --out_root ./skull_strip/no_bleed_ct_mri \
  --skull_strip_method hdbet \
  --mr_norm dicom \
  --b0_ww_adjust 1.8
```


### 3. Data Quality Control (QC)
Visually inspect the CT and MRI images using `data_cert.py` and save low-quality images (top/bottom of the brain, bone-heavy slices, etc.) to an exclusion list.
```
python data_cert.py \
  --root ./skull_strip/no_bleed_ct_mri \
  --save-csv exclusions.csv \
  --start_case 0 \
  --start_index 0
```


**Keyboard Shortcuts:**
- `A/D/방향키`: Previous / Next image
- `E`: Toggle exclusion / inclusion of current image
- `S`: Save exclusion list to CSV
- `H/?`: Show / Hide help
- `Q`: Exit (Auto-save)


### 4. Final Dataset Generation
Run `data_postprocessing.ipynb` to filter out images excluded in QC and generate the final training dataset.



## Model
### Download Weights
1. **Download from Google Drive** (Recommended)  
   Download from the [🔗 Google Drive 링크](https://drive.google.com/drive/folders/1j3K79B5kJi28BA2bQ83YJW7bhGdqSs_f?usp=sharing) and place it in the `diffusion_model/` folder.

2. **Download from Official Repository**  
   From the Model Zoo of the [Latent Diffusion 공식 저장소](https://github.com/CompVis/latent-diffusion):
   - `kl-f4.zip`: Autoencoder Model
   - `celeba.zip`: LDM Model


### Training and Application
Run `diffusion_model/latent_diffusion.ipynb` to apply the Latent Diffusion Model to medical data.



## Project Structure
```
CT2MRI/
├── CT_MRI_slice_matcher_with_skullstrip.py  # 데이터 전처리 및 정합, Data preprocessing and registration
├── data_cert.py                               # 품질 검사 도구, Quality check tool
├── data_postprocessing.ipynb                  # 최종 데이터셋 생성, Final dataset generation
├── diffusion_model/
│   └── latent_diffusion.ipynb                # LDM 학습/추론, LDM training / inference
└── requirements.txt                           # 필수 패키지 목록, List of required packages
```


## BibTex
```
BibTex
@InProceedings{NaYou_Learning_MICCAI2026,
        author = { Na, You-Kyoung AND Choi, Kang-Ho AND Yang, Hyung-Jeong AND Kim, Jahae AND Cho, Yeong-Jun},
        title = { { Learning from Good Neighbors: Slice-wise Quality-aware CT-to-Diffusion MRI Synthesis } },
        booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
        year = {2026},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 16890},
        month = {September},
        page = {pending}
}
```

## References
- [HD-BET: Automated Brain Extraction](https://github.com/MIC-DKFZ/HD-BET)
- [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)
#-= # References
- [HD-BET: Automated Brain Extraction](https://github.com/MIC-DKFZ/HD-BET)
- [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)
