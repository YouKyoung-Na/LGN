# CT2MRI
심뇌혈관질환을 위한 초광역 연합 의료 AI 기술 개발 - CT2MRI

## Environments
한번에 설치 : 
`pip install -r requirements.txt`  


따로 설치 : 
```
nnunetv2==2.6.2
freesurfer-stats==1.2.1
monai==1.4.0
HD_BET==2.0.1
pydicom==3.0.1
```


## Dataset Preprocessing
데이터 전처리 및 정합을 위한 코드는 `CT_MRI_slice_matcher_With_skullstrip.py` 이다.   


HD-BET 모델 파라미터 다운로드를 위해 아래를 실행한다.    
`python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"`  


DICOM형태의 의료 이미지를 입력으로 받아 실행된다.
```
python CT_MRI_slice_matcher_with_skullstrip.py \
  --ct_root CT/ct_microbleed_0 \
  --mr_root MRI/mri_microbleed_0/250218_0 \
  --out_root ./skull_strip/no_bleed_ct_mri \
  --skull_strip_method hdbet \
  --mr_norm dicom \
  --b0_ww_adjust 1.8
```


## Model Weight
**[ldm 관련 weight는 링크 🔗]**(https://drive.google.com/drive/folders/1j3K79B5kJi28BA2bQ83YJW7bhGdqSs_f?usp=sharing, "Google Drive")에서 다운로드 후, `diffusion_model/` 폴더에 넣는다.  
  
**[공식 사이트 weight 다운로드]** (https://github.com/CompVis/latent-diffusion.git "https://github.com/CompVis/latent-diffusion.git") 의 Model Zoo의 `kl-f4.zip`(autoencoder), `celeba.zip`(ldm)을 사용한다.

의료 데이터에 ldm을 적용하고 싶다면, `diffusion_model/latent_diffusion.ipynb `를 실행하면 된다.

