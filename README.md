# CT2MRI
심뇌혈관질환을 위한 초광역 연합 의료 AI 기술 개발 - CT2MRI

## Environments
`pip install -r requirements.txt`  


## Dataset 
데이터 전처리 및 정합을 위한 코드는 `CT_MRI_slice_matcher_With_skullstrip.py` 이다.   
HD-BET 패키지를 이용하여 진행했다.  
`pip install HD-BET` 


모델 파라미터 다운로드를 위해 아래를 실행한다.    
`python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"`  


DICOM형태의 의료 이미지를 입력으로 받아 실행이된다.
```
python CT_MRI_slice_matcher_with_skullstrip.py \
  --ct_root CT/ct_microbleed_0 \
  --mr_root MRI/mri_microbleed_0/250218_0 \
  --out_root ./skull_strip/no_bleed_ct_mri \
  --skull_strip_method hdbet \
  --mr_norm dicom \
  --b0_ww_adjust 1.8
```
