# Installment guidance

## Requirment 
please refer to the [requirements.txt](requirements.txt) file.
note that some requirements are not so strict.   
```
pip install -r requirements.txt 
```
### Enviroment
pytorch 1.10.1  
mmcv-full 1.3.16  
cuda 11.3   
cv2

### install mmcv
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.10.0/index.html  
```  
See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.  

### Clone the MMAction2 repository.
```
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
```
Install build requirements and then install MMAction2.
```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

### other dependency
```
pip install tqdm tensorboardX timm einops
```
