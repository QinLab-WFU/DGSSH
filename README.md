# Deep Global Semantic Structure-preserving Hashing via Corrective Triplet Loss for Remote Sensing Image Retrieval
This paper is accepted for publication with Expert Systems with Applications. 
If you have any questions please contact zhouhongyan199607@163.com.

## Dependencies
We use python to build our code, you need to install those package to run
- Python 3.9.7
- Pytorch 1.12.1
- torchvision 13.1
- CUDA 11.3


## Training

### Processing dataset
Before training, you need to download the UCMerced dataset http://weegee.vision.ucmerced.edu/datasets/landuse.html,
AID dataset from https://captain-whu.github.io/AID ,WHURS dataset from https://captain-whu.github.io/BED4RS.


### Download Swin Transformer pretrained model
Pretrained model will download from  https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth.

### Start
After the dataset has been prepared, we could run the follow command to train.
> Python main.py --cfg ./config/swin_config.yaml --batch_size 32
