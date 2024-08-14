# DAMO
Dynamic Adversarial defense with Multi-branch diversity

Two Heads are Better than One: Robust Learning Meets Multi-branch Models


## install
```
git clone https://github.com/huangd1999/DAMO.git
cd DAMO
pip install torchattacks
mkdir checkpoint
$ Download pretrain model from Google Drive
```

## fast eval of DAMO with WRN-28-10 on cifar10 dataset
[here](https://drive.google.com/drive/folders/1AtUTIfQ1C3yEoMybvKaXHoRUTD8K_mH-?usp=sharing) is DAMO's pretrained model. 

CIFAR100 pretrained model will be released in the future.

`python test.py --attack AutoAttack --weights './checkpoint/DAMO-WRN-28-10-CIFAR10-PGD.pt' ` 
