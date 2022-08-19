# BORT
Two Heads are Better than One: Robust Learning Meets Multi-branch Models


## install
```
git clone https://github.com/huangd1999/BORT.git
cd BORT
pip install torchattacks
mkdir checkpoint
$ Download pretrain model from Google Drive
```

## fast eval of BORT with WRN-28-10 on cifar10 dataset
[here](https://drive.google.com/drive/folders/1AtUTIfQ1C3yEoMybvKaXHoRUTD8K_mH-?usp=sharing) is BORT's pretrained model. 

CIFAR100 pretrained model will be released in the future.

`python test.py --attack AutoAttack --weights './checkpoint/BORT-WRN-28-10-CIFAR10-PGD.pt' ` 
