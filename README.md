# GEC-Syntax-Data-Augmentation

## 设置
实验环境：
```sh
. scripts/set_environment.sh

# 对于Errant (v2.0.0)评估，需要python 3.6
# 确保环境中有python 3.6，然后运行:
. scripts/set_py36_environment.sh
```

数据格式：
```txt
S   [src]
T   [tgt]

S   [src]
T   [tgt]
```
实验环境和英文数据集请参考[gecdi](https://github.com/Jacob-Zhou/gecdi)仓库，中文数据集可[在此](https://drive.google.com/file/d/1qpbEkdGL_EpSu7hI4e98YQ7A3QrTQQDS/view?usp=sharing)获得

## 模型训练
请参考[train_zh_baseline.sh](train_zh_baseline.sh)与[train_zh_aug.sh](train_zh_aug.sh)

## 推理和评估
请参考[pred_zh.sh](pred_zh.sh)


