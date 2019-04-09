# kddcup99-cnn

## 介绍

该项目使用一个非常简单的三层卷积网络来对kddcup99数据集进行训练，准确率：0.993921。

**项目尽量保持代码的简单，方便大家学习、扩展。**

训练过程和效果可以直接点击`kdd-pytorch.ipynb`查看。

## 模型

#### data.py

kddcup99共有41维特征，分为23类。可以转换为8*8的矩阵，缺少的位用0补齐。

其中特征是字符串的，使用`sklearn.preprocessing.LabelEncoder`转换为数字。

```python

test_data = [0, b'tcp', b'http', b'SF', 219, 643, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 0.0,
 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 44, 255, 1.0, 0.0, 0.02, 0.03, 0.0, 0.0, 0.0, 0.0]

label = b'normal'

```

#### model.py

卷积层：Conv(1, 6, 3) -> BN -> ReLU -> Conv(6, 16, 3) -> BN -> ReLU -> MaxPool(2, 2)

全连接层：Linear(144, 512) -> Linaer(512, 256) -> Linaer(256, 23)

#### train.py

- batch_size = 128
- learning_rate = 1e-2
- num_epoches = 20

## 结果

```
epoch 1
**********
Finish 1 epoch, Loss: 0.208683, Acc: 0.954045
Test Loss: 1.635921, Acc: 0.568327

epoch 2
**********
Finish 2 epoch, Loss: 0.098167, Acc: 0.982132
Test Loss: 0.154757, Acc: 0.988131

... ...

epoch 18
**********
Finish 18 epoch, Loss: 0.024409, Acc: 0.992707
Test Loss: 0.075168, Acc: 0.992011

epoch 19
**********
Finish 19 epoch, Loss: 0.023752, Acc: 0.992970
Test Loss: 0.049177, Acc: 0.993765

epoch 20
**********
Finish 20 epoch, Loss: 0.023343, Acc: 0.993097
Test Loss: 0.059661, Acc: 0.993921
```

```python
def predict(data, multiple=False):
    _data = dataset.encode(data)
    _data = torch.from_numpy(
        np.pad(_data, (0, 64 - len(_data)), 'constant').astype(np.float32)
    ).reshape(-1, 1, 8, 8).cuda()
    _out = int(torch.max(model(_data).data, 1)[1].cpu().numpy())
    return dataset.decode(_out, label=True)
    
predict(test_data)

# output: b'normal.'
```