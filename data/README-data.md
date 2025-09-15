# **【AI入门系列】城市探险家：街景字符识别学习赛**

# AI赛题解析

<table><tr><td>文档</td><td>大小</td><td>操作</td><td>ossutil命令</td></tr><tr><td>mchar_data_lis_0515.csv</td><td>csv(659B)</td><td>下载</td><td>复制命令</td></tr></table>

# 一、赛题数据

赛题来源自Google街景图像中的门牌号数据集（TheStreetViewHouseNumbersDataset，SVHN），并根据一定方式采样得到比赛数据集

数据集报名后可见并可下载，该数据来自真实场景的门牌号。训练集数据包括3W张照片，验证集数据包括1W张照片，每张照片包括颜色值的编码类别和具体位置；为了保证比赛的公平性，测试集A包括4W张照片，测试集B包括4W张照片。

![](images/a166e645db4688b94b8385f155dd391c8b0d9056c94da644da08a6637f25ee52.jpg)

需要注意的是本赛题需要选手识别图片中所有的字符，为了降低比赛难度，我们提供了训练集、验证集和测试集中字符的位置框。

# 字段表

所有的数据（训练集、验证集和测试集）的标注使用JSON格式，并使用文件名进行索引。如果一个文件中包括多个字符，则使用列表将字合。

<table><tr><td>Field</td><td>Description</td></tr><tr><td>top</td><td>左上角坐标Y</td></tr><tr><td>height</td><td>字符高度</td></tr><tr><td>left</td><td>左上角坐标X</td></tr><tr><td>width</td><td>字符宽度</td></tr><tr><td>label</td><td>字符编码</td></tr></table>

注：数据集来源自SVHN，并进行匿名处理和噪音处理，请各位选手使用比赛给定的数据集完成训练。

# 二、评测标准

评价标准为准确率，选手提交结果与实际图片的编码进行对比，以编码整体识别准确率为评价指标，结果越大越好，具体计算公式如下：

$$
score = \frac{编码识别正确的数量}{测试集图片数量}
$$

# 三、结果提交

提交前请确保预测结果的格式与sample_submit.csv中的格式一致，以及提交文件后缀名为csv。

```makefile
file_name, file_code
0010000.jpg,451
0010001.jpg,232
0010002.jpg,45
0010003.jpg,67
0010004.jpg,191
0010005.jpg,892
```


