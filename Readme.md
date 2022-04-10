# Readme

## 文件说明

**完成本次projcet需要下载除了readme.md的所有文件**



1. tcktorch文件夹

   该文件夹是我基于numpy，仿照pytorch实现的自定义模组，具体介绍见于文件夹内的readme文件。

   本次project的训练、参数搜索、测试都基于该模组

2. minist文件夹

   该文件夹存放MINIST数据集

3. train.py

   该文件是本次project的**训练**文件，**执行该文件即可训练模型**。**需要预先下载minist文件夹以及tcktorch文件夹。**

   修改模型架构在 `execute`

   改变学习率、隐藏层大小、正则化强度（weight decay) 在`execute`

   改变 batch size、#epoch、是否输出loss图像和accuracy图像、是否保存训练出来的模型、是否记录训练信息在 `train`

4. paraSearch.py

   该文件是本次project的**参数搜索**文件，在一定参数范围内执行随机搜索。**需要预先下载minist文件夹、tcktorch文件夹以及train.py文件**。

   **直接执行该文件即可进行参数搜索**，并将最好的模型的学习率、隐藏层大小、正则化强度输出在指定文件内。如果需要在参数搜索的同时保存模型，请将`train.py`文件内`train`函数的`save`设为True。

   可自行设定搜索范围以及搜索个数。

5. eval.py文件

   该文件是本次project的**测试集测试**文件，导入模型，用经过参数查找后的模型进行测试，输出分类精度。**需要预先下载minist文件夹、tcktorch文件夹、model.pkl以及result2.txt**。

   **直接执行该文件即可进行在测试集上进行测试。**

6. model.pkl

   该文件是经过参数搜索后得到的最佳模型

7. result2.txt

   该文件是参数搜索的输出文件，需要从该文件读取模型的超参数，以下是文件内容：

   Best model: learning_rate:0.7066002283160423	hidden_units:362	weight_decay:2.039352353360306e-05	valid_acc:98.1167

