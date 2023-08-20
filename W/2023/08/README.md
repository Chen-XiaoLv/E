## 8/16

+ 早上九点十分来的。先吃了口饭，然后复盘昨天的内容并总结。
  + 开小差了，在群里讨论兴趣入门这一概念
+ 十点上了十五分钟厕所。回来将Blog2和这个图标进行了更换
  + 完成了数据清洗，说一下现在的情况，红高粱的数据只到2006年，小米的数据在2005和2020有个长时间的空缺。
  + 肚子有点痛

| TimeStamp |                Event                 |   Flag   | CostTime |
| :-------: | :----------------------------------: | :------: | :------: |
|  10:47-   |          复盘影像裁剪与导出          |    s     |    -     |
|           |            看LK99最近争论            | 没啥结果 |    -     |
|   11:06   |          复盘影像裁剪与导出          |    e     |    19    |
|   11:07   |        查看GEE task tiff进展         |    s     |    -     |
|           | 看了会知乎，程老师叫我下午帮忙拿快递 |    -     |    -     |
|  11：28   |                 摸了                 |    e     |    21    |
|   11:29   |             复盘GEE操作              |    s     |    -     |
|   11:41   |             复盘GEE操作              |    e     |    11    |

![image-20230820230155081](README/image-20230820230155081.png)

好好好，昨天的数据有问题。我重新改一下scale再试试

| TimeStamp |                  Event                   | Flag | CostTime |
| :-------: | :--------------------------------------: | :--: | :------: |
|   14:22   |                  开工!                   |  s   |    -     |
|   14:33   |     修正Temperature，并获得最终结果      |  -   |    11    |
|   15:28   | 困死了，困死了，看完了Asset Manage的部分 |  e   |    55    |
|   15:29   |                 开始记录                 |  s   |    -     |
|   16:05   |                   完成                   |  e   |    36    |
|   16:07   |                   摸鱼                   |  s   |    -     |
|   16:48   |          干活，导出数据文档查看          |  S   |    -     |
|   17:29   |            摸鱼，看缅北割腰子            |      |    1     |

![image-20230820230202409](README/image-20230820230202409.png)

看，这个就是最终结果，并不能很好的匹配，底图是土地利用图，因为采样分辨率的问题，并不能对上。

`hover over ` 悬停

`quota  ` 配额

`more specifically` 更具体的

---

## 8/17

| TimeStamp |                Event                 |   Flag   | CostTime |
| :-------: | :----------------------------------: | :------: | :------: |
|   9:26    |          干活，复习昨日知识          |    s     |    -     |
|           |          学习GEE IMPORT操作          | 没啥结果 |    -     |
|   10:30   |            学习GEE Export            |    e     |    64    |
|   10:31   |                break                 |    s     |    -     |
|           |  静思，想一想为什么学了四年啥也不会  |    -     |    -     |
|  10：47   |                Export                |    s     |    -     |
|  11：14   | 准备跟进一下Global Forest Change项目 |    s     |    27    |
|   11:41   |             复盘GEE操作              |    e     |    11    |





+ `specify` n. 指定xxx
+ `To sum up`，总而言之
+ `subtly misleading` 隐晦误导
+ `As a shortcut` 作为捷径
+ `duplicate` 重复的

请注意，导出地图瓦片、BigQuery和提取图像数据编程并没有记录。

| TimeStamp  |                        Event                         | Flag | CostTime |
| :--------: | :--------------------------------------------------: | :--: | :------: |
|   14：49   |              摸鱼回来了，摸了50分钟知乎              |  s   |    -     |
|   14：49   |                看一看实验接下来的方向                |  s   |          |
|   15:20    |  修改了原始的土地利用Label，根据谢高地表做了张新表   |  -   |    -     |
|   15:40    | 看文献，大致知道ESV咋算了，并且更新了Lattics工作记录 |  s   |    -     |
|   16:13    |                 摸鱼。并开始整个小活                 |  -   |    -     |
|   16:30    |  看看我的CSDN博客，有些感慨。还有人催更，可惜我现在  |      |          |
|            |                        整大活                        |  s   |    27    |
|   19:40    |                        整大活                        |  e   |    11    |
|   21:00    |               不整活了，边训练边刷视频               |      |          |
|   23:00    |           回宿舍，为啥一定要花时间刷视频？           |      |          |
| 23:00-2:00 |             洗澡，为陈聿为提诗？？我有毒             |      |          |



复现了之前的LSTM，之前的代码带有池化.....LOSS只能到1500

删掉池化后，LOSS到了52.池化问题特别大，一维卷积凭啥用池化啊？

300次后，甚至有0.0002的LOSS。。。。

我现在再加上bn层看看吧。

结果不是很好，估计是我把卷积换成了全连接。奇了怪了。

又换回来了，现在是卷积+BN+RELU，很正常的组合

效果不稳定。

还是换成之前的吧

> 训练时希望每个通道有超过 1 个值，得到输入大小 torch.Size([1, 64, 1])

出现了这样的错误，原因在于样本太小了一个人为啥要标准化。好吧，那就不用。

搞错了...原先的输出LOSS是最后一位，现在改成平均了。

```python
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import alive_progress
import torch.nn.functional as F
from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import CurrentConfig

path=r"E:\科研\ESV计算\Sri Lanka"
yield_c=pd.read_csv(path+r"\yield_c.csv")
pp_c=pd.read_csv(path+r"\producerPrices_c.csv")
yield_c[yield_c["Year"]>2015].groupby('Item').mean()
x=pp_c[pp_c['Item']=="Maize (corn)"]['Value']
x=torch.FloatTensor(x).view(-1)

windows_size=4
# 设置迭代器
class MyDataSet(Dataset):
    def __init__(self,seq,ws=4):
        self.ori=[i for i in seq[:ws]]
        self.label=[i for i in seq[ws:]]
        self.reset()
        self.ws=ws

    def set(self,dpi):
        self.x.append(dpi)
    def reset(self):
        self.x=self.ori[:]
    def get(self,idx):
        return self.x[idx:idx+self.ws],self.label[idx]
    def __len__(self):
        return len(self.x)
train_data=MyDataSet(x,windows_size)

class Net5(nn.Module):
    def __init__(self,in_features=54,n_hidden1=54,n_hidden2=256,n_hidden3=512,out_features=7):
        super(Net5, self).__init__()
        self.flatten=nn.Flatten()
        self.hidden1=nn.Sequential(
            nn.Linear(in_features,n_hidden1,False),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU()
        )
        self.hidden2=nn.Sequential(
            nn.Linear(in_features,n_hidden2),
            nn.BatchNorm1d(n_hidden2),
            nn.ReLU()
        )
        self.hidden3=nn.Sequential(
            nn.Linear(n_hidden2,n_hidden2),
            nn.BatchNorm1d(n_hidden2),
            nn.ReLU()
        )
        self.hidden4=nn.Sequential(
            nn.Linear(n_hidden2,n_hidden3),
            nn.BatchNorm1d(n_hidden3),
            nn.ReLU(),

        )
        self.out=nn.Sequential(nn.Linear(n_hidden3,out_features))

    def forward(self,x):
        x=self.flatten(x)
        x1=self.hidden1(x)
        x2=self.hidden2(x+x1)
        x3=self.hidden3(x2)
        o=self.hidden4(x3+x2)
        return F.softmax(self.out(o),dim=1)


class CNN(nn.Module):
    def __init__(self, output_dim=1):
        super(CNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.lr = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(64, 128, 1)

        self.bn1, self.bn2 = nn.BatchNorm1d(64), nn.BatchNorm1d(128)
        self.bn3, self.bn4 = nn.BatchNorm1d(1024), nn.BatchNorm1d(128)
        self.flatten = nn.Flatten()
        self.lstm1 = nn.LSTM(128, 1024)
        self.lstm2 = nn.LSTM(1024, 256)
        self.lstm3=nn.LSTM(256,512)
        self.fc = nn.Linear(512, 512)
        self.fc4=nn.Linear(512,256)
        self.fc1 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_dim)

    @staticmethod
    def reS(x):
        return x.reshape(-1, x.shape[-1], x.shape[-2])

    def forward(self, x):
        x = self.reS(x)
        x = self.conv1(x)  # torch.Size([1, 64, 1])
        x = self.lr(x)

        x = self.conv2(x)  # torch.Size([1, 128, 32])
        x = self.lr(x)

        # x = self.conv3(x)  # torch.Size([32, 300, 298])
        # x = self.maxpool3(x)  # torch.Size([32, 300, 100])
        x = self.flatten(x)
        # 注意Flatten层后输出为(N×T,C_new)，需要转换成(N,T,C_new)

        # LSTM部分
        x, h = self.lstm1(x)
        x, h = self.lstm2(x)
        x,h=self.lstm3(x)
        # 注意这里只使用隐层的输出
        x, _ = h

        x = self.fc(x.reshape(-1, ))
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

def Train(model,train_data,seed=1):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    Mloss=100000
    path=r"E:\科研\ESV计算\Sri Lanka\Train\MODEL_\%s.pth"%seed
    # 设置损失函数,这里使用的是均方误差损失
    criterion = nn.MSELoss()
    # 设置优化函数和学习率lr
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,betas=(0.9,0.99),
                               eps=1e-07,weight_decay=0)
    # 设置训练周期
    epochs = 10000
    criterion=criterion.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss=0

        for i in range(len(x)-4):
            # 每次更新参数前都梯度归零和初始化
            seq,y_train=train_data.get(i)
            seq,y_train=torch.FloatTensor(seq),torch.FloatTensor([y_train])
            seq=seq.unsqueeze(dim=0)
            seq,y_train=seq.to(device),y_train.to(device)

            optimizer.zero_grad()
            # 注意这里要对样本进行reshape，
            # 转换成conv1d的input size（batch size, channel, series length）
            y_pred = model(seq)
            loss = criterion(y_pred, y_train)
            loss.backward()
            train_data.set(y_pred.to("cpu").item())
            optimizer.step()
            total_loss+=loss

        train_data.reset()
        if total_loss.tolist()<Mloss:
            Mloss=total_loss.tolist()
            torch.save(model.state_dict(),path)
            print("Saving")
        print(f'Epoch: {epoch+1:2} Mean Loss: {total_loss.tolist()/len(train_data):10.8f}')
    return model

A=CNN()
Train(A,train_data,"CNN_1")

checkpoint=torch.load(r"E:\科研\ESV计算\Sri Lanka\Train\MODEL_\CNN_1.pth")
A.load_state_dict(checkpoint)
A.to("cpu")

print(A(torch.FloatTensor([[14.65,18.11,19.54,21.27]])))


pre,ppre=[i.item() for i in x[:4]],[]
for i in range(len(x)-3):
    ppre.append(A(torch.FloatTensor(x[i:i+4]).unsqueeze(dim=0)))
    pre.append(A(torch.FloatTensor(pre[-4:]).unsqueeze(dim=0)).item())
l=Line()
l.add_xaxis([i for i in range(len(x))])
l.add_yaxis("Original Data",x.tolist())
l.add_yaxis("Pred Data(Using Raw Datas)",x[:4].tolist()+[i.item() for i in ppre])
l.add_yaxis("Pred Data(Using Pred Datas)",x[:4].tolist()+pre)
l.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
l.set_global_opts(title_opts=opts.TitleOpts(title='LSTM CNN')
                            
                             )

l.render_notebook()
```

![image-20230817200641094](README/image-20230817200641094.png)

只靠学习自己的，似乎并不能取得更好的结果。

所以我们要改变训练思路。





![image-20230817225538048](README/image-20230817225538048.png)

![image-20230817225637450](README/image-20230817225637450.png)

其实分析上面的结果，我们可以发现LSTM的波动要比CNN好，CNN后面死水一潭，应该是梯度消失导致的，前面信息没有了，后面信息又是自个构造的，这就导致了到后面变成了线性情况。

后面窗口大小改成6了，效果好多了。现在就是，将其归档吧，要不发篇blog吧，好久没发了。![image-20230818143611974](README/image-20230818143611974.png)



---

## 818

| TimeStamp |                            Event                             | Flag | CostTime |
| :-------: | :----------------------------------------------------------: | :--: | :------: |
|   9:10    |                          醒了，好困                          |  s   |    -     |
|   9:40    |                    摸鱼，看到了世界的参差                    |  -   |    -     |
|   9:50    |           修正了昨天的结果，更换了窗口，完善了代码           |  -   |    -     |
|  11：10   | 得到了较好的结果，中间很累，大脑混乱什么都想不起来，又depression了 |  -   |    -     |
|   11:30   | 摸鱼，打算写一篇Blog，下午的工作就是Blog+GEE生态环境变化吧，中午去报修一下电表？ |  -   |    -     |
|   11:34   |                           总结下吧                           |      |          |
|  11：14   |             准备跟进一下Global Forest Change项目             |  s   |    27    |
|   11:41   |                         复盘GEE操作                          |  e   |    11    |

现在又有一点问题。

我选择了窗口5对小米进行计算。很可惜的是，最终结果与官方数据相去甚远。甚至官方本身的结果也是跳变，纳闷。

![image-20230818144001139](README/image-20230818144001139.png)、

官方价格在一年翻了一倍。牛逼。

那就不要用小米了？算了我不想浪费，官方这个，产量增加30%,价格增加100%，这符合市场规律吗？

周五下午摆烂。无所谓，又Depression了。

然后就是打算写一篇Blog。外加下数据了。

晚上健身去了。然后点了茶话弄他们家的南山烟雨、芊芊绿雪 还可以，都是茶

