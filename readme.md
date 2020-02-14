## 2019新型冠状病毒在武汉地区的流行病学分析

> Author: 吴烜圣 (wuxsmail@163.com)
>
> Date: 2020-2-14
>
> Full Page Paper download: http://www.kitgram.cn/downloads/something/SEIDC_19nCoV.pdf

### Data

我们使用[丁香园](https://github.com/BlankerL/DXY-COVID-19-Data)、[腾讯实时监测](https://news.qq.com/zt2020/page/feiyan.htm)数据和[国家卫检委](https://github.com/839Studio/Novel-Coronavirus-Updates)公开发布的数据整理了1月23日至2月1日武汉市的疫情情况。具体采集到的变量包括每天的确诊人数、死亡人数和治愈人数。

### Model - SEIDC

本文主要仿照SEIR传染病模型并按照2019-nCoV病毒的传播特性进行了修改。我们先划分出5个类别的人群，分别为可能感染该病的健康人员（Susceptible, S），潜伏期患者（Exposed, E），确诊患者（Infected, I），死者（Dead, D）和治愈患者（C）。接着，我们定义潜伏期患者导致感染的概率为$\alpha_1$，确诊患者导致的感染的概率为$\alpha_2$，潜伏期患者被确诊的概率为$\beta$，确诊患者被治愈的概率为$\sigma$，确诊患者死亡的概率为$\gamma$。我们通过下式（1）的常微分方程组定义五个人群间的转移方程来描述传染病传播情况。
$$ {math}
\begin{aligned}
\frac{dS}{dt} &= - \alpha_{1} E - \alpha_{2} I + \sigma I \\
\frac{dE}{dt} &= \alpha_{1} E + \alpha_{2} I - \beta E\\
\frac{dI}{dt} &= \beta E - \sigma I - \gamma I\\
\frac{dD}{dt} &= \gamma E\\
\frac{dC}{dt} &= \sigma I \\
\end{aligned}
$$

由于我们不确定1月23日武汉封城时武汉市内的市民准确数量，也难以观测到潜伏期患者数量，于是我们尽可能利用已知数据，则定义我们模型的损失函数为：
$$ {math}
Loss=\sum_{i=1}^n(I_i-\hat{I_i})^2+\sum_{i=1}^n(D_i-\hat{D_i})^2+\sum_{i=1}^n(C_i-\hat{C_i})^2
$$
最后，使用模拟退火算法估计模型（1）中的系数。同时，由于作为基期的1月23日的潜伏期患者人数难以被观测到，于是从50至1100之间以25为间隔训练了一系列的模型，并最终选择模型损失最小的模型为本文的最终模型。

### Result

病毒的平均潜伏期为：5.40天 

病毒的再生基数为：2.39 （意味着一个患者可能导致2.39个人感染）

1月23日0时武汉市内的潜伏期患者：100人