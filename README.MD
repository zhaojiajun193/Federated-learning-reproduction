联邦学习论文复现
===
对2021年Google发表在ICLR上的”Adaptive Federated Optimization“一文进行了三种算法在Emnist、Cifar10、Cifar100上的pytorch复现工作，由于本人能力有限，并未做的十分出色，对于NLP上StackOverflow、Shakespeare数据集由于缺乏处理经验并未完成，如果您对此感兴趣欢迎与我联系共同探讨（联系方式xdu1816@qq.com），在此将我的代码与大家分享，欢迎批评指导。


配置环境
----
1.所用环境为pytorch1.8.1 numpy1.19.2 pandas 1.1.3 python 3.8.5

代码注释及各个文件夹下的内容解析
----
2.在cifar10文件夹下（其他文件夹也均类似）有四个python文件，cifar10fedadagrad、cifar10fedadam、cifar10fedyogi分别对应Adaptive Federated Optimiziation一文中的三种算法，resnet.py对应resnet的模型，在cifar10fedadagrad.py中对所有代码都进行了详细的注解，由于代码结构大致相同，并未在其他代码文件中进行赘述，如有需要可直接联系邮箱xdu1816@163.com或1756523104@qq.com。

代码运行方式
----
3.代码下载后，按照上述环境要求配置完相同环境后，在python编辑器中打开诸如cifar10fedadagrad.py字样的代码文件，保证与相应的模型构建代码文件处于同一文件夹下，ctrl+shift+F10即可运行代码，在运行代码时会现在代码所在同级文件夹创建data文件夹供数据集下载，下载完成后稍等片刻进行模型的创建后即可开始训练工作，训练过程中将会在命令行提示训练轮次及该轮次结束后在测试集上的训练精度。

改善与不足
----
4.所用为pytorch，由于本人缺乏和概率论相关的部分数学基础，在数据集分配上采用均匀随机分配，并未实现tensorflow_federated中对数据集的分类，仍值得改进。
5.由于缺乏NLP经验且时间较短，并未实现对Stack Overflow数据集及Shakespeare数据集的next-word-prediction工作。

总结
----
6.目前fedyogi、fedadam和fedadagrad在分类问题上都没有出现问题，且能取得不错的结果（因准确率达到一定程度后上升比较慢，考虑到电脑性能的问题，没有训练到最高的准确率），最终训练结果为cifar10达到了72%-73%，cifar100到达了48%-49%，emnist达到了76%-77%，对于自监督生成问题，所使用模型为Autoencoder，在不使用联邦学习的情况下取得了很好的结果（证明模型本身没有问题），又采用自己所写的给clinet分配数据集的算法设定总共500个client，并给他们分配数据，在这种情况下，训练了5500个epoch，也得到了不错的结果（我认为我所写的数据集分配方法没有问题），最后把模型、分配数据集结合起来后，能很模糊的看清里边确实有一定训练成果，但是更新很慢（而且autoencoder本身自己就几乎不怎么更新权重，deltat和vt基本上都是零）。
