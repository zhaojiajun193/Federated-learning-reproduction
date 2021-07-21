联邦学习论文复现
===
对2021年Google发表在ICLR上的”Adaptive Federated Optimization“一文进行了三种算法在Emnist、Cifar10、Cifar100上的pytorch复现工作，由于本人能力有限，仍未一名本科生，并未做的十分出色，对于NLP上StackOverflow、Shakespeare数据集由于缺乏处理经验并未完成，如果您对此感兴趣欢迎与我联系共同探讨（联系方式xdu1816@qq.com），在此将我的代码与大家分享，欢迎批评指导。

代码思路
----

由于时间原因，后续进行详细更新，各种代码大同小异，不一一赘述。

split_and_shuffle_labels、get_iid_subsamples_indices、create_iid_subsamples均为数据集分配函数

select_client函数为选择每一轮此参与联邦学习训练的client，返回一个数组

get_new_main_model为根据论文中伪代码，将一个轮次训练完成后各个client的model以及之前的main_model在main_model上进行梯度下降

create_model_optimizer_criterion_dict为创建各个参与训练的client的模型、优化算法以及损失函数

send_main_model_to_nodes_and_update_model_dict为在每一轮开始向各个client分配新的main_model，方法较为笨拙，一层一层的将权重赋给其他模型

start_train_end_node_process_without_print为开始训练

训练过程中参数取自于论文中所给的最优参数，并未多次训练寻找最优参数或者加入优化参数的算法。

所有工作均为本人个人完成，之前有参考federated Average中pytorch的复现结果，但是对应的代码找不到了，之后找到会更新到这。

模型的创建并未与模型训练放到一个代码文件中，采用在模型训练代码中引用模型创建函数的形式创建相应模型，模型均为该论文所给模型，欢迎各位在此基础上更新迭代。

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
