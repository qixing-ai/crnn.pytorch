import alphabets

# 关于数据和网络
alphabet = alphabets.alphabet
keep_ratio = False # 是否按比例调整图像大小
manualSeed = 1234 # 人工种子
random_sample = True # 是否使用随机采样器对数据集进行采样
imgH = 32 # 输入图像到网络的高度
imgW = 100 # 输入图像到网络的宽度
nh = 256 # lstm隐藏状态的大小
nc = 1
pretrained = 'expr/netCRNN_99_50.pth' # 预训练模型的路径(继续训练)
expr_dir = 'expr' # 哪里存放样品和模型
dealwith_lossnan = False # 是否将梯度中的所有nan/inf替换为零

# 计算机硬件
cuda = True # 支持cuda
multi_gpu = False # 是否使用multi gpu
ngpu = 1 # 要使用的gpu数量。记得设置multi_gpu为true
workers = 0 # 数据加载工人的数量

# 训练参数
displayInterval = 10 # 打印时间间隔
valInterval = 10 # 计算模型损失与精度的时间间隔
saveInterval = 40 # 模型保存间隔
n_val_disp = 20 # 验证模型的样本数量

# 微调
nepoch = 1000 # 要为几个时代而训练
batchSize = 64 # 输入批大小
lr = 0.0001 # 评论家的学习速度，不被adadealta使用
beta1 = 0.5 # beta1亚当。默认= 0.5
adam = False # 是否使用adam(默认是rmsprop)
adadelta = False # 是否使用adadelta(默认是rmsprop)