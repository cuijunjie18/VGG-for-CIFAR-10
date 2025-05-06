from frame_special import *
from VGG import *

# 加载原始数据
train_arrays,train_labels = load_data()
test_arrays = load_test()

# 构造训练数据迭代器
batch_size = 64
train_dataset = TrainDataset(train_arrays,train_labels)
train_iter = data.DataLoader(train_dataset,batch_size,shuffle = True)

# 构造测试数据迭代器
test_dataset = TestDataset(test_arrays)
test_iter = data.DataLoader(test_dataset,batch_size,shuffle = False)

# 搭建网络
conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))
object_map = get_map()
net = vgg(conv_arch,in_H = 244,in_W = 244) # 调整大小

# 定义超参数
num_epochs = 20
lr = 3e-5
loss_fn = nn.CrossEntropyLoss()

# 模型训练
loss_plt = train_resize(net,train_iter,lr,num_epochs,loss_fn)

# 保存模型
torch.save(net,"models/vgg11.pt")
print(f"model save successfully!")

