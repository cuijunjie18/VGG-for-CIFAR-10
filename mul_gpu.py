train_datasets = customData(train_txt)  #创建dataset
train_dataloaders = torch.utils.data.DataLoader(train_datasets,opt.batch_size,num_workers=train_num_workers,shuffle=True)  #创建dataloader
model = efficientnet_b0(num_classes = opt.num_class)  #创建model
device_list = list(map(int,list(opt.device_id)))
print("Using gpu"," ".join([str(v) for v in device_list]))
device = device_list[0]  #主GPU，也就是分发任务和结果回收的GPU，也是梯度传播更新的GPU
model = torch.nn.DataParallel(model,device_ids=device_list)
model.to(device)
 
for data in train_dataloaders: 
   model.train(True)
   inputs, labels = data
   inputs = Variable(inputs.to(device))  #将数据放到主要GPU
   labels = Variable(labels.to(device)) 