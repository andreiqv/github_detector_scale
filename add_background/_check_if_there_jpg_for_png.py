import os
png = os.listdir('.')
png = [x[:-4] for x in png]

train = os.listdir('/home/andrei/Data/Datasets/ScalesDetector/dataset-splited/train/')
train = list(filter(lambda x : os.path.splitext(x)[1]=='.jpg' , train))
train = [x[:-4] for x in train]

sp = set(png)
st = set(train)
print(sp < st)
