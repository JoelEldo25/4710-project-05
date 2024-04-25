from torch import tensor

data = tensor([1,2,3])
original = data.clone().detach()
print(data)
print(original)
data[1] = 1000
print(data)
print(original)