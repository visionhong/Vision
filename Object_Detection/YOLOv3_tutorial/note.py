import torch
import numpy as np

# predic = torch.rand((4,10,85))
# mask = (predic[:,:,4] > 0.5).float().unsqueeze(2)
#
# print(predic.new(predic.shape))


# ten = np.array([5,9,1,5,4,2,6,1,0,6]).reshape(-1,1)
# tensor_ten = torch.from_numpy(ten)
# ten1 = torch.from_numpy(np.unique(ten))
#
#
# tensor_res = tensor_ten.new(ten1.shape) # unique값 개수만큼의 텐서
# print(tensor_res)
# print(tensor_res.copy_(ten1))

#
# a = torch.randn(8,7)
# aa = torch.randint(5, (8,7))
# mask = torch.nonzero(aa[:,-2]).squeeze()
#
# print(torch.nonzero(aa[:,-2]).squeeze())
# print(a[mask].shape)

# add = torch.tensor([4,3,1,2])
# index = torch.sort(add, descending=True)[1]
# print(index)
# # index = [2,0,1]
# print(add[index])

# print(torch.clamp(torch.tensor([5]), min = 0))
a = torch.tensor([[2,0,3],[0,2,1]])
print(torch.nonzero(a[:,2]).squeeze())
