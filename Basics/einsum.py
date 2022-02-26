import torch

x = torch.rand((2, 3))
y = torch.tensor([[1,2,3],[4,5,6]])

print(x)
# Permutation of Tensors
print(torch.einsum('ij->ji', x))

# Summation
print(torch.einsum('ij->', x))

# Column Sum
print(torch.einsum('ij->j', x))

# Row Sum
print(torch.einsum('ij->i', x))

# Matrix-Vector Multiplication
v = torch.rand((1, 3))
print(torch.einsum('ij, kj->ik', x,v))  # (2,3) x (3,1) = (2,1)

print(y)
# Dot product first row with first row of matrix
print(torch.einsum('i,i->', y[0], y[0]))

# Dot product with matrix
print(torch.einsum('ij,ij->', y, y))  # arrow 뒤에 값이 없으면 dimension이 없다는 의미가 되므로 값이 한개만 나옴

# Hadamard Product (element-wise multiplication)
print(torch.einsum('ij,ij->ij', y, y))

# Outer Product
a = torch.rand((3))
b = torch.rand((5))
print(torch.einsum('i,j->ij', a, b))  # 3x5 텐서 생성

# Batch Matrix Multiplication  same torch.bmm
a = torch.rand((3, 2, 5))
b = torch.rand((3, 5, 3))
print(torch.einsum('ijk,ikl->ijl', a, b).shape)  # shape(3, 2, 3)

# Matrix Diagonal(대각)
x = torch.rand((3, 3))
print(x)
print(torch.einsum('ii->i', x))

# Matrix Trace
print(torch.einsum('ii->', x))  # 대각의 합
