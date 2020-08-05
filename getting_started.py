from __future__ import print_function
import torch

x = torch.empty(5, 3)  #uninitilazed 5x3 matrix
#print(x)

x = torch.rand(5, 3)  #initialized matrix
#print(x)

x = torch.zeros(5, 3, dtype=torch.long)
#print(x)

x = torch.tensor([5.5, 3])
#print(x)

x = x.new_ones(5, 3, dtype=torch.double)
#print(x)

x = torch.randn_like(x, dtype=torch.float)
#print(x)
#print(x.size())

y = torch.rand(5, 3)
#print(x + y)
#print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
#print(result)

y.add_(x)   #addition in place
#print(y)

#print(x[:1])

x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1, 8)

#print(x.size(), y.size(), z.size())

x = torch.randn(1)
#print(x)
#print(x.item())

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)

print(a)
print(b)