import torch

# using pytorch's built in funcs to check the work I did with numpy

xi = torch.tensor([6, 4, 7, 2, 1], dtype=torch.float32)
yi = torch.tensor([0, 1, 0], dtype=torch.float32)

W = torch.tensor([[0, 4, -4, 5, 4],
                  [2, 2, -4, 3, 0]], dtype=torch.float32, requires_grad=True)
V = torch.tensor([[1, -2],
                  [2,  3],
                  [1,  2]], dtype=torch.float32, requires_grad=True)

# forward pass
# still excluding biases
h = torch.relu(W @ xi)
s = torch.sigmoid(V @ h)
p = torch.softmax(s, dim=0)

# loss
loss = -yi @ torch.log(p)

# compute gradients
loss.backward()

print("grad_W:", W.grad)
print("grad_V:", V.grad)