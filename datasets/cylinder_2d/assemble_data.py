import pyvista as pv
import torch
import pickle as pkl
import os

t_train_val = 1301
t_val_test = 1401
n_tstep = 1501
x_ulim = 320 # up to 640

true_t = torch.linspace(0, 15, n_tstep)

#train_x = torch.zeros(1, t_train_val, 2, 80, x_ulim)
train_f = torch.zeros(1, t_train_val, 1)
train_mu = torch.zeros(1, 1)
train_t = true_t[0:t_train_val].unsqueeze(0)

#val_x = torch.zeros(1, t_val_test - t_train_val, 2, 80, x_ulim)
val_f = torch.zeros(1, t_val_test - t_train_val, 1)
val_mu = torch.zeros(1, 1)
val_t = true_t[t_train_val:t_val_test].unsqueeze(0)

#test_x = torch.zeros(1, n_tstep - t_val_test, 2, 80, x_ulim)
test_f = torch.zeros(1, n_tstep - t_val_test, 1)
test_mu = torch.zeros(1, 1)
test_t = true_t[t_val_test:].unsqueeze(0)

filename = "cylinder2d.vti"
mesh = pv.read(filename)
print(mesh)
u = mesh["u"].reshape(1501, 80, 640)
v = mesh["v"].reshape(1501, 80, 640)
print(u.shape, v.shape)

x = torch.stack([torch.tensor(u), torch.tensor(v)], dim=1).unsqueeze(0)
print(x.shape)
train_x = x[:, 0:t_train_val, :, :, :x_ulim].clone()
val_x = x[:, t_train_val:t_val_test, :, :, :x_ulim].clone()
test_x = x[:, t_val_test:, :, :, :x_ulim].clone()
print(train_x.shape, val_x.shape, test_x.shape)

data = {"train_mu": train_mu,
        "train_t": train_t,
        "train_x": train_x,
        "train_f": train_f,
        "val_mu": val_mu,
        "val_t": val_t,
        "val_x": val_x,
        "val_f": val_f,
        "test_mu": test_mu,
        "test_t": test_t,
        "test_x": test_x,
        "test_f": test_f}

os.system("rm data.pkl")
with open("data.pkl", "wb") as f:
    pkl.dump(data, f, protocol=4)

print("Data assembled and saved.")