from  torch import tensor, nn, tanh, sign

device= "cuda"
#1 identical
# latent_code = tensor([[5],[5],[5],[5]])
# attribute_tensor = tensor([[10],[10],[10],[10]]).to(device)
#2 diff/random
latent_code = tensor([[5],[6],[7],[8]])
attribute_tensor = tensor([[10],[11],[12],[13]]).to(device)

# compute latent distance matrix
latent_code = latent_code.to(device).reshape(-1, 1)

lc_dist_mat = latent_code - latent_code.T


# compute attribute distance matrix
attribute_tensor = attribute_tensor.reshape(-1, 1)
attribute_dist_mat = attribute_tensor - attribute_tensor.T

# compute regularization loss
loss_fn = nn.L1Loss()
lc_tanh = tanh(lc_dist_mat * 1.0)
attribute_sign = sign(attribute_dist_mat)
sign_loss = loss_fn(lc_tanh, attribute_sign.float())
print(f'sign_loss {sign_loss}')