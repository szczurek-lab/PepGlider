from  torch import tensor, nn, tanh, sign

def compute_reg_loss(z, labels, reg_dim, gamma, device, factor=1.0):
    """
    Computes the regularization loss
    """
    x = z[:, reg_dim]
    reg_loss = reg_loss_sign(x, labels, device = device, factor=factor)
    return gamma * reg_loss

def reg_loss_sign(latent_code, attribute, device, factor=1.0):
    """
    Computes the regularization loss given the latent code and attribute
    Args:
        latent_code: torch Variable, (N,)
        attribute: torch Variable, (N,)
        factor: parameter for scaling the loss
    Returns
        scalar, loss
    """
    # compute latent distance matrix
    # print(f'latent_code shape {latent_code.shape}')
    latent_code = latent_code.to(device).reshape(-1, 1)
    # print(f'latent_code shape after reshape {latent_code.shape}')

    lc_dist_mat = latent_code - latent_code.T
    # print(f'latent_code shape {lc_dist_mat.shape}')


    # compute attribute distance matrix
    attribute_tensor = tensor(attribute).to(device)
    # print(f'attribute_tensor shape {attribute_tensor.shape}')
    attribute_tensor = attribute_tensor.reshape(-1, 1)
    # print(f'attribute_tensor shape after reshape {attribute_tensor.shape}')
    attribute_dist_mat = attribute_tensor - attribute_tensor.T
    # print(f'attribute_dist_mat shape {attribute_dist_mat.shape}')

    # compute regularization loss
    loss_fn = nn.L1Loss()
    lc_tanh = tanh(lc_dist_mat * factor)
    # print(f'lc_tanh shape {lc_tanh.shape}')
    attribute_sign = sign(attribute_dist_mat)
    # print(f'attribute_sign shape {attribute_sign.shape}')
    sign_loss = loss_fn(lc_tanh, attribute_sign.float())
    # print(f'sign_loss shape {sign_loss.shape}')
    # print(f'sign_loss {sign_loss}')
    return sign_loss.to(device)
