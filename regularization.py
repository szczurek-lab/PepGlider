from  torch import tensor, nn, tanh, sign

def compute_reg_loss(z, labels, reg_dim, gamma, gamma_multiplier, device, factor=1.0, factor_multiplier=1.0):
    """
    Computes the regularization loss
    """
    x = z[:, reg_dim]
    reg_loss = reg_loss_sign(x, labels, device = device, factor=factor, factor_multiplier = factor_multiplier)
    return (gamma*gamma_multiplier) * reg_loss, reg_loss

def reg_loss_sign(latent_code, attribute, device, factor=1.0, factor_multiplier=1.0):
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
    latent_code = latent_code.to(device).reshape(-1, 1)
    lc_dist_mat = latent_code - latent_code.T

    # compute attribute distance matrix
    attribute_tensor = tensor(attribute).to(device)
    attribute_tensor = attribute_tensor.reshape(-1, 1)
    attribute_dist_mat = attribute_tensor - attribute_tensor.T

    # compute regularization loss
    loss_fn = nn.L1Loss()
    lc_tanh = tanh(lc_dist_mat * factor * factor_multiplier)
    attribute_sign = sign(attribute_dist_mat)
    sign_loss = loss_fn(lc_tanh, attribute_dist_mat.float())
    return sign_loss.to(device)
