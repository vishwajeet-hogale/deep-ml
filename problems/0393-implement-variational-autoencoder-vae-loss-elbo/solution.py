import numpy as np

def vae_loss(x: np.ndarray, x_reconstructed: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> tuple:
    """
    Compute the VAE loss (negative ELBO).

    Args:
        x: np.ndarray of shape (batch_size, features), original input
        x_reconstructed: np.ndarray of shape (batch_size, features), reconstructed input
        mu: np.ndarray of shape (batch_size, latent_dim), latent mean
        log_var: np.ndarray of shape (batch_size, latent_dim), latent log-variance

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence) as floats
    """
    reconstruction_loss = ((x - x_reconstructed) ** 2).sum(axis=-1).mean()
    latent_dim = mu.shape[1]
    
    d_per_sample = (-1/2) * (1 + log_var - mu**2 - np.exp(log_var)).sum(axis = -1)


    kl_divergence_loss = d_per_sample.mean()
    total_loss = reconstruction_loss + kl_divergence_loss
    return (total_loss, reconstruction_loss,kl_divergence_loss)
    