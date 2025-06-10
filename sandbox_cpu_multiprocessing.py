# import multiprocessing as mp
# import data.data_describe as d
# import data.dataset as dataset_lib
# from torch import randint 
# import time

# start_time = time.time()
# batch = randint(1,21, (25,512))
# physchem_original_async = d.calculate_physchem_test(dataset_lib.decoded(batch, ""),)
# end_time = time.time()
# print(f'final time counting without pool: {end_time-start_time}')

# try:
#     mp.set_start_method('spawn')
# except RuntimeError:
#     pass    
# with mp.Pool(processes=8) as pool:
#     start_time = time.time()
#     batch = randint(1,21, (25,512))
#     physchem_original_async = d.calculate_physchem(pool, dataset_lib.decoded(batch, ""),) 
#     end_time = time.time()
#     print(f'final time counting with pool: {end_time-start_time}')

#final time counting without pool: 0.09377741813659668
#final time counting with pool: 0.16204023361206055


import multiprocessing as mp
import time
from torch import randn, exp, cat
from torch.distributions import Normal
import numpy as np
import ar_vae_metrics as m

batch_size_encoder_output = 512
latent_dim = 56
num_features_before_reshape = 10
num_dataloader_iterations = 17809

mu = randn(batch_size_encoder_output, latent_dim)
std = exp(0.5 * randn(batch_size_encoder_output, latent_dim))
q_distr = Normal(mu, std)

all_processed_z_samples = []
for _ in range(num_dataloader_iterations):
    z_raw_sample = q_distr.rsample((num_features_before_reshape,))
    z_reshaped = z_raw_sample.reshape(-1, z_raw_sample.shape[2])
    all_processed_z_samples.append(z_reshaped.cpu().detach())
latent_codes_torch = cat(all_processed_z_samples, dim=0)
latent_codes = latent_codes_torch.numpy()
attributes = np.random.normal(loc=0.0, scale=1.0, size=(44530, 4))
attr_list = ['Length']

start_time = time.time()
interp_metrics = m.compute_interpretability_metric(
    latent_codes, attributes, attr_list
)
ar_vae_metrics = {}
ar_vae_metrics["Interpretability"] = interp_metrics
ar_vae_metrics.update(m.compute_correlation_score(latent_codes, attributes))
ar_vae_metrics.update(m.compute_modularity(latent_codes, attributes))
ar_vae_metrics.update(m.compute_mig(latent_codes, attributes))
ar_vae_metrics.update(m.compute_sap_score(latent_codes, attributes))
end_time = time.time()
print(f'final time counting without pool: {end_time-start_time}')

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass    
with mp.Pool(processes=8) as pool:
    start_time = time.time()
    async_metrics = m.compute_all_metrics_async(pool, latent_codes, attributes, attr_list)
    end_time = time.time()
    print(f'final time counting with pool: {end_time-start_time}')