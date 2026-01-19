import os
from training_functions import set_seed, run

if __name__ == '__main__':
    set_seed()
    encoder_filepath = os.path.join(
        # os.sep, "net","tscratch","people","plggwiazale", "AR-VAE", "first_working_models",
        os.sep, "home","gwiazale", "AR-VAE", "first_working_models",
        "hyperparams_tuning_pepglider_original_mics_ar-vae_epoch980_encoder.pt"
    )
    decoder_filepath = os.path.join(
        # os.sep, "net","tscratch","people","plggwiazale", "AR-VAE", "first_working_models",
        os.sep, "home","gwiazale", "AR-VAE", "first_working_models",
        "hyperparams_tuning_pepglider_original_mics_ar-vae_epoch980_decoder.pt"
    )
    # run(['positiv_negativ_AMPs'])
    run(['positiv_AMPs'], encoder_filepath, decoder_filepath)    
    # run(['positiv_AMPs'])

