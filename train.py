import os
from training_functions import set_seed, run

if __name__ == '__main__':
    set_seed()
    encoder_filepath = os.path.join(
        # os.sep, "net","tscratch","people","plggwiazale", "AR-VAE", "first_working_models",
        os.sep, "home","gwiazale", "AR-VAE", "first_working_models",
        # "hyperparams_tuning_pepglider_original_mics_ar-vae_epoch980_encoder.pt"
        # "hyperparams_tuning_pepglider_mic_log2_01range_ar-vae_epoch2400_encoder.pt"
        # "hyperparams_tuning_pepglider_mic_with_signum_log2_01range_ar-vae_epoch2600_encoder.pt"
        # "hyperparams_tuning_pepglider_physchem_mic_01range_ar-vae_epoch1600_encoder.pt"
        # "hyperparams_tuning_pepglider_physchem_mic_log2_01range_ar-vae_epoch1900_encoder.pt"
        "hyperparams_tuning_pepglider_mic_with_signum_01range_ar-vae_epoch2000_encoder.pt"
        # "hyperparams_tuning_pepglider_physchem_mic_01range_ar-vae_epoch1500_encoder.pt"
    )
    decoder_filepath = os.path.join(
        # os.sep, "net","tscratch","people","plggwiazale", "AR-VAE", "first_working_models",
        os.sep, "home","gwiazale", "AR-VAE", "first_working_models",
        # "hyperparams_tuning_pepglider_original_mics_ar-vae_epoch980_decoder.pt"
        # "hyperparams_tuning_pepglider_mic_log2_01range_ar-vae_epoch2400_decoder.pt"
        # "hyperparams_tuning_pepglider_mic_with_signum_log2_01range_ar-vae_epoch2600_decoder.pt"
        # "hyperparams_tuning_pepglider_physchem_mic_01range_ar-vae_epoch1600_decoder.pt"
        # "hyperparams_tuning_pepglider_physchem_mic_log2_01range_ar-vae_epoch1900_decoder.pt"
        "hyperparams_tuning_pepglider_mic_with_signum_01range_ar-vae_epoch2000_decoder.pt"
        # "hyperparams_tuning_pepglider_physchem_mic_01range_ar-vae_epoch1500_decoder.pt"
    )
    # run(['positiv_negativ_AMPs'])
    run(['positiv_AMPs'], encoder_filepath, decoder_filepath)    
    # run(['positiv_AMPs'])

