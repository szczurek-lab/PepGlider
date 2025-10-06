import os
from training_functions import set_seed, run

if __name__ == '__main__':
    set_seed()
    # print('AMPs/nonAMPs')
    # run(['positiv_negativ_AMPs'])
    # run(['positiv_AMPs'], encoder_filepath, decoder_filepath)
    run(['positiv_AMPs'])
    # run(['positiv_negativ_AMPs'], encoder_filepath, decoder_filepath)
    #run(['uniprot'], encoder_filepath, decoder_filepath)
    # print('merged data AMPs/nonAMPs + Uniprot')
    # run(['uniprot','positiv_negativ_AMPs'], encoder_filepath, decoder_filepath)
