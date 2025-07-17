to run the captioning trainer use the format
python3 captioning_trainer.py VERSION EPOCHS Validation_frequency device

    -> VERSION is of the format camalayer_no.attenlayer_no.decoder_layerno.headno for eg 4.3.1.8 means the model has 4 CaMa layers, 3 Cross Attention layers 1 decoder layer and 8 heads in the encoder_transformer

    -> EPOCHS: no. of epochs

    -> Validation_frequency : the frequency at which validation is done for eg 5 means, it is done every 5 epochs 

    -> device set to 'cuda' or if multiple GPUs exist set to 'cuda:0' or 'cuda:1', I personally always use 'cuda:1'

activate the conda kernel before running, for eg:

conda activate Siraj && python3 captioning_trainer.py 4.3.1.16 25 2 'cuda:1'