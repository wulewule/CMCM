The checkpoint stores the pretrained weights of the autoencoder, corresponding to autoencoder_kl_32x32x4.yaml, available at https://github.com/CompVis/latent-diffusion.

First Stage: autoencodetrain.py is used to train the autoencoder.
Second Stage: cmcm.py is used to train CMCM.
The code is based on LDM. For details, refer to https://github.com/CompVis/latent-diffusion.
