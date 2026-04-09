import torch
ckpt = torch.load('./outputs_msvqvae_128/ae_K128_epoch020.pt', map_location='cpu', weights_only=False)
print('AE keys:', list(ckpt.keys()))
prior = torch.load('./outputs_msvqvae_128/priors_K128_epoch015.pt', map_location='cpu', weights_only=False)
print('Prior keys:', list(prior.keys()))
