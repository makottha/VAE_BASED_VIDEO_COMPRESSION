import sys, torch
sys.path.insert(0, '.')
from dataset_motion import FrameDiffDataset
from models import MSVQVAE2Video
from losses import recon_loss_fn
from torch.utils.data import DataLoader

device = torch.device('cuda')
ds = FrameDiffDataset('../dataset_64/train/tensors', subset_fraction=0.01, shuffle=False)
dl = DataLoader(ds, batch_size=2, shuffle=False)
batch = next(iter(dl)).to(device)
print(f'Batch: {batch.shape}  range: [{batch.min():.3f}, {batch.max():.3f}]')

ae = MSVQVAE2Video(top_dim=256, bottom_dim=128, K_top=256, K_bottom=256,
    commit_top=0.25, commit_bottom=0.25, norm='gn', use_vgg=False,
    soft_temp=1.0, use_ema=True).to(device)
ae.train()
out = ae(batch)
print(f'Recon: {out["recon"].shape}  entropy_top: {out["entropy_top"]:.3f}')

loss = recon_loss_fn(out['recon'], batch, mode='l1') + 0.25 * out['vq_loss']
loss.backward()
print(f'Loss: {loss.item():.4f}  — forward+backward OK')
print('All good. Ready to train.')
