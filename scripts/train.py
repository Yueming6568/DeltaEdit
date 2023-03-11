import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(".")
sys.path.append("..")

from datasets.train_dataset import TrainLatentsDataset
from options.train_options import TrainOptions
from delta_mapper import DeltaMapper

def main(opts):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = TrainLatentsDataset(opts)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=opts.batch_size,
                                  shuffle=True,
                                  num_workers=int(opts.workers),
                                  drop_last=True)

    #Initialze DeltaMapper
    net = DeltaMapper().to(device)

    #Initialize optimizer
    optimizer = torch.optim.Adam(list(net.parameters()), lr=opts.learning_rate)

    #Initialize loss
    l2_loss = torch.nn.MSELoss().to(device)
    cosine_loss = torch.nn.CosineSimilarity(dim=-1).to(device)

    #save dir
    os.makedirs(os.path.join(opts.checkpoint_path, opts.classname), exist_ok=True)

    for batch_idx, batch in enumerate(train_dataloader):

        latent_s, delta_c, delta_s = batch
        latent_s = latent_s.to(device)
        delta_c = delta_c.to(device)
        delta_s = delta_s.to(device)

        fake_delta_s = net(latent_s, delta_c)

        optimizer.zero_grad()
        loss_l2 = l2_loss(fake_delta_s, delta_s)
        loss_cos = 1 - torch.mean(cosine_loss(fake_delta_s, delta_s))

        loss = opts.l2_lambda * loss_l2 + opts.cos_lambda * loss_cos
        loss.backward()
        optimizer.step()

        if batch_idx % opts.print_interval == 0 :
            print(batch_idx, loss.detach().cpu().numpy(), loss_l2.detach().cpu().numpy(), loss_cos.detach().cpu().numpy())

        if batch_idx % opts.save_interval == 0:
            torch.save(net.state_dict(), os.path.join(opts.checkpoint_path, opts.classname, "net_%06d.pth" % batch_idx))

if __name__ == "__main__":
    opts = TrainOptions().parse()
    main(opts)