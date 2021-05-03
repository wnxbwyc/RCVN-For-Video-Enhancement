import torch
import models.RCAN_A
import models.RCVN_B
import models.RCVN_C

def make_model(opt):
    device = torch.device('cpu' if opt.cpu else 'cuda')
    if (opt.model.lower() == 'rcan_a'):
        return RCAN_A.make_model(opt).to(device)
    elif (opt.model.lower() == 'rcvn_b'):
        return RCVN_B.make_model(opt).to(device)
    elif (opt.model.lower() == 'rcvn_c'):
        return RCVN_C.make_model(opt).to(device)
