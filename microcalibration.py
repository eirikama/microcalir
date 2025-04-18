import pandas as pd
from scipy.io import loadmat
import numpy as np
from scipy.signal import savgol_filter
import torch

import matplotlib.pyplot as plt 
import matplotlib as ml
plt.style.use("default")

from biospectools.preprocessing import emsc
from biospectools.utils.interpolate import interp2wns

from models.microcal_models import calibration_model
from models.microcal_models import micro_to_macro_model
from utils import get_Opus_data


wn = np.load("data/wn.npz")["wn"]

filename = 'data/test.0'
im, wn_fungi = get_Opus_data(filename)
spectra, _ = interp2wns(wn_fungi, wn, im.reshape(im.shape[0]*im.shape[1], -1))
im = spectra.reshape(im.shape[0], im.shape[1], -1)
plt.imshow(im[:, :, np.argmin(np.abs(1745 - wn_fungi))], cmap="coolwarm")
plt.colorbar()
plt.show()


def predict_microcal_im(transferred_im, weights_path, device="cuda"):
    mc_model = calibration_model(out_layers=1).to(device);
    mc_model.load_state_dict(torch.load(weights_path, weights_only=True))
    mc_model.eval();
    with torch.no_grad():
        mc_im = []
        for i, spec in enumerate(transferred_im):
            print(i, end="\r")
            line = []
            for s in spec:
                c = mc_model(torch.from_numpy(s[None, None, :]).float().to(device))
                line.append(c.detach().cpu().numpy().squeeze())
            mc_im.append(line)
    return np.array(mc_im)



device = "cuda"
transfer_model = micro_to_macro_model().float().to(device)
transfer_model.load_state_dict(torch.load(f"weights/scatteredFPAh_to_HTS_1903.t7", weights_only=True))
transfer_model.eval();

with torch.no_grad():
    transfer_im = []
    for i, spec in enumerate(im):
        print(i, end="\r")
        corr_line = []
        for s in spec:
            c = transfer_model(torch.from_numpy(s[None, None, :]).float().to(device))
            corr_line.append(c.detach().cpu().numpy().squeeze())
        transfer_im.append(corr_line)
    transfer_im = np.array(transfer_im)

lipid_im = predict_microcal_im(transfer_im, "weights/pre_lipid_MC1_Ca1_Pi1.t7", device = "cuda")
ga_im = predict_microcal_im(transfer_im, "weights/pred_GA.t7", device = "cuda")
pfa_im = predict_microcal_im(transfer_im, "weights/pred_PFA_MC1_Ca1.t7", device = "cuda")
sfa_im = predict_microcal_im(transfer_im, "weights/pred_SFA_MC1_Ca1.t7", device = "cuda")





