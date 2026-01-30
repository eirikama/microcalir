import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib as ml
plt.style.use("default")

from biospectools.preprocessing import emsc
from biospectools.utils.interpolate import interp2wns

from models.microcal_models import CalibrationModel, MicroToMacroModel


wn = np.load('data/wn.npz')['wn']   # wn range used for models

df = pd.read_csv('data/test.csv')
wn_fungi = df.columns.values.astype('float')
spectra = df.values

spectra, _ = interp2wns(wn_fungi, wn, spectra)
im = spectra.reshape(128, 128, -1)
plt.imshow(im[:, :, np.argmin(np.abs(1745 - wn))], cmap='coolwarm')
plt.colorbar()
plt.show()


def predict_microcal_im(transferred_im, weights_path, device="cuda"):
    cal_model = CalibrationModel(out_layers=1).to(device).float();
    cal_model.load_state_dict(torch.load(weights_path, weights_only=True))
    cal_model.eval();
    with torch.no_grad():
        mc_im = []
        for i, spec in enumerate(transferred_im):
            print(f' {i} / {transferred_im.shape[0]}   ', end='\r')
            line = []
            for s in spec:
                c = cal_model(torch.from_numpy(s[None, None, :]).float().to(device))
                line.append(c.detach().cpu().numpy().squeeze())
            mc_im.append(line)
    return np.array(mc_im)



device = "cuda"
transfer_model = MicroToMacroModel().float().to(device).float()
transfer_model.load_state_dict(torch.load(f"weights/scatteredFPAh_to_HTS.t7", weights_only=True))
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

lipid_im = predict_microcal_im(transfer_im, "weights/pred_lipid.t7", device = "cuda")
ga_im = predict_microcal_im(transfer_im, "weights/pred_GA.t7", device = "cuda")
pfa_im = predict_microcal_im(transfer_im, "weights/pred_PFA.t7", device = "cuda")
sfa_im = predict_microcal_im(transfer_im, "weights/pred_SFA.t7", device = "cuda")





