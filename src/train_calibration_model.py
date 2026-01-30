import torch
import numpy as np

import simulation
from models.microcal_models import CalibrationModel


#hyperparameter
emsc_range = np.array([[0.9, 1.1], [.0, .05], [-0.001, 0.0001], [-0.00005, 0.00005]])
max_noise = 0.001

lr = 1e-4
epochs = 10
batch_size = 8


# define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cal_model = CalibrationModel(out_layers=1).to(device).float()
cal_model.train()

objective_function = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(cal_model.parameters(), lr=lr)


# load training data (just small set as minimal example)
data = np.load('data/lipid_hts_wn.npz')
macro_spectra = data['hts']            # macroscopic spectra
reference_analysis = data['lipid']      # reference analysis values
wn = data['wn']                        # wavenumbers


# train model
for i in range(epochs):
    
    emsc_params = np.array([np.random.uniform(low, high, batch_size) 
                            for low, high in emsc_range])[:, :, None]

    batch_idc = np.random.randint(0, reference_analysis.shape[0], batch_size)
    
    spectra_batch = macro_spectra[batch_idc, :]
    spectra_batch = simulation.add_polynomial(spectra_batch, wn, emsc_params)
    spectra_batch = simulation.add_whitenoise(spectra_batch, max_noise)
    
    input_tensor = torch.from_numpy(spectra_batch).to(device).float().unsqueeze(1)
    reference_tensor = torch.from_numpy(reference_analysis[batch_idc]).to(device).float()

    optimizer.zero_grad()
    outputs = cal_model(input_tensor)
    loss = objective_function(outputs.squeeze(), reference_tensor)
    loss.backward()
    optimizer.step()
