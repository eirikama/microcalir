import torch
import numpy as np

from models.microcal_models import MicroToMacroModel
import simulation

# hyperparameters
emsc_range = np.array([[0.5, 1.5], [.0, .3], [-0.01, 0.01], [-0.0005, 0.0005]])
max_noise = 0.01

lr = 1e-4
batch_size = 8
epochs = 10

# define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transfer_model = MicroToMacroModel().to(device).float()
objective_function = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(transfer_model.parameters(), lr=lr)


# load training data (just small set as minimal example)
data = np.load('data/fpah_hts_wn.npz')

hts_spectra = data['hts']        # macroscopic spectra
fpah_spectra = data['fpah']      # homogonized microscopic spectra
wn = data['wn']                  # wavenumbers

N = fpah_spectra.shape[0]


# train model
transfer_model.train()
for _ in range(epochs):

    indc = np.random.randint(0, N, batch_size)
    fpah, hts = fpah_spectra[indc, :], hts_spectra[indc, :]

    theta_max = np.random.uniform(0.1, 0.2)
    n0s, rs, n_ims, hs, scatt_coeffs = [np.random.uniform(low, high, (batch_size, 1)) 
                                        for low, high in [(1.25, 1.5), (3, 12), (1e-5, 1e-2), (1., 2.), (1., 2.)]]
    emsc_params = np.array([np.random.uniform(low, high, batch_size) for low, high in emsc_range])[:, :, None]

    scatt = simulation.add_scattering(fpah, wn, rs, n0s, n_ims, theta_max, hs, scatt_coeffs)
    scatt = simulation.add_polynomial(scatt, wn, emsc_params)
    scatt = simulation.add_whitenoise(scatt, max_noise)

    input_tensor = torch.from_numpy(scatt).float().to(device).unsqueeze(1)
    hts_tensor = torch.from_numpy(hts).float().to(device).unsqueeze(1)

    optimizer.zero_grad()
    corr = transfer_model(input_tensor)
    
    loss = objective_function(corr, hts_tensor)
    loss.backward()
    optimizer.step()
