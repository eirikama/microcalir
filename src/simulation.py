from typing import Union as U
import numpy as np
import scipy.special
import scipy.integrate
from scipy.signal import hilbert

def as_ndarrays(*args):
    return [np.asarray(a) for a in args]

def get_nkk(imag_part, wavelengths: np.ndarray, pad_size=200):

    pad_last_axis = [(0, 0)] * imag_part.ndim
    pad_last_axis[-1] = (pad_size, pad_size)
    nkk = np.imag(hilbert(np.pad(imag_part, pad_last_axis, mode='edge')))
    nkk = nkk[..., pad_size:-pad_size]

    wls_increase = wavelengths[..., 0] < wavelengths[..., -1]
    if wls_increase:
        return nkk.copy()
    else:
        return -nkk
        
def get_imagpart(pure_absorbance, wavelength, radius, factor=1):

    deff = np.pi / 2 * radius * factor
    imagpart = (pure_absorbance * np.log(10)) / \
               (4 * np.pi * deff / wavelength)
    return imagpart

def q_ext_sca_na(
        m: U[float, np.ndarray],
        lambd: U[float, np.ndarray],
        r: U[float, np.ndarray],
        theta_na: U[float, np.ndarray, list] = 0.4,
        theta_resolution: int = 12,
        theta_min: float = 0):
    
    m, lambd, r = as_ndarrays(m, lambd, r)
    np.broadcast(m, lambd, r)  

    x = 2 * np.pi * r / lambd

    x_max = np.max(x)
    nmax = int(round(2 + x_max + 4 * x_max ** (1 / 3)))
    n = np.arange(nmax + 1)

    an, bn = mie_abcd(m[..., np.newaxis], x[..., np.newaxis], n)

    if isinstance(theta_na, (int, float)):
        thetas = np.linspace(theta_min, theta_na, theta_resolution)
    elif isinstance(theta_na, (list, np.ndarray)):
        thetas = theta_na
    else:
        raise ValueError(f'Unexpected type of theta_na: {type(theta_na)}')
    pi_n, tao_n = mie_pi_tao(thetas, nmax)

    cn = 2 * n[1:] + 1
    inv_x2 = 1 / x ** 2
    qext = 2 * inv_x2 * np.sum(cn * (an.real + bn.real), axis=-1)
    qsca_total = 2 * inv_x2 * np.sum(cn * ((an * an.conj()).real +
                                           (bn * bn.conj()).real), axis=-1)

    scn = cn / (n[1:]*(n[1:] + 1))
    san = an[..., None, :]
    sbn = bn[..., None, :]
    s1 = np.sum(scn * (san * pi_n + sbn * tao_n), axis=-1)
    s2 = np.sum(scn * (sbn * pi_n + san * tao_n), axis=-1)
    qsca = (s1*s1.conj() + s2*s2.conj()).real * np.sin(thetas)
    qsca = inv_x2 * scipy.integrate.trapz(qsca, thetas, axis=-1)
    
    return qext, qsca_total, qsca


def mie_pi_tao(thetas, nmax):
    cos_t = np.cos(thetas)
    ns = np.arange(1, nmax + 1)
    pi_n = np.stack([scipy.special.lpmn(0, nmax, c)[1][0] for c in cos_t], axis=0)
    tao_n = ns * cos_t[:, None] * pi_n[..., 1:] - (ns + 1) * pi_n[..., :-1]
    return pi_n[..., 1:], tao_n


def mie_abcd(m: np.ndarray, x: np.ndarray, n: np.ndarray):
    z = m * x

    bx = scipy.special.spherical_jn(n, x)
    bz = scipy.special.spherical_jn(n, z)
    yx = scipy.special.spherical_yn(n, x)
    hx = bx[..., 1:] + 1j * yx[..., 1:]

    bx[..., 0] = (np.sin(x) / x)[..., 0]
    bz[..., 0] = (np.sin(z) / z)[..., 0]
    yx[..., 0] = (-np.cos(x) / x)[..., 0]
    b1x = bx[..., :-1]
    b1z = bz[..., :-1]
    y1x = yx[..., :-1]
    h1x = b1x + 1j * y1x

    bx = bx[..., 1:]
    bz = bz[..., 1:]
    n = n[1:]

    ax = x * b1x - n * bx
    az = z * b1z - n * bz
    ahx = x * h1x - n * hx

    m2 = m ** 2
    an = (m2 * bz * ax - bx * az) / (m2 * bz * ahx - hx * az)
    bn = (bz * ax - bx * az) / (bz * ahx - hx * az)

    return an, bn
    
    
def add_scattering(
    spec, wn, r, n0, n_im, theta_max, h, scatt_coeff, theta_res=30
):
    
    n_const = n0 + n_im * 1j
    wls = 10e+3 / wn[None]
    
    n_i = get_imagpart(spec, wls, r, factor=h)
    n_r = get_nkk(n_i, wls.squeeze())      
    ms = n_const + n_r + 1j * n_i

    Qext, Qsca, QscaNA = q_ext_sca_na(
        ms,
        wls,
        r,
        theta_na=theta_max,
        theta_resolution=theta_res,
    )

    A = Qsca - QscaNA + scatt_coeff * (Qext - Qsca)
    A = -np.log10((1 - 0.6 * A / np.abs(A).max(axis=1, keepdims=True)))
        
    return A
    
    
def add_whitenoise(spectra, max_noise):
    
    spectra += np.random.normal(
            np.zeros(spectra.shape),
            np.random.uniform(0, max_noise, spectra.shape[0])[:, None],
            spectra.shape,
        )
    
    return spectra

def add_polynomial(spectra, wn, params):

    half_rng = np.abs(wn[0] - wn[-1]) / 2
    norm_wn = (wn - np.mean(wn)) / half_rng

    return params[1] * spectra + params[0] + params[2] * norm_wn + params[3] * norm_wn ** 2