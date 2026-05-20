# Microcalibration for Infrared Spectroscopy (microcalIR)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/paper-Analytical%20Chemistry%202025-red)](https://doi.org/10.1021/acs.analchem.5c03049)
[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.analchem.5c03049-orange)](https://doi.org/10.1021/acs.analchem.5c03049)


A deep learning-based **micro-calibration transfer** framework for quantitative chemical analysis in infrared microscopic imaging. Enables spatially resolved, pixel-level quantitative IR spectroscopy by bridging the domain gap between macroscopic bulk measurements and microscopic hyperspectral images. Published in *Analytical Chemistry* (2025).


<img border="0" align="Center" src="microcalibration_illustration.svg" alt="Microcalibration illustration" width=100%/>


## Overview

Infrared spectroscopy of bulk samples can be calibrated against reference measurements (e.g. lipid profiles from gas chromatography) to produce fast, quantitative analytical models. However, directly applying these models to IR microscopy images is not feasible as the pixel spectra from microscopic measurements occupy a fundamentally different domain to macroscopic spectra.

**microcalir** solves this with a deep learning calibration transfer model that maps microscopic pixel spectra into the domain of macroscopic spectra, allowing existing bulk calibration models to be applied directly to hyperspectral IR images. This enables **spatially resolved quantitative chemical analysis** at the pixel level, something not previously possible with standard IR calibration workflows.

The approach is validated on IR microspectroscopic data of oleaginous filamentous fungi, calibrated toward lipid profiles (gas chromatography) and glucosamine content.

**Key capabilities:**
- Transfers microscopic pixel spectra to the macroscopic spectral domain
- Enables pixel-level quantitative predictions from bulk-calibrated PLS models
- Applicable to any hyperspectral FTIR microscopy dataset
- Pretrained weights included for immediate use

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{magnussen2025microcalibration,
  title   = {Calibration for Quantitative Chemical Analysis in IR Microscopic Imaging},
  author  = {Magnussen, Eirik Almklov and Zimmermann, Boris and Dzurendova, Simona
             and Slany, Ondrej and Tafintseva, Valeria and Liland, Kristian Hovde
             and T{\o}ndel, Kristin and Shapaval, Volha and Kohler, Achim},
  journal = {Analytical Chemistry},
  volume  = {97},
  number  = {40},
  pages   = {21947--21955},
  year    = {2025},
  doi     = {10.1021/acs.analchem.5c03049}
}
```

---

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material for any purpose, provided appropriate credit is given.
