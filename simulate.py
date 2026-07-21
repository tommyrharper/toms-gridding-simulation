"""
Ideal, fully-analytic visibility + dirty-image simulator for the gridding ML testbed.

Everything here is EXACT, no gridding approximation anywhere:
  - a point source is a delta in the image  ->  its visibility is an analytic fringe
  - the sampling function S(u,v) is just the set of uv points we choose to measure
  - the dirty image is a brute-force DFT (the T1 "slow map")

Textbook picture:
    V_measured(u,v) = V_true(u,v) * S(u,v)
    dirty_image     = true_sky (*) dirty_beam,   dirty_beam = FT of S(u,v)

Conventions match the Imaging-Tutorial (Ye et al. 2019):
  u,v in wavelengths;  vis = sum_k S_k exp(+2 pi i (u l + v m));
  dirty image  I(x,y) = (1/sum w) Re sum_k w_k V_k exp(+2 pi i (u_k x + v_k y)).
Narrow field: w-term ignored (n ~ 1).
"""
import numpy as np

ARCSEC = np.pi / (180.0 * 3600.0)   # radians per arcsec
