import numpy as np
import matplotlib.pyplot as plt
from arrays import list_arrays

RADIO = ["vla.a", "vla.b", "vla.c", "vla.d", "vlba", "ngvla-revC", "wsrt", "atca_6a", "meerkat"]

# ─────────────────────────────────────────────────────────────────────────────
#  Antenna configurations in configs/  (241 total; run list_arrays() for them all).
#  Pick one by its file stem, e.g.  array = "vla.a".   Resolution ∝ 1 / max-baseline.
# ─────────────────────────────────────────────────────────────────────────────
#  VLA — Very Large Array, New Mexico, 27×25 m, cm (1–50 GHz), reconfigurable "Y":
#     vla.a   max baseline 36 km   highest res (~0.04–1")  compact / point sources
#     vla.b   max baseline 11 km   high res
#     vla.c   max baseline 3.4 km  moderate res
#     vla.d   max baseline 1.0 km  lowest res — extended emission, large-scale flux
#     vla.bna vla.cnb vla.dnc      hybrid (extended North arm) — low / southern Dec
#  VLBA — Very Long Baseline Array, 10 antennas across the continent:
#     vlba    ~8600 km baselines   milli-arcsec res — AGN cores, masers, astrometry
#  ngVLA — next-generation VLA (planned, use for forecasts):
#     ngvla-revC / -revD           full array
#     ngvla-core/-plains/-mid/-main/-lba/-sba/-gb-vlba   sub-arrays
#  ALMA — Atacama mm/submm, 50×12 m, Chile:
#     alma.cycleN.M                N = cycle, M = 1 (compact ~0.16 km) … 10 (~16 km)
#     aca.* , aca.cycle*           Atacama Compact Array (7 m) — large-scale mm flux
#  Other radio:
#     meerkat  64×13.5 m, South Africa, L/UHF/S — SKA precursor; HI, transients
#     wsrt     Westerbork, Netherlands — E–W 14×25 m
#     atca_*   ATCA, Australia, 6×22 m — many E–W / hybrid (atca_6a, atca_750a, atca_ew352 …)
#     sma.*    Submillimeter Array, Mauna Kea (submm)
#     carma.*  CARMA (decommissioned) mm ;   pdbi-a…d  Plateau de Bure / NOEMA mm
#  Supported coord systems: XYZ (VLA/VLBA/ngVLA/WSRT/ATCA) + LOC (ALMA/ACA/MeerKAT/SMA).
#  UTM configs (a few ALMA / CARMA) are NOT supported.
array_w = "vla.a"

def main():
    print("Hello from toms-gridding-simulation!")
    print(len(list_arrays()), 'configurations available')
    # array_w = 
    


if __name__ == "__main__":
    main()
