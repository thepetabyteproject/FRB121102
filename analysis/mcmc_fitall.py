from burstfit.fit import BurstFit
from burstfit.data import BurstData
from burstfit.model import Model, SgramModel
from burstfit.utils.plotter import *
from burstfit.utils.functions import *
from burstfit.io import BurstIO
from burstfit.utils.astro import radiometer
import os
from copy import deepcopy

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import numpy as np
import logging

logging_format = "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
)

import pandas as pd
import glob
import h5py


def fit(h5s):
    errors = []
    save = True
    plot = False
    for h5 in h5s:
        logging.info(f"Fitting for burst {h5}")
        with h5py.File(h5, "r") as f:
            bd = BurstData(  # fclean_21
                fp=f.attrs["filename"],
                dm=f.attrs["dm"],
                width=f.attrs["width"],
                snr=f.attrs["snr"],
                tcand=f.attrs["tcand"],
            )
            rfi_mask = f.attrs["rfi_mask"]

        mask = np.where(rfi_mask == True)[0].tolist()
        max_ncomp = 3
        pf = pulse_fn_vec
        profile_bounds = []
        spectra_bounds = []
        pnames = ["S", "mu_t", "sigma_t", "tau"]

        # Fit coniditions (RFI masks, bounds, etc) for some bursts
        if "tstart_57644.407719907409_tcand_1063.5400000" in h5:
            mask += [51, 52]
            profile_bounds = ([0, 1000, 0, 0], [1000, 1500, 50, 50])
        elif "tstart_57644.407719907409_tcand_3817.8100000" in h5:
            mask += [51, 52]
        elif "tstart_57644.407719907409_tcand_3750.0500000" in h5:
            max_ncomp = 1
        elif "tstart_57645.409861111111_tcand_2735.4700000" in h5:
            max_ncomp = 1
        elif "57644.407719907409_tcand_248.2970000_dm_562.05600_snr_97.12390" in h5:
            mask = [0, 41, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
        elif "57645.409861111111_tcand_2979.0300000" in h5:
            mask += [19]
        elif "tstart_57645.409861111111_tcand_3713.9400000" in h5:
            mask += [10, 19]
        elif "tstart_57645.409861111111_tcand_4833.0500000" in h5:
            max_ncomp = 1
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 30, 30])
        elif (
            "tstart_57645.409861111111_tcand_56.3636000_dm_562.05600_snr_43.19220" in h5
        ):
            max_ncomp = 1
        elif (
            "tstart_57645.409861111111_tcand_4286.4700000_dm_562.05600_snr_168.68900"
            in h5
        ):
            mask += [51, 52]
            max_ncomp = 3
            profile_bounds = ([0, 1000, 0, 0], [20000, 1500, 100, 100])
        elif "57644.407719907409_tcand_1762.0400000_dm_565.30000_snr_8.49012" in h5:
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 50, 100])
        elif (
            "tstart_57644.407719907409_tcand_3198.8400000_dm_565.30000_snr_6.94612"
            in h5
        ):
            mask += [51, 52]
        elif (
            "tstart_57644.407719907409_tcand_5951.0100000_dm_565.30000_snr_9.23291"
            in h5
        ):
            mask += [(41, 53)]
        elif "57645.409861111111_tcand_1460.2500000_dm_555.62500_snr_12.63900" in h5:
            mask += [51]
        elif (
            "tstart_57645.409861111111_tcand_529.4350000_dm_555.62500_snr_10.62310"
            in h5
        ):
            mask += [51]
            max_ncomp = 1
        elif (
            "cand_tstart_57644.407719907409_tcand_1564.2000000_dm_568.56200_snr_7.26308"
            in h5
        ):
            mask += [51]
        elif (
            "cand_tstart_57644.407719907409_tcand_3333.8700000_dm_558.83100_snr_8.33006"
            in h5
        ):
            mask += [(41, 64)]
        elif (
            "cand_tstart_57644.407719907409_tcand_1115.1100000_dm_568.56200_snr_7.00356.h5"
            in h5
        ):
            mask += [51]
        elif (
            "cand_tstart_57644.407719907409_tcand_3873.8400000_dm_634.24800_snr_6.59650.h5"
            in h5
        ):
            mask += [51]
        elif (
            "cand_tstart_57644.407719907409_tcand_127.5000000_dm_568.56200_snr_8.73822.h5"
            in h5
        ):
            profile_bounds = ([0, 1100, 0, 0], [1400, 1500, 20, 20])
        elif (
            "cand_tstart_57644.407719907409_tcand_2232.9900000_dm_552.43700_snr_6.90756"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 50, 50])
            bd.dm = 560
        elif (
            "cand_tstart_57644.407719907409_tcand_2298.7400000_dm_565.30000_snr_6.38018.h5"
            in h5
        ):
            profile_bounds = ([0, 1200, 0, 0], [2000, 1500, 20, 20])
        elif (
            "cand_tstart_57644.407719907409_tcand_2643.3300000_dm_568.56200_snr_16.48840.h5"
            in h5
        ):
            mask += [5, 6]
        elif (
            "cand_tstart_57644.407719907409_tcand_2647.6700000_dm_562.05600_snr_9.49854.h5"
            in h5
        ):
            mask += [5, 6, 51]
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 50, 50])
        elif (
            "cand_tstart_57644.407719907409_tcand_349.1770000_dm_486.74800_snr_6.24940.h5"
            in h5
        ):
            bd.dm = 550
            bd.tcand += 50e-3
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 50, 50])
        elif (
            "cand_tstart_57644.407719907409_tcand_349.2190000_dm_605.71300_snr_8.36633.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 30, 30])
            mask += [51]
            bd.dm = 569
        elif (
            "cand_tstart_57644.407719907409_tcand_432.4670000_dm_527.58200_snr_8.07396.h5"
            in h5
        ):
            bd.dm = 550
        elif (
            "cand_tstart_57644.407719907409_tcand_4864.6800000_dm_568.56200_snr_6.72464.h5"
            in h5
        ):
            mask += [1, 2]
        elif (
            "cand_tstart_57644.407719907409_tcand_542.0000000_dm_562.05600_snr_6.35508.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 30, 30])
            mask += [11]
        elif (
            "cand_tstart_57644.407719907409_tcand_873.5360000_dm_571.84300_snr_9.83474.h5"
            in h5
        ):
            bd.dm = 563
        elif (
            "cand_tstart_57645.409861111111_tcand_1305.3300000_dm_562.05600_snr_6.81263.h5"
            in h5
        ):
            mask += [4]
        elif (
            "cand_tstart_57645.409861111111_tcand_1743.9100000_dm_558.83100_snr_13.70630.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 20, 20])
            max_ncomp = 1
            mask += [4]
        elif (
            "cand_tstart_57645.409861111111_tcand_2064.2400000_dm_549.26700_snr_8.31295.h5"
            in h5
        ):
            bd.dm = 554
            mask += [51]
        elif (
            "cand_tstart_57645.409861111111_tcand_210.1250000_dm_581.80000_snr_6.53356.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 50, 50])
            mask += [51]
        elif (
            "cand_tstart_57645.409861111111_tcand_277.1580000_dm_578.46200_snr_10.72190.h5"
            in h5
        ):
            bd.dm = 566
        elif (
            "cand_tstart_57645.409861111111_tcand_3024.2700000_dm_552.43700_snr_8.02753.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 50, 50])
        elif (
            "cand_tstart_57645.409861111111_tcand_4502.7000000_dm_568.56200_snr_7.30701.h5"
            in h5
        ):
            mask += [22]
            max_ncomp = 1
        elif (
            "cand_tstart_57645.409861111111_tcand_5349.8500000_dm_558.83100_snr_6.12860.h5"
            in h5
        ):
            mask += [1, 2]
        elif (
            "cand_tstart_57645.409861111111_tcand_849.1810000_dm_562.05600_snr_16.06050.h5"
            in h5
        ):
            profile_bounds = ([0, 800, 0, 0], [2000, 1500, 20, 5])
        elif (
            "cand_tstart_57644.407719907409_tcand_127.5000000_dm_568.56200_snr_8.73822.h5"
            in h5
        ):
            profile_bounds = ([0, 1100, 0, 0], [1400, 1500, 20, 20])
        elif (
            "cand_tstart_57645.409861111111_tcand_878.0750000_dm_568.56200_snr_6.64522.h5"
            in h5
        ):
            bd.dm = 557
            mask += [34, 35, 42]
        elif (
            "cand_tstart_57645.409861111111_tcand_985.8850000_dm_568.56200_snr_6.48642.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 20, 20])
        elif (
            "cand_tstart_57644.407719907409_tcand_1642.5000000_dm_558.83100_snr_6.08731.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 200, 200])
        elif (
            "cand_tstart_57644.407719907409_tcand_1705.5600000_dm_486.74800_snr_6.68708.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 2000, 200, 200])
            bd.dm = 585
        elif (
            "cand_tstart_57644.407719907409_tcand_5174.6400000_dm_568.56200_snr_11.69680.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 2000, 200, 200])
            mask += [24]
        elif (
            "cand_tstart_57645.409861111111_tcand_1409.0500000_dm_568.56200_snr_6.27196.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 1300, 15, 15])
        elif (
            "cand_tstart_57645.409861111111_tcand_1938.5100000_dm_555.62500_snr_6.28058.h5"
            in h5
        ):
            mask += [51]
        elif (
            "cand_tstart_57644.407719907409_tcand_873.3120000_dm_565.00000_snr_6.00000.h5"
            in h5
        ):
            max_ncomp = 1
        elif (
            "cand_tstart_57644.407719907409_tcand_3415.0300000_dm_558.83100_snr_8.64735.h5"
            in h5
        ):
            profile_bounds = ([0, 1000, 0, 0], [2000, 1500, 50, 50])
            mask += [51]
        elif (
            "cand_tstart_57645.409861111111_tcand_4670.9600000_dm_581.80000_snr_6.24578.h5"
            in h5
        ):
            bd.dm = 569
        elif (
            "tstart_57645.409861111111_tcand_1086.2000000_dm_562.05600_snr_6.40849"
            in h5
        ):
            mask += [51]
        else:
            profile_bounds = []
            spectra_bounds = []

        if "57644.407719907409_tcand_1898.2800000_dm_562.05600_snr_7.12105" in h5:
            bd.prepare_data(mask_chans=mask, time_window=0.1)
        else:
            bd.prepare_data(mask_chans=mask, time_window=0.2)

        pulseModel = Model(pf, param_names=pnames)
        spectraModel = Model(gauss_norm, param_names=["mu_f", "sigma_f"])
        sgramModel = SgramModel(
            pulseModel, spectraModel, sgram_fn_vec, mask=bd.mask, clip_fac=bd.clip_fac
        )
        bf = BurstFit(
            sgram_model=sgramModel,
            sgram=bd.sgram,
            width=bd.width,
            dm=bd.dm,
            foff=bd.foff,
            fch1=bd.fch1,
            tsamp=bd.tsamp,
            clip_fac=bd.clip_fac,
            mask=bd.mask,
            mcmcfit=False,
        )

        bf.fitall(
            plot=plot,
            max_ncomp=max_ncomp,
            profile_bounds=profile_bounds,
            spectra_bounds=spectra_bounds,
        )

        # Some bursts had multiple components that weren't automatically detected.
        # Fitting those components.
        if (
            "cand_tstart_57644.407719907409_tcand_127.5000000_dm_568.56200_snr_8.73822.h5"
            in h5
        ):
            bf.comp_num += 1
            bf.fitcycle(plot=plot)
            bf.fit_all_components(plot=plot)
            test_res = bf.run_tests
        elif (
            "cand_tstart_57644.407719907409_tcand_349.1770000_dm_486.74800_snr_6.24940.h5"
            in h5
        ):
            bf.comp_num += 1
            bf.fitcycle(
                plot=plot, profile_bounds=([0, 1200, 0, 0], [2000, 1700, 30, 30])
            )
            bf.fit_all_components(plot=plot)
            test_res = bf.run_tests
        elif (
            "cand_tstart_57644.407719907409_tcand_349.2190000_dm_605.71300_snr_8.36633.h5"
            in h5
        ):
            bf.comp_num += 1
            bf.fitcycle(
                plot=plot, profile_bounds=([0, 1200, 0, 0], [2000, 1700, 30, 30])
            )
            bf.fit_all_components(plot=plot)
            test_res = bf.run_tests
        elif (
            "cand_tstart_57644.407719907409_tcand_4864.6800000_dm_568.56200_snr_6.72464.h5"
            in h5
        ):
            bf.comp_num += 1
            bf.fitcycle(plot=plot)
            bf.fit_all_components(plot=plot)
            test_res = bf.run_tests
        elif (
            "cand_tstart_57644.407719907409_tcand_4886.4000000_dm_565.30000_snr_15.69560.h5"
            in h5
        ):
            bf.comp_num += 1
            bf.fitcycle(
                plot=plot, profile_bounds=([0, 2000, 0, 0], [2000, 2500, 30, 30])
            )
            bf.fit_all_components(plot=plot)
            test_res = bf.run_tests
        elif (
            "cand_tstart_57644.407719907409_tcand_873.5360000_dm_571.84300_snr_9.83474.h5"
            in h5
        ):
            bf.comp_num += 1
            bf.fitcycle(
                plot=plot, profile_bounds=([0, 1000, 0, 0], [2000, 1500, 30, 30])
            )
            bf.fit_all_components(plot=plot)
            test_res = bf.run_tests
        elif (
            "cand_tstart_57645.409861111111_tcand_1743.9100000_dm_558.83100_snr_13.70630.h5"
            in h5
        ):
            bf.comp_num += 1
            bf.fitcycle(
                plot=plot, profile_bounds=([0, 1000, 0, 0], [2000, 1300, 20, 20])
            )
            bf.fit_all_components(plot=plot)
            test_res = bf.run_tests
        elif (
            "cand_tstart_57645.409861111111_tcand_1409.0500000_dm_568.56200_snr_6.27196.h5"
            in h5
        ):
            bf.comp_num += 1
            bf.fitcycle(
                plot=plot, profile_bounds=([0, 1000, 0, 0], [2000, 1500, 15, 15])
            )
            bf.fit_all_components(plot=plot)
            test_res = bf.run_tests
        elif (
            "cand_tstart_57645.409861111111_tcand_159.4330000_dm_558.83100_snr_7.95436.h5"
            in h5
        ):
            bf.comp_num += 1
            bf.fitcycle(
                plot=plot, profile_bounds=([0, 800, 0, 0], [2000, 1200, 30, 30])
            )
            bf.fit_all_components(plot=plot)
            test_res = bf.run_tests

        # Setting MCMC parameters
        mcmc_kwargs = {}
        mcmc_kwargs = {
            "nwalkers": 20,
            "nsteps": 5 * 50000 * bf.ncomponents,
            "skip": 10000,
            "ncores": 20,
            "start_pos_dev": 0.01,
            "prior_range": 0.8,
            "save_results": True,
            "outname": bd.id,
        }

        # Running MCMC fit using curve_fit parameters
        bf.run_mcmc(plot=True, **mcmc_kwargs)
        popt = []
        for i in range(1, bf.ncomponents + 1):
            popt += bf.mcmc_params[i]["popt"]

        # Plotting fit results
        _ = plot_2d_fit(
            bf.sgram,
            bf.model_from_params,
            popt,
            bf.tsamp,
            save=True,
            show=False,
            outname=bd.id + "_fit_res",
            title=f"{bd.id} \n rch: {bf.reduced_chi_sq:.3f} \n ncomp: {bf.ncomponents}",
            outdir="121102_paper/mcmc_final/",
        )
        if save:
            bio = BurstIO(bf, bd, outdir="121102_paper/mcmc_final/")
            d = bio.save_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run burstfit with MCMC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--files", help="h5 file to use for fitting", required=True, nargs="+"
    )
    values = parser.parse_args()
    print(values.files)
    fit(values.files)
