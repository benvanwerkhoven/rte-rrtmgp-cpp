/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/earth-system-radiation/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/earth-system-radiation/rte-rrtmgp-cpp
 *
 * Contact: Chiel van Heerwaarden
 * email: chiel.vanheerwaarden@wur.nl
 *
 * Copyright 2020, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#ifndef RRTMGP_KERNELS_H
#define RRTMGP_KERNELS_H


// Kernels of fluxes.
namespace rrtmgp_kernels
{
    extern "C" void sum_broadband(
            int* ncol, int* nlev, int* ngpt,
            Real* spectral_flux, Real* broadband_flux);

    extern "C" void net_broadband_precalc(
            int* ncol, int* nlev,
            Real* broadband_flux_dn, Real* broadband_flux_up,
            Real* broadband_flux_net);

    extern "C" void sum_byband(
            int* ncol, int* nlev, int* ngpt, int* nbnd,
            int* band_lims,
            Real* spectral_flux,
            Real* byband_flux);

    extern "C" void net_byband_precalc(
            int* ncol, int* nlev, int* nbnd,
            Real* byband_flux_dn, Real* byband_flux_up,
            Real* byband_flux_net);

    extern "C" void zero_array_3D(
            int* ni, int* nj, int* nk, Real* array);

    extern "C" void zero_array_4D(
             int* ni, int* nj, int* nk, int* nl, Real* array);

    extern "C" void interpolation(
                int* ncol, int* nlay,
                int* ngas, int* nflav, int* neta, int* npres, int* ntemp,
                int* flavor,
                Real* press_ref_log,
                Real* temp_ref,
                Real* press_ref_log_delta,
                Real* temp_ref_min,
                Real* temp_ref_delta,
                Real* press_ref_trop_log,
                Real* vmr_ref,
                Real* play,
                Real* tlay,
                Real* col_gas,
                int* jtemp,
                Real* fmajor, Real* fminor,
                Real* col_mix,
                Bool* tropo,
                int* jeta,
                int* jpress);

    extern "C" void compute_tau_absorption(
            int* ncol, int* nlay, int* nband, int* ngpt,
            int* ngas, int* nflav, int* neta, int* npres, int* ntemp,
            int* nminorlower, int* nminorklower,
            int* nminorupper, int* nminorkupper,
            int* idx_h2o,
            int* gpoint_flavor,
            int* band_lims_gpt,
            Real* kmajor,
            Real* kminor_lower,
            Real* kminor_upper,
            int* minor_limits_gpt_lower,
            int* minor_limits_gpt_upper,
            Bool* minor_scales_with_density_lower,
            Bool* minor_scales_with_density_upper,
            Bool* scale_by_complement_lower,
            Bool* scale_by_complement_upper,
            int* idx_minor_lower,
            int* idx_minor_upper,
            int* idx_minor_scaling_lower,
            int* idx_minor_scaling_upper,
            int* kminor_start_lower,
            int* kminor_start_upper,
            Bool* tropo,
            Real* col_mix, Real* fmajor, Real* fminor,
            Real* play, Real* tlay, Real* col_gas,
            int* jeta, int* jtemp, int* jpress,
            Real* tau);

    extern "C" void reorder_123x321_kernel(
            int* dim1, int* dim2, int* dim3,
            Real* array, Real* array_out);

    extern "C" void combine_and_reorder_2str(
            int* ncol, int* nlay, int* ngpt,
            Real* tau_local, Real* tau_rayleigh,
            Real* tau, Real* ssa, Real* g);

    extern "C" void compute_Planck_source(
            int* ncol, int* nlay, int* nbnd, int* ngpt,
            int* nflav, int* neta, int* npres, int* ntemp, int* nPlanckTemp,
            Real* tlay, Real* tlev, Real* tsfc, int* sfc_lay,
            Real* fmajor, int* jeta, Bool* tropo, int* jtemp, int* jpress,
            int* gpoint_bands, int* band_lims_gpt, Real* pfracin, Real* temp_ref_min,
            Real* totplnk_delta, Real* totplnk, int* gpoint_flavor,
            Real* sfc_src, Real* lay_src, Real* lev_src, Real* lev_source_dec,
            Real* sfc_src_jac);

    extern "C" void compute_tau_rayleigh(
            int* ncol, int* nlay, int* nband, int* ngpt,
            int* ngas, int* nflav, int* neta, int* npres, int* ntemp,
            int* gpoint_flavor,
            int* band_lims_gpt,
            Real* krayl,
            int* idx_h2o, Real* col_dry, Real* col_gas,
            Real* fminor, int* eta,
            Bool* tropo, int* jtemp,
            Real* tau_rayleigh);

    extern "C" void apply_BC_0(
            int* ncol, int* nlay, int* ngpt,
            Bool* top_at_1, Real* gpt_flux_dn);

    extern "C" void apply_BC_gpt(
            int* ncol, int* nlay, int* ngpt,
            Bool* top_at_1, Real* inc_flux, Real* gpt_flux_dn);

    extern "C" void lw_solver_noscat_GaussQuad(
            int* ncol, int* nlay, int* ngpt, Bool* top_at_1, int* n_quad_angs,
            Real* gauss_Ds_subset, Real* gauss_wts_subset,
            Real* tau,
            Real* lay_source, Real* lev_source_inc, Real* lev_source_dec,
            Real* sfc_emis_gpt, Real* sfc_source,
            Real* gpt_flux_up, Real* gpt_flux_dn,
            Real* sfc_source_jac, Real* gpt_flux_up_jac);

    extern "C" void apply_BC_factor(
            int* ncol, int* nlay, int* ngpt,
            Bool* top_at_1, Real* inc_flux,
            Real* factor, Real* flux_dn);

    extern "C" void sw_solver_2stream(
            int* ncol, int* nlay, int* ngpt, Bool* top_at_1,
            Real* tau,
            Real* ssa,
            Real* g,
            Real* mu0,
            Real* sfc_alb_dir_gpt, Real* sfc_alb_dif_gpt,
            Real* gpt_flux_up, Real* gpt_flux_dn, Real* gpt_flux_dir);

    extern "C" void increment_2stream_by_2stream(
            int* ncol, int* nlev, int* ngpt,
            Real* tau_inout, Real* ssa_inout, Real* g_inout,
            Real* tau_in, Real* ssa_in, Real* g_in);

    extern "C" void increment_1scalar_by_1scalar(
            int* ncol, int* nlev, int* ngpt,
            Real* tau_inout, Real* tau_in);

    extern "C" void inc_2stream_by_2stream_bybnd(
            int* ncol, int* nlev, int* ngpt,
            Real* tau_inout, Real* ssa_inout, Real* g_inout,
            Real* tau_in, Real* ssa_in, Real* g_in,
            int* nbnd, int* band_lims_gpoint);

    extern "C" void inc_1scalar_by_1scalar_bybnd(
            int* ncol, int* nlev, int* ngpt,
            Real* tau_inout, Real* tau_in,
            int* nbnd, int* band_lims_gpoint);

    extern "C" void delta_scale_2str_k(
            int* ncol, int* nlev, int* ngpt,
            Real* tau_inout, Real* ssa_inout, Real* g_inout);
}
#endif
