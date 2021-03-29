#include <chrono>

#include "Array.h"
#include "rrtmgp_kernel_launcher_cuda.h"
#include "tools_gpu.h"


namespace
{
    __device__
    Real interpolate1D(
            const Real val,
            const Real offset,
            const Real delta,
            const int len,
            const Real* __restrict__ table)
    {
        Real val0 = (val - offset)/delta;
        Real frac = val0 - int(val0);
        int idx = min(len-1, max(1, int(val0)+1));
        return table[idx-1] + frac * (table[idx] - table[idx-1]);
    }

    __device__
    void interpolate2D_byflav_kernel(const Real* __restrict__ fminor,
                                     const Real* __restrict__ kin,
                                     const int gpt_start, const int gpt_end,
                                     Real* __restrict__ k,
                                     const int* __restrict__ jeta,
                                     const int jtemp,
                                     const int ngpt,
                                     const int neta)
    {
        const int band_gpt = gpt_end-gpt_start;
        const int j0 = jeta[0];
        const int j1 = jeta[1];
        for (int igpt=0; igpt<band_gpt; ++igpt)
        {
            k[igpt] = fminor[0] * kin[igpt + (j0-1)*ngpt + (jtemp-1)*neta*ngpt] +
                      fminor[1] * kin[igpt +  j0   *ngpt + (jtemp-1)*neta*ngpt] +
                      fminor[2] * kin[igpt + (j1-1)*ngpt + jtemp    *neta*ngpt] +
                      fminor[3] * kin[igpt +  j1   *ngpt + jtemp    *neta*ngpt];
        }
    }

    __device__
    void interpolate3D_byflav_kernel(
            const Real* __restrict__ scaling,
            const Real* __restrict__ fmajor,
            const Real* __restrict__ k,
            const int gpt_start, const int gpt_end,
            const int* __restrict__ jeta,
            const int jtemp,
            const int jpress,
            const int ngpt,
            const int neta,
            const int npress,
            Real* __restrict__ tau_major)
    {
        const int band_gpt = gpt_end-gpt_start;
        const int j0 = jeta[0];
        const int j1 = jeta[1];
        for (int igpt=0; igpt<band_gpt; ++igpt)
        {
            tau_major[igpt] = scaling[0]*
                              (fmajor[0] * k[igpt + (j0-1)*ngpt + (jpress-1)*neta*ngpt + (jtemp-1)*neta*ngpt*npress] +
                               fmajor[1] * k[igpt +  j0   *ngpt + (jpress-1)*neta*ngpt + (jtemp-1)*neta*ngpt*npress] +
                               fmajor[2] * k[igpt + (j0-1)*ngpt + jpress*neta*ngpt     + (jtemp-1)*neta*ngpt*npress] +
                               fmajor[3] * k[igpt +  j0   *ngpt + jpress*neta*ngpt     + (jtemp-1)*neta*ngpt*npress])
                            + scaling[1]*
                              (fmajor[4] * k[igpt + (j1-1)*ngpt + (jpress-1)*neta*ngpt + jtemp*neta*ngpt*npress] +
                               fmajor[5] * k[igpt +  j1   *ngpt + (jpress-1)*neta*ngpt + jtemp*neta*ngpt*npress] +
                               fmajor[6] * k[igpt + (j1-1)*ngpt + jpress*neta*ngpt     + jtemp*neta*ngpt*npress] +
                               fmajor[7] * k[igpt +  j1   *ngpt + jpress*neta*ngpt     + jtemp*neta*ngpt*npress]);
        }
    }



    __global__
    void reorder12x21_kernel(
            const int ni, const int nj,
            const Real* __restrict__ arr_in, Real* __restrict__ arr_out)
    {
        const int ii = blockIdx.x*blockDim.x + threadIdx.x;
        const int ij = blockIdx.y*blockDim.y + threadIdx.y;
        if ( (ii < ni) && (ij < nj) )
        {
            const int idx_out = ii + ij*ni;
            const int idx_in = ij + ii*nj;
            arr_out[idx_out] = arr_in[idx_in];
        }
    }

    __global__
    void reorder123x321_kernel(
            const int ni, const int nj, const int nk,
            const Real* __restrict__ arr_in, Real* __restrict__ arr_out)
    {
        const int ii = blockIdx.x*blockDim.x + threadIdx.x;
        const int ij = blockIdx.y*blockDim.y + threadIdx.y;
        const int ik = blockIdx.z*blockDim.z + threadIdx.z;
        if ( (ii < ni) && (ij < nj) && (ik < nk))
        {
            const int idx_out = ii + ij*ni + ik*nj*ni;
            const int idx_in = ik + ij*nk + ii*nj*nk;
            arr_out[idx_out] = arr_in[idx_in];
        }
    }

    __global__
    void zero_array_kernel(
            const int ni, const int nj, const int nk,
            Real* __restrict__ arr)
    {
        const int ii = blockIdx.x*blockDim.x + threadIdx.x;
        const int ij = blockIdx.y*blockDim.y + threadIdx.y;
        const int ik = blockIdx.z*blockDim.z + threadIdx.z;
        if ( (ii < ni) && (ij < nj) && (ik < nk))
        {
            const int idx = ii + ij*ni + ik*nj*ni;
            arr[idx] = Real(0.);
        }
    }

    __global__
    void Planck_source_kernel(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int nPlanckTemp,
            const Real* __restrict__ tlay, const Real* __restrict__ tlev,
            const Real* __restrict__ tsfc,
            const int sfc_lay,
            const Real* __restrict__ fmajor, const int* __restrict__ jeta,
            const Bool* __restrict__ tropo, const int* __restrict__ jtemp,
            const int* __restrict__ jpress, const int* __restrict__ gpoint_bands,
            const int* __restrict__ band_lims_gpt, const Real* __restrict__ pfracin,
            const Real temp_ref_min, const Real totplnk_delta,
            const Real* __restrict__ totplnk, const int* __restrict__ gpoint_flavor,
            const Real* __restrict__ ones, const Real delta_Tsurf,
            Real* __restrict__ sfc_src, Real* __restrict__ lay_src,
            Real* __restrict__ lev_src_inc, Real* __restrict__ lev_src_dec,
            Real* __restrict__ sfc_src_jac, Real* __restrict__ pfrac)
    {
        const int ibnd = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int icol = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (ibnd < nband))
        {
            const int idx_collay = icol + ilay * ncol;
            const int itropo = !tropo[idx_collay];
            const int gpt_start = band_lims_gpt[2 * ibnd] - 1;
            const int gpt_end = band_lims_gpt[2 * ibnd + 1];
            const int iflav = gpoint_flavor[itropo + 2 * gpt_start] - 1;
            const int idx_fcl3 = 2 * 2 * 2* (iflav + icol * nflav + ilay * ncol * nflav);
            const int idx_fcl1 = 2 * (iflav + icol * nflav + ilay * ncol * nflav);
            const int idx_tau = gpt_start + ilay * ngpt + icol * nlay * ngpt;

            //major gases//
            interpolate3D_byflav_kernel(ones, &fmajor[idx_fcl3],
                                        &pfracin[gpt_start], gpt_start, gpt_end,
                                        &jeta[idx_fcl1], jtemp[idx_collay],
                                        jpress[idx_collay]+itropo, ngpt, neta, npres+1,
                                        &pfrac[idx_tau]);

            // compute surface source irradiances
            if (ilay == sfc_lay - 1) // Subtract one to correct for fortran indexing.
            {
                const Real planck_function_sfc1 = interpolate1D(tsfc[icol],               temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk[ibnd * nPlanckTemp]);
                const Real planck_function_sfc2 = interpolate1D(tsfc[icol] + delta_Tsurf, temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk[ibnd * nPlanckTemp]);

                for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
                {
                    const int idx_in  = igpt + ilay*ngpt + icol*nlay*ngpt;
                    const int idx_out = igpt + icol*ngpt;
                    sfc_src[idx_out] = pfrac[idx_in] * planck_function_sfc1;
                    sfc_src_jac[idx_out] = pfrac[idx_in] * (planck_function_sfc2 - planck_function_sfc1);
                }
            }

            // compute layer source irradiances.
            const int idx_tmp = icol + ilay*ncol;
            const Real planck_function_lay = interpolate1D(tlay[idx_tmp], temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk[ibnd * nPlanckTemp]);
            for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
            {
                const int idx_inout  = igpt + ilay*ngpt + icol*nlay*ngpt;
                lay_src[idx_inout] = pfrac[idx_inout] * planck_function_lay;
            }

            // compute level source irradiances.
            const int idx_tmp1 = icol + (ilay+1)*ncol;
            const int idx_tmp2 = icol + ilay*ncol;
            const Real planck_function_lev1 = interpolate1D(tlev[idx_tmp1], temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk[ibnd * nPlanckTemp]);
            const Real planck_function_lev2 = interpolate1D(tlev[idx_tmp2], temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk[ibnd * nPlanckTemp]);
            for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
            {
                const int idx_inout  = igpt + ilay*ngpt + icol*nlay*ngpt;
                lev_src_inc[idx_inout] = pfrac[idx_inout] * planck_function_lev1;
                lev_src_dec[idx_inout] = pfrac[idx_inout] * planck_function_lev2;
            }
        }
    }

    __global__
    void interpolation_kernel(
            const int ncol, const int nlay, const int ngas, const int nflav,
            const int neta, const int npres, const int ntemp, const Real tmin,
            const int* __restrict__ flavor,
            const Real* __restrict__ press_ref_log,
            const Real* __restrict__ temp_ref,
            Real press_ref_log_delta,
            Real temp_ref_min,
            Real temp_ref_delta,
            Real press_ref_trop_log,
            const Real* __restrict__ vmr_ref,
            const Real* __restrict__ play,
            const Real* __restrict__ tlay,
            Real* __restrict__ col_gas,
            int* __restrict__ jtemp,
            Real* __restrict__ fmajor, Real* __restrict__ fminor,
            Real* __restrict__ col_mix,
            Bool* __restrict__ tropo,
            int* __restrict__ jeta,
            int* __restrict__ jpress)
    {
        const int ilay = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (ilay < nlay) )
        {
            const int idx = icol + ilay*ncol;

            jtemp[idx] = int((tlay[idx] - (temp_ref_min-temp_ref_delta)) / temp_ref_delta);
            jtemp[idx] = min(ntemp-1, max(1, jtemp[idx]));
            const Real ftemp = (tlay[idx] - temp_ref[jtemp[idx]-1]) / temp_ref_delta;

            const Real locpress = Real(1.) + (log(play[idx]) - press_ref_log[0]) / press_ref_log_delta;
            jpress[idx] = min(npres-1, max(1, int(locpress)));
            const Real fpress = locpress - Real(jpress[idx]);

            tropo[idx] = log(play[idx]) > press_ref_trop_log;
            const int itropo = !tropo[idx];

            for (int iflav=0; iflav<nflav; ++iflav)
            {
                const int gas1 = flavor[2*iflav];
                const int gas2 = flavor[2*iflav+1];
                for (int itemp=0; itemp<2; ++itemp)
                {
                    const int vmr_base_idx = itropo + (jtemp[idx]+itemp-1) * (ngas+1) * 2;
                    const int colmix_idx = itemp + 2*(iflav + nflav*icol + nflav*ncol*ilay);
                    const int colgas1_idx = icol + ilay*ncol + gas1*nlay*ncol;
                    const int colgas2_idx = icol + ilay*ncol + gas2*nlay*ncol;
                    Real eta;
                    const Real ratio_eta_half = vmr_ref[vmr_base_idx + 2 * gas1] /
                                              vmr_ref[vmr_base_idx + 2 * gas2];
                    col_mix[colmix_idx] = col_gas[colgas1_idx] + ratio_eta_half * col_gas[colgas2_idx];
                    if (col_mix[colmix_idx] > Real(2.)*tmin)
                    {
                        eta = col_gas[colgas1_idx] / col_mix[colmix_idx];
                    } else
                    {
                        eta = Real(0.5);
                    }
                    const Real loceta = eta * Real(neta-1);
                    jeta[colmix_idx] = min(int(loceta)+1, neta-1);
                    const Real feta = fmod(loceta, Real(1.));
                    const Real ftemp_term  = Real(1-itemp) + Real(2*itemp-1)*ftemp;
                    // compute interpolation fractions needed for minot species
                    const int fminor_idx = 2*(itemp + 2*(iflav + icol*nflav + ilay*ncol*nflav));
                    fminor[fminor_idx] = (Real(1.0)-feta) * ftemp_term;
                    fminor[fminor_idx+1] = feta * ftemp_term;
                    // compute interpolation fractions needed for major species
                    const int fmajor_idx = 2*2*(itemp + 2*(iflav + icol*nflav + ilay*ncol*nflav));
                    fmajor[fmajor_idx] = (Real(1.0)-fpress) * fminor[fminor_idx];
                    fmajor[fmajor_idx+1] = (Real(1.0)-fpress) * fminor[fminor_idx+1];
                    fmajor[fmajor_idx+2] = fpress * fminor[fminor_idx];
                    fmajor[fmajor_idx+3] = fpress * fminor[fminor_idx+1];

                }
            }
        }
    }

    __global__
    void compute_tau_major_absorption_kernel(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int* __restrict__ gpoint_flavor,
            const int* __restrict__ band_lims_gpt,
            const Real* __restrict__ kmajor,
            const Real* __restrict__ col_mix, const Real* __restrict__ fmajor,
            const int* __restrict__ jeta, const Bool* __restrict__ tropo,
            const int* __restrict__ jtemp, const int* __restrict__ jpress,
            Real* __restrict__ tau, Real* __restrict__ tau_major)
    {
        // Fetch the three coordinates.
        const int ibnd = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int icol = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (ibnd < nband) ) {
            const int idx_collay = icol + ilay * ncol;
            const int itropo = !tropo[idx_collay];
            const int gpt_start = band_lims_gpt[2 * ibnd] - 1;
            const int gpt_end = band_lims_gpt[2 * ibnd + 1];
            const int iflav = gpoint_flavor[itropo + 2 * gpt_start] - 1;
            const int idx_fcl3 = 2 * 2 * 2* (iflav + icol * nflav + ilay * ncol * nflav);
            const int idx_fcl1 = 2 * (iflav + icol * nflav + ilay * ncol * nflav);
            const int idx_tau = gpt_start + ilay * ngpt + icol * nlay * ngpt;

            //major gases//
            interpolate3D_byflav_kernel(&col_mix[idx_fcl1], &fmajor[idx_fcl3],
                                        &kmajor[gpt_start], gpt_start, gpt_end,
                                        &jeta[idx_fcl1], jtemp[idx_collay],
                                        jpress[idx_collay]+itropo, ngpt, neta, npres+1,
                                        &tau_major[idx_tau]);

            for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
            {
                const int idx_out = igpt + ilay*ngpt + icol*nlay*ngpt;
                tau[idx_out] += tau_major[idx_out];
                //should be '+=' later on, but we first need the zero_arrays for that
            }
        }
    }

    __global__
    void compute_tau_minor_absorption_kernel(
            const int ncol, const int nlay, const int ngpt,
            const int ngas, const int nflav, const int ntemp, const int neta,
            const int nscale_lower,
            const int nscale_upper,
            const int nminor_lower,
            const int nminor_upper,
            const int nminork_lower,
            const int nminork_upper,
            const int idx_h2o,
            const int* __restrict__ gpoint_flavor,
            const Real* __restrict__ kminor_lower,
            const Real* __restrict__ kminor_upper,
            const int* __restrict__ minor_limits_gpt_lower,
            const int* __restrict__ minor_limits_gpt_upper,
            const Bool* __restrict__ minor_scales_with_density_lower,
            const Bool* __restrict__ minor_scales_with_density_upper,
            const Bool* __restrict__ scale_by_complement_lower,
            const Bool* __restrict__ scale_by_complement_upper,
            const int* __restrict__ idx_minor_lower,
            const int* __restrict__ idx_minor_upper,
            const int* __restrict__ idx_minor_scaling_lower,
            const int* __restrict__ idx_minor_scaling_upper,
            const int* __restrict__ kminor_start_lower,
            const int* __restrict__ kminor_start_upper,
            const Real* __restrict__ play,
            const Real* __restrict__ tlay,
            const Real* __restrict__ col_gas,
            const Real* __restrict__ fminor,
            const int* __restrict__ jeta,
            const int* __restrict__ jtemp,
            const Bool* __restrict__ tropo,
            Real* __restrict__ tau,
            Real* __restrict__ tau_minor)
    {
        // Fetch the three coordinates.
        const int ilay = blockIdx.x * blockDim.x + threadIdx.x;
        const int icol = blockIdx.y * blockDim.y + threadIdx.y;
        const Real PaTohPa = 0.01;
        const int ncl = ncol * nlay;
        if ((icol < ncol) && (ilay < nlay))
        {
            //kernel implementation
            const int idx_collay = icol + ilay * ncol;
            const int idx_collaywv = icol + ilay * ncol + idx_h2o * ncl;

            if (tropo[idx_collay] == 1)
            {
                for (int imnr = 0; imnr < nscale_lower; ++imnr)
                {
                    Real scaling = col_gas[idx_collay + idx_minor_lower[imnr] * ncl];
                    if (minor_scales_with_density_lower[imnr])
                    {
                        scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];
                        if (idx_minor_scaling_lower[imnr] > 0)
                        {
                            Real vmr_fact = Real(1.) / col_gas[idx_collay];
                            Real dry_fact = Real(1.) / (Real(1.) + col_gas[idx_collaywv] * vmr_fact);
                            if (scale_by_complement_lower[imnr])
                            {
                                scaling *= (Real(1.) - col_gas[idx_collay + idx_minor_scaling_lower[imnr] * ncl] * vmr_fact * dry_fact);
                            }
                            else
                            {
                                scaling *= col_gas[idx_collay + idx_minor_scaling_lower[imnr] * ncl] * vmr_fact * dry_fact;
                            }
                        }
                    }
                    const int gpt_start = minor_limits_gpt_lower[2*imnr]-1;
                    const int gpt_end = minor_limits_gpt_lower[2*imnr+1];
                    const int iflav = gpoint_flavor[2*gpt_start]-1;
                    const int idx_fcl2 = 2 * 2 * (iflav + icol * nflav + ilay * ncol * nflav);
                    const int idx_fcl1 = 2 * (iflav + icol * nflav + ilay * ncol * nflav);
                    const int idx_tau = gpt_start + ilay*ngpt + icol*nlay*ngpt;

                    interpolate2D_byflav_kernel(&fminor[idx_fcl2], &kminor_lower[kminor_start_lower[imnr]-1],
                                                kminor_start_lower[imnr]-1, kminor_start_lower[imnr]-1 + (gpt_end - gpt_start),
                                                &tau_minor[idx_tau], &jeta[idx_fcl1],
                                                jtemp[idx_collay], nminork_lower, neta);

                    for (int igpt = gpt_start; igpt < gpt_end; ++igpt)
                    {
                        const int idx_out = igpt + ilay * ngpt + icol * nlay * ngpt;
                        tau[idx_out] += tau_minor[idx_out] * scaling;
                    }
                }
            }
            else
            {
                for (int imnr = 0; imnr < nscale_upper; ++imnr)
                {
                    Real scaling = col_gas[idx_collay + idx_minor_upper[imnr] * ncl];
                    if (minor_scales_with_density_upper[imnr])
                    {
                        scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];
                        if (idx_minor_scaling_upper[imnr] > 0)
                        {
                            Real vmr_fact = Real(1.) / col_gas[idx_collay];
                            Real dry_fact = Real(1.) / (Real(1.) + col_gas[idx_collaywv] * vmr_fact);
                            if (scale_by_complement_upper[imnr])
                            {
                                scaling *= (Real(1.) - col_gas[idx_collay + idx_minor_scaling_upper[imnr] * ncl] * vmr_fact * dry_fact);
                            }
                            else
                            {
                                scaling *= col_gas[idx_collay + idx_minor_scaling_upper[imnr] * ncl] * vmr_fact * dry_fact;
                            }
                        }
                    }
                    const int gpt_start = minor_limits_gpt_upper[2*imnr]-1;
                    const int gpt_end = minor_limits_gpt_upper[2*imnr+1];
                    const int iflav = gpoint_flavor[2*gpt_start+1]-1;
                    const int idx_fcl2 = 2 * 2 * (iflav + icol * nflav + ilay * ncol * nflav);
                    const int idx_fcl1 = 2 * (iflav + icol * nflav + ilay * ncol * nflav);
                    const int idx_tau = gpt_start + ilay*ngpt + icol*nlay*ngpt;

                    interpolate2D_byflav_kernel(&fminor[idx_fcl2], &kminor_upper[kminor_start_upper[imnr]-1],
                                                kminor_start_upper[imnr]-1, kminor_start_upper[imnr]-1 + (gpt_end - gpt_start),
                                                &tau_minor[idx_tau], &jeta[idx_fcl1],
                                                jtemp[idx_collay], nminork_upper, neta);

                    for (int igpt = gpt_start; igpt < gpt_end; ++igpt)
                    {
                        const int idx_out = igpt + ilay * ngpt + icol * nlay * ngpt;
                        tau[idx_out] += tau_minor[idx_out] * scaling;
                    }
                }
            }
        }
    }

    __global__
    void compute_tau_rayleigh_kernel(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* __restrict__ gpoint_flavor,
            const int* __restrict__ band_lims_gpt,
            const Real* __restrict__ krayl,
            int idx_h2o, const Real* __restrict__ col_dry, const Real* __restrict__ col_gas,
            const Real* __restrict__ fminor, const int* __restrict__ jeta,
            const Bool* __restrict__ tropo, const int* __restrict__ jtemp,
            Real* __restrict__ tau_rayleigh, Real* __restrict__ k)
    {
        // Fetch the three coordinates.
        const int ibnd = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int icol = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (ibnd < nbnd) )
        {
            //kernel implementation
            const int idx_collay = icol + ilay*ncol;
            const int idx_collaywv = icol + ilay*ncol + idx_h2o*nlay*ncol;
            const int itropo = !tropo[idx_collay];
            const int gpt_start = band_lims_gpt[2*ibnd]-1;
            const int gpt_end = band_lims_gpt[2*ibnd+1];
            const int iflav = gpoint_flavor[itropo+2*gpt_start]-1;
            const int idx_fcl2 = 2*2*(iflav + icol*nflav + ilay*ncol*nflav);
            const int idx_fcl1   = 2*(iflav + icol*nflav + ilay*ncol*nflav);
            const int idx_krayl  = gpt_start + ngpt*neta*ntemp*itropo;
            const int idx_k = gpt_start + ilay*ngpt + icol*nlay*ngpt;
            interpolate2D_byflav_kernel(&fminor[idx_fcl2],
                                        &krayl[idx_krayl],
                                        gpt_start, gpt_end, &k[idx_k],
                                        &jeta[idx_fcl1],
                                        jtemp[idx_collay],
                                        ngpt, neta);

            for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
            {
                const int idx_out = igpt + ilay*ngpt + icol*nlay*ngpt;
                tau_rayleigh[idx_out] = k[idx_k+igpt-gpt_start]*(col_gas[idx_collaywv]+col_dry[idx_collay]);
            }
        }
    }


    __global__
    void combine_and_reorder_2str_kernel(
            const int ncol, const int nlay, const int ngpt, const Real tmin,
            const Real* __restrict__ tau_abs, const Real* __restrict__ tau_rayleigh,
            Real* __restrict__ tau, Real* __restrict__ ssa, Real* __restrict__ g)
    {
        // Fetch the three coordinates.
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
        const int ilay = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
        {
            const int idx_in  = igpt + ilay*ngpt + icol*(ngpt*nlay);
            const int idx_out = icol + ilay*ncol + igpt*(ncol*nlay);

            const Real tau_tot = tau_abs[idx_in] + tau_rayleigh[idx_in];
            tau[idx_out] = tau_tot;
            g  [idx_out] = Real(0.);
            if (tau_tot>(Real(2.)*tmin))
                ssa[idx_out] = tau_rayleigh[idx_in]/tau_tot;
            else
                ssa[idx_out] = Real(0.);
        }
    }
}

namespace rrtmgp_kernel_launcher_cuda
{
    void reorder123x321(const int ni, const int nj, const int nk,
                        const Array_gpu<Real,3>& arr_in, Array_gpu<Real,3>& arr_out)
    {
        const int block_i = 32;
        const int block_j = 16;
        const int block_k = 1;

        const int grid_i  = ni/block_i + (ni%block_i > 0);
        const int grid_j  = nj/block_j + (nj%block_j > 0);
        const int grid_k  = nk/block_k + (nk%block_k > 0);

        dim3 grid_gpu(grid_i, grid_j, grid_k);
        dim3 block_gpu(block_i, block_j, block_k);

        reorder123x321_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, nk, arr_in.ptr(), arr_out.ptr());
    }

    void reorder12x21(const int ni, const int nj,
                      const Array_gpu<Real,2>& arr_in, Array_gpu<Real,2>& arr_out)
    {
        const int block_i = 32;
        const int block_j = 16;

        const int grid_i  = ni/block_i + (ni%block_i > 0);
        const int grid_j  = nj/block_j + (nj%block_j > 0);

        dim3 grid_gpu(grid_i, grid_j);
        dim3 block_gpu(block_i, block_j);

        reorder12x21_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, arr_in.ptr(), arr_out.ptr());
    }

    void zero_array(const int ni, const int nj, const int nk, Array_gpu<Real,3>& arr)
    {
        const int block_i = 32;
        const int block_j = 16;
        const int block_k = 1;

        const int grid_i  = ni/block_i + (ni%block_i > 0);
        const int grid_j  = nj/block_j + (nj%block_j > 0);
        const int grid_k  = nk/block_k + (nk%block_k > 0);

        dim3 grid_gpu(grid_i, grid_j, grid_k);
        dim3 block_gpu(block_i, block_j, block_k);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, nk, arr.ptr());

    }

    void interpolation(
            const int ncol, const int nlay,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array_gpu<int,2>& flavor,
            const Array_gpu<Real,1>& press_ref_log,
            const Array_gpu<Real,1>& temp_ref,
            Real press_ref_log_delta,
            Real temp_ref_min,
            Real temp_ref_delta,
            Real press_ref_trop_log,
            const Array_gpu<Real,3>& vmr_ref,
            const Array_gpu<Real,2>& play,
            const Array_gpu<Real,2>& tlay,
            Array_gpu<Real,3>& col_gas,
            Array_gpu<int,2>& jtemp,
            Array_gpu<Real,6>& fmajor, Array_gpu<Real,5>& fminor,
            Array_gpu<Real,4>& col_mix,
            Array_gpu<Bool,2>& tropo,
            Array_gpu<int,4>& jeta,
            Array_gpu<int,2>& jpress)
    {
        const int block_lay = 16;
        const int block_col = 32;

        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_lay, grid_col);
        dim3 block_gpu(block_lay, block_col);

        Real tmin = std::numeric_limits<Real>::min();
        interpolation_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngas, nflav, neta, npres, ntemp, tmin,
                flavor.ptr(), press_ref_log.ptr(), temp_ref.ptr(),
                press_ref_log_delta, temp_ref_min,
                temp_ref_delta, press_ref_trop_log,
                vmr_ref.ptr(), play.ptr(), tlay.ptr(),
                col_gas.ptr(), jtemp.ptr(), fmajor.ptr(),
                fminor.ptr(), col_mix.ptr(), tropo.ptr(),
                jeta.ptr(), jpress.ptr());

    }

    void combine_and_reorder_2str(
            const int ncol, const int nlay, const int ngpt,
            const Array_gpu<Real,3>& tau_abs, const Array_gpu<Real,3>& tau_rayleigh,
            Array_gpu<Real,3>& tau, Array_gpu<Real,3>& ssa, Array_gpu<Real,3>& g)
    {
        const int block_col = 32;
        const int block_gpt = 32;
        const int block_lay = 1;

        const int grid_col  = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_gpt, grid_lay);
        dim3 block_gpu(block_col, block_gpt, block_lay);

        Real tmin = std::numeric_limits<Real>::min();
        combine_and_reorder_2str_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt, tmin,
                tau_abs.ptr(), tau_rayleigh.ptr(),
                tau.ptr(), ssa.ptr(), g.ptr());
    }

    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array_gpu<int,2>& gpoint_flavor,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<Real,4>& krayl,
            int idx_h2o, const Array_gpu<Real,2>& col_dry, const Array_gpu<Real,3>& col_gas,
            const Array_gpu<Real,5>& fminor, const Array_gpu<int,4>& jeta,
            const Array_gpu<Bool,2>& tropo, const Array_gpu<int,2>& jtemp,
            Array_gpu<Real,3>& tau_rayleigh)
    {
        const int k_size = ncol*nlay*ngpt*sizeof(Real);
        Real* k;
        cuda_safe_call(cudaMalloc((void**)&k, k_size));

        // Call the kernel.
        const int block_bnd = 14;
        const int block_lay = 1;
        const int block_col = 32;

        const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_bnd, grid_lay, grid_col);
        dim3 block_gpu(block_bnd, block_lay, block_col);

        compute_tau_rayleigh_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                gpoint_flavor.ptr(),
                band_lims_gpt.ptr(),
                krayl.ptr(),
                idx_h2o, col_dry.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(),
                tropo.ptr(), jtemp.ptr(),
                tau_rayleigh.ptr(), k);

        cuda_safe_call(cudaFree(k));
    }

    void compute_tau_absorption(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int nminorlower, const int nminorklower,
            const int nminorupper, const int nminorkupper,
            const int idx_h2o,
            const Array_gpu<int,2>& gpoint_flavor,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<Real,4>& kmajor,
            const Array_gpu<Real,3>& kminor_lower,
            const Array_gpu<Real,3>& kminor_upper,
            const Array_gpu<int,2>& minor_limits_gpt_lower,
            const Array_gpu<int,2>& minor_limits_gpt_upper,
            const Array_gpu<Bool,1>& minor_scales_with_density_lower,
            const Array_gpu<Bool,1>& minor_scales_with_density_upper,
            const Array_gpu<Bool,1>& scale_by_complement_lower,
            const Array_gpu<Bool,1>& scale_by_complement_upper,
            const Array_gpu<int,1>& idx_minor_lower,
            const Array_gpu<int,1>& idx_minor_upper,
            const Array_gpu<int,1>& idx_minor_scaling_lower,
            const Array_gpu<int,1>& idx_minor_scaling_upper,
            const Array_gpu<int,1>& kminor_start_lower,
            const Array_gpu<int,1>& kminor_start_upper,
            const Array_gpu<Bool,2>& tropo,
            const Array_gpu<Real,4>& col_mix, const Array_gpu<Real,6>& fmajor,
            const Array_gpu<Real,5>& fminor, const Array_gpu<Real,2>& play,
            const Array_gpu<Real,2>& tlay, const Array_gpu<Real,3>& col_gas,
            const Array_gpu<int,4>& jeta, const Array_gpu<int,2>& jtemp,
            const Array_gpu<int,2>& jpress, Array_gpu<Real,3>& tau)
    {
        const int tau_size = tau.size()*sizeof(Real);
        Real* tau_major;
        Real* tau_minor;
        cuda_safe_call(cudaMalloc((void**)& tau_major, tau_size));
        cuda_safe_call(cudaMalloc((void**)& tau_minor, tau_size));

        const int block_bnd_maj = 14;
        const int block_lay_maj = 1;
        const int block_col_maj = 32;

        const int grid_bnd_maj  = nband/block_bnd_maj + (nband%block_bnd_maj > 0);
        const int grid_lay_maj  = nlay/block_lay_maj + (nlay%block_lay_maj > 0);
        const int grid_col_maj  = ncol/block_col_maj + (ncol%block_col_maj > 0);

        dim3 grid_gpu_maj(grid_bnd_maj, grid_lay_maj, grid_col_maj);
        dim3 block_gpu_maj(block_bnd_maj, block_lay_maj, block_col_maj);

        compute_tau_major_absorption_kernel<<<grid_gpu_maj, block_gpu_maj>>>(
                ncol, nlay, nband, ngpt,
                nflav, neta, npres, ntemp,
                gpoint_flavor.ptr(), band_lims_gpt.ptr(),
                kmajor.ptr(), col_mix.ptr(), fmajor.ptr(), jeta.ptr(),
                tropo.ptr(), jtemp.ptr(), jpress.ptr(),
                tau.ptr(), tau_major);

        const int nscale_lower = scale_by_complement_lower.dim(1);
        const int nscale_upper = scale_by_complement_upper.dim(1);
        const int block_lay_min = 32;
        const int block_col_min = 32;

        const int grid_lay_min  = nlay/block_lay_min + (nlay%block_lay_min > 0);
        const int grid_col_min  = ncol/block_col_min + (ncol%block_col_min > 0);

        dim3 grid_gpu_min(grid_lay_min, grid_col_min);
        dim3 block_gpu_min(block_lay_min, block_col_min);

        compute_tau_minor_absorption_kernel<<<grid_gpu_min, block_gpu_min>>>(
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nscale_lower, nscale_upper,
                nminorlower, nminorupper,
                nminorklower,nminorkupper,
                idx_h2o,
                gpoint_flavor.ptr(),
                kminor_lower.ptr(), kminor_upper.ptr(),
                minor_limits_gpt_lower.ptr(), minor_limits_gpt_upper.ptr(),
                minor_scales_with_density_lower.ptr(), minor_scales_with_density_upper.ptr(),
                scale_by_complement_lower.ptr(), scale_by_complement_upper.ptr(),
                idx_minor_lower.ptr(), idx_minor_upper.ptr(),
                idx_minor_scaling_lower.ptr(), idx_minor_scaling_upper.ptr(),
                kminor_start_lower.ptr(), kminor_start_upper.ptr(),
                play.ptr(), tlay.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                tropo.ptr(), tau.ptr(), tau_minor);

        cuda_safe_call(cudaFree(tau_major));
        cuda_safe_call(cudaFree(tau_minor));
    }

    void Planck_source(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int nPlanckTemp,
            const Array_gpu<Real,2>& tlay,
            const Array_gpu<Real,2>& tlev,
            const Array_gpu<Real,1>& tsfc,
            const int sfc_lay,
            const Array_gpu<Real,6>& fmajor,
            const Array_gpu<int,4>& jeta,
            const Array_gpu<Bool,2>& tropo,
            const Array_gpu<int,2>& jtemp,
            const Array_gpu<int,2>& jpress,
            const Array_gpu<int,1>& gpoint_bands,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<Real,4>& pfracin,
            const Real temp_ref_min, const Real totplnk_delta,
            const Array_gpu<Real,2>& totplnk,
            const Array_gpu<int,2>& gpoint_flavor,
            Array_gpu<Real,2>& sfc_src,
            Array_gpu<Real,3>& lay_src,
            Array_gpu<Real,3>& lev_src_inc,
            Array_gpu<Real,3>& lev_src_dec,
            Array_gpu<Real,2>& sfc_src_jac)
    {
        Real ones_cpu[2] = {Real(1.), Real(1.)};
        const Real delta_Tsurf = Real(1.);

        const int pfrac_size = lay_src.size() * sizeof(Real);
        const int ones_size = 2 * sizeof(Real);
        Real* pfrac;
        Real* ones;

        cuda_safe_call(cudaMalloc((void**)& pfrac, pfrac_size));
        cuda_safe_call(cudaMalloc((void**)& ones, ones_size));

        // Copy the data to the GPU.
        cuda_safe_call(cudaMemcpy(ones, ones_cpu, ones_size, cudaMemcpyHostToDevice));

        // Call the kernel.
        const int block_bnd = 14;
        const int block_lay = 1;
        const int block_col = 32;

        const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_bnd, grid_lay, grid_col);
        dim3 block_gpu(block_bnd, block_lay, block_col);

        Planck_source_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ngpt,
                nflav, neta, npres, ntemp, nPlanckTemp,
                tlay.ptr(), tlev.ptr(), tsfc.ptr(), sfc_lay,
                fmajor.ptr(), jeta.ptr(), tropo.ptr(), jtemp.ptr(),
                jpress.ptr(), gpoint_bands.ptr(), band_lims_gpt.ptr(),
                pfracin.ptr(), temp_ref_min, totplnk_delta,
                totplnk.ptr(), gpoint_flavor.ptr(), ones,
                delta_Tsurf, sfc_src.ptr(), lay_src.ptr(),
                lev_src_inc.ptr(), lev_src_dec.ptr(),
                sfc_src_jac.ptr(), pfrac);

        cuda_safe_call(cudaFree(pfrac));
        cuda_safe_call(cudaFree(ones));
    }
}
