
#include "rte_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"

    template<typename TF>__device__
    void sw_adding_kernel(const int icol, const int igpt,
                          const int ncol, const int nlay, const BOOL_TYPE top_at_1,
                          const TF* __restrict__ sfc_alb_dif, const TF* __restrict__ r_dif, const TF* __restrict__ t_dif,
                          const TF* __restrict__ source_dn, const TF* __restrict__ source_up, const TF* __restrict__ source_sfc,
                          TF* __restrict__ flux_up, TF* __restrict__ flux_dn, const TF* __restrict__ flux_dir,
                          TF* __restrict__ albedo, TF* __restrict__ src, TF* __restrict__ denom)
    {
        if (top_at_1)
        {
            const int sfc_idx_3d = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            const int sfc_idx_2d = icol + igpt*ncol;
            albedo[sfc_idx_3d] = sfc_alb_dif[sfc_idx_2d];
            src[sfc_idx_3d] = source_sfc[sfc_idx_2d];

            for (int ilay=nlay-1; ilay >= 0; --ilay)
            {
                const int lay_idx  = icol + ilay*ncol + igpt*ncol*nlay;
                const int lev_idx1 = icol + ilay*ncol + igpt*ncol*(nlay+1);
                const int lev_idx2 = icol + (ilay+1)*ncol + igpt*ncol*(nlay+1);
                denom[lay_idx] = TF(1.)/(TF(1.) - r_dif[lay_idx] * albedo[lev_idx2]);
                albedo[lev_idx1] = r_dif[lay_idx] + t_dif[lay_idx] * t_dif[lay_idx]
                                                  * albedo[lev_idx2] * denom[lay_idx];
                src[lev_idx1] = source_up[lay_idx] + t_dif[lay_idx] * denom[lay_idx] *
                                (src[lev_idx2] + albedo[lev_idx2] * source_dn[lay_idx]);
            }
            const int top_idx = icol + igpt*(nlay+1)*ncol;
            flux_up[top_idx] = flux_dn[top_idx]*albedo[top_idx] + src[top_idx];

            for (int ilay=1; ilay < (nlay+1); ++ilay)
            {
                const int lev_idx1 = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                const int lev_idx2 = icol + (ilay-1)*ncol + igpt*(nlay+1)*ncol;
                const int lay_idx = icol + (ilay-1)*ncol + igpt*(nlay)*ncol;
                flux_dn[lev_idx1] = (t_dif[lay_idx]*flux_dn[lev_idx2] +
                                     r_dif[lay_idx]*src[lev_idx1] +
                                     source_dn[lay_idx]) * denom[lay_idx];
                flux_up[lev_idx1] = flux_dn[lev_idx1] * albedo[lev_idx1] + src[lev_idx1];
            }

            for (int ilay=0; ilay < (nlay+1); ++ilay)
            {
                const int idx = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                flux_dn[idx] += flux_dir[idx];
            }
        }
        else
        {
            const int sfc_idx_3d = icol + igpt*(nlay+1)*ncol;
            const int sfc_idx_2d = icol + igpt*ncol;
            albedo[sfc_idx_3d] = sfc_alb_dif[sfc_idx_2d];
            src[sfc_idx_3d] = source_sfc[sfc_idx_2d];

            for (int ilay=0; ilay<nlay; ++ilay)
            {
                const int lay_idx  = icol + ilay*ncol + igpt*ncol*nlay;
                const int lev_idx1 = icol + ilay*ncol + igpt*ncol*(nlay+1);
                const int lev_idx2 = icol + (ilay+1)*ncol + igpt*ncol*(nlay+1);
                denom[lay_idx] = TF(1.)/(TF(1.) - r_dif[lay_idx] * albedo[lev_idx1]);
                albedo[lev_idx2] = r_dif[lay_idx] + (t_dif[lay_idx] * t_dif[lay_idx] *
                                                     albedo[lev_idx1] * denom[lay_idx]);
                src[lev_idx2] = source_up[lay_idx] + t_dif[lay_idx]*denom[lay_idx]*
                                                     (src[lev_idx1]+albedo[lev_idx1]*source_dn[lay_idx]);
            }
            const int top_idx = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            flux_up[top_idx] = flux_dn[top_idx] *albedo[top_idx] + src[top_idx];

            for (int ilay=nlay-1; ilay >= 0; --ilay)
            {
                    const int lev_idx1 = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                    const int lev_idx2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                    const int lay_idx = icol + ilay*ncol + igpt*nlay*ncol;
                    flux_dn[lev_idx1] = (t_dif[lay_idx]*flux_dn[lev_idx2] +
                                         r_dif[lay_idx]*src[lev_idx1] +
                                         source_dn[lay_idx]) * denom[lay_idx];
                    flux_up[lev_idx1] = flux_dn[lev_idx1] * albedo[lev_idx1] + src[lev_idx1];
            }
            for (int ilay=nlay; ilay >= 0; --ilay)
            {
                const int idx = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                flux_dn[idx] += flux_dir[idx];
            }
        }
    }

    template<typename TF>__device__
    void sw_source_kernel(const int icol, const int igpt,
                          const int ncol, const int nlay, const BOOL_TYPE top_at_1,
                          TF* __restrict__ r_dir, TF* __restrict__ t_dir, TF* __restrict__ t_noscat,
                          const TF* __restrict__ sfc_alb_dir, TF* __restrict__ source_up, TF* __restrict__ source_dn,
                          TF* __restrict__ source_sfc, TF* __restrict__ flux_dir)
    {

        if (top_at_1)
        {
            for (int ilay=0; ilay<nlay; ++ilay)
            {
                const int idx_lay  = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_lev1 = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                const int idx_lev2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                source_up[idx_lay] = r_dir[idx_lay] * flux_dir[idx_lev1];
                source_dn[idx_lay] = t_dir[idx_lay] * flux_dir[idx_lev1];
                flux_dir[idx_lev2] = t_noscat[idx_lay] * flux_dir[idx_lev1];

            }
            const int sfc_idx = icol + igpt*ncol;
            const int flx_idx = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            source_sfc[sfc_idx] = flux_dir[flx_idx] * sfc_alb_dir[icol];
        }
        else
        {
            for (int ilay=nlay-1; ilay>=0; --ilay)
            {
                const int idx_lay  = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_lev1 = icol + (ilay)*ncol + igpt*(nlay+1)*ncol;
                const int idx_lev2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                source_up[idx_lay] = r_dir[idx_lay] * flux_dir[idx_lev2];
                source_dn[idx_lay] = t_dir[idx_lay] * flux_dir[idx_lev2];
                flux_dir[idx_lev1] = t_noscat[idx_lay] * flux_dir[idx_lev2];

            }
            const int sfc_idx = icol + igpt*ncol;
            const int flx_idx = icol + igpt*(nlay+1)*ncol;
            source_sfc[sfc_idx] = flux_dir[flx_idx] * sfc_alb_dir[icol];
        }

    }

    template<typename TF>__global__
    void sw_source_adding_kernel(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                                 const TF* __restrict__ sfc_alb_dir, const TF* __restrict__ sfc_alb_dif,
                                 TF* __restrict__ r_dif, TF* __restrict__ t_dif,
                                 TF* __restrict__ r_dir, TF* __restrict__ t_dir, TF* __restrict__ t_noscat,
                                 TF* __restrict__ flux_up, TF* __restrict__ flux_dn, TF* __restrict__ flux_dir,
                                 TF* __restrict__ source_up, TF* __restrict__ source_dn, TF* __restrict__ source_sfc,
                                 TF* __restrict__ albedo, TF* __restrict__ src, TF* __restrict__ denom)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int igpt = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (igpt < ngpt) )
        {
            sw_source_kernel(icol, igpt, ncol, nlay, top_at_1, r_dir, t_dir,
                             t_noscat, sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir);

            sw_adding_kernel(icol, igpt, ncol, nlay, top_at_1, sfc_alb_dif,
                             r_dif, t_dif, source_dn, source_up, source_sfc,
                             flux_up, flux_dn, flux_dir, albedo, src, denom);
        }
    }

