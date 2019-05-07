#ifndef RTE_LW_H
#define RTE_LW_H

namespace rrtmgp_kernels
{
    extern "C" void apply_BC_0(
            int* ncol, int* nlay, int* ngpt,
            int* top_at_1, double* gpt_flux_dn);

    extern "C" void lw_solver_noscat_GaussQuad(
            int* ncol, int* nlay, int* ngpt, int* top_at_1, int* n_quad_angs,
            double* gauss_Ds_subset, double* gauss_wts_subset,
            double* tau,
            double* lay_source, double* lev_source_inc, double* lev_source_dec,
            double* sfc_emis_gpt, double* sfc_source,
            double* gpt_flux_up, double* gpt_flux_dn);

    template<typename TF>
    void apply_BC(
            int ncol, int nlay, int ngpt,
            int top_at_1, Array<TF,3>& gpt_flux_dn)
    {
        apply_BC_0(
                &ncol, &nlay, &ngpt,
                &top_at_1, gpt_flux_dn.v().data());
    }

    template<typename TF>
    void lw_solver_noscat_GaussQuad(
            int ncol, int nlay, int ngpt, int top_at_1, int n_quad_angs,
            const Array<TF,2>& gauss_Ds_subset,
            const Array<TF,2>& gauss_wts_subset,
            const Array<TF,3>& tau,
            const Array<TF,3>& lay_source, const Array<TF,3>& lev_source_inc, const Array<TF,3>& lev_source_dec,
            const Array<TF,2>& sfc_emis_gpt, const Array<TF,2>& sfc_source,
            Array<TF,3>& gpt_flux_up, Array<TF,3>& gpt_flux_dn)
    {
        lw_solver_noscat_GaussQuad(
                &ncol, &nlay, &ngpt, &top_at_1, &n_quad_angs,
                const_cast<TF*>(gauss_Ds_subset.v().data()),
                const_cast<TF*>(gauss_wts_subset.v().data()),
                const_cast<TF*>(tau.v().data()),
                const_cast<TF*>(lay_source.v().data()),
                const_cast<TF*>(lev_source_inc.v().data()),
                const_cast<TF*>(lev_source_dec.v().data()),
                const_cast<TF*>(sfc_emis_gpt.v().data()),
                const_cast<TF*>(sfc_source.v().data()),
                gpt_flux_up.v().data(),
                gpt_flux_dn.v().data());
    }
}

template<typename TF>
class Rte_lw
{
    public:
        static void rte_lw(
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                const int top_at_1,
                const Source_func_lw<TF>& sources,
                const Array<TF,2>& sfc_emis,
                std::unique_ptr<Fluxes<TF>>& fluxes,
                const int n_gauss_angles)
        {
            const int max_gauss_pts = 4;
            const Array<TF,2> gauss_Ds(
                    {      1.66,         0.,         0.,         0.,
                     1.18350343, 2.81649655,         0.,         0.,
                     1.09719858, 1.69338507, 4.70941630,         0.,
                     1.06056257, 1.38282560, 2.40148179, 7.15513024},
                    { max_gauss_pts, max_gauss_pts });

            const Array<TF,2> gauss_wts(
                    {         0.5,           0.,           0.,           0.,
                     0.3180413817, 0.1819586183,           0.,           0.,
                     0.2009319137, 0.2292411064, 0.0698269799,           0.,
                     0.1355069134, 0.2034645680, 0.1298475476, 0.0311809710},
                    { max_gauss_pts, max_gauss_pts });

            const int ncol  = optical_props->get_ncol();
            const int nlay  = optical_props->get_nlay();
            const int ngpt  = optical_props->get_ngpt();
            const int nband = optical_props->get_nband();

            Array<TF,3> gpt_flux_up({ncol, nlay+1, ngpt});
            Array<TF,3> gpt_flux_dn({ncol, nlay+1, ngpt});
            Array<TF,2> sfc_emis_gpt({ncol, ngpt});

            expand_and_transpose(optical_props, sfc_emis, sfc_emis_gpt);

            // Upper boundary condition.
            rrtmgp_kernels::apply_BC(ncol, nlay, ngpt, top_at_1, gpt_flux_dn);

            // Run the radiative transfer solver
            const int n_quad_angs = 1;

            Array<TF,2> gauss_Ds_subset  = gauss_Ds .subset({ {{1, n_quad_angs}, {n_quad_angs, n_quad_angs}} });
            Array<TF,2> gauss_wts_subset = gauss_wts.subset({ {{1, n_quad_angs}, {n_quad_angs, n_quad_angs}} });

            rrtmgp_kernels::lw_solver_noscat_GaussQuad(
                    ncol, nlay, ngpt, top_at_1, n_quad_angs,
                    gauss_Ds_subset, gauss_wts_subset,
                    optical_props->get_tau(),
                    sources.get_lay_source(), sources.get_lev_source_inc(), sources.get_lev_source_dec(),
                    sfc_emis_gpt, sources.get_sfc_source(),
                    gpt_flux_up, gpt_flux_dn);

            fluxes->reduce(gpt_flux_up, gpt_flux_dn, optical_props, top_at_1);
        }

        static void expand_and_transpose(
                const std::unique_ptr<Optical_props_arry<TF>>& ops,
                const Array<TF,2> arr_in,
                Array<TF,2> arr_out)
        {
            const int ncol = arr_in.dim(2);
            const int nband = ops->get_nband();
            const int ngpt = ops->get_ngpt();
            Array<int,2> limits = ops->get_band_lims_gpoint();

            for (int iband=1; iband<=nband; ++iband)
                for (int icol=1; icol<=ncol; ++icol)
                    for (int igpt=limits({1, iband}); igpt<=limits({2, iband}); ++igpt)
                        arr_out({icol, igpt}) = arr_in({iband, icol});
        }
};
#endif
