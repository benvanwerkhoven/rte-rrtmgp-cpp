/*
 * This file is imported from MicroHH (https://github.com/earth-system-radiation/earth-system-radiation)
 * and is adapted for the testing of the C++ interface to the
 * RTE+RRTMGP radiation code.
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef RADIATION_SOLVER_H
#define RADIATION_SOLVER_H


#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics_rrtmgp.h"
#include "Cloud_optics.h"


class Radiation_solver_longwave
{
    public:
        Radiation_solver_longwave(
                const Gas_concs& gas_concs,
                const std::string& file_name_gas,
                const std::string& file_name_cloud);

        void solve(
                const bool switch_fluxes,
                const bool switch_cloud_optics,
                const bool switch_output_optical,
                const bool switch_output_bnd_fluxes,
                const Gas_concs& gas_concs,
                const Array<Real,2>& p_lay, const Array<Real,2>& p_lev,
                const Array<Real,2>& t_lay, const Array<Real,2>& t_lev,
                const Array<Real,2>& col_dry,
                const Array<Real,1>& t_sfc, const Array<Real,2>& emis_sfc,
                const Array<Real,2>& lwp, const Array<Real,2>& iwp,
                const Array<Real,2>& rel, const Array<Real,2>& rei,
                Array<Real,3>& tau, Array<Real,3>& lay_source,
                Array<Real,3>& lev_source_inc, Array<Real,3>& lev_source_dec, Array<Real,2>& sfc_source,
                Array<Real,2>& lw_flux_up, Array<Real,2>& lw_flux_dn, Array<Real,2>& lw_flux_net,
                Array<Real,3>& lw_bnd_flux_up, Array<Real,3>& lw_bnd_flux_dn, Array<Real,3>& lw_bnd_flux_net) const;

        int get_n_gpt() const { return this->kdist->get_ngpt(); };
        int get_n_bnd() const { return this->kdist->get_nband(); };

        Array<int,2> get_band_lims_gpoint() const
        { return this->kdist->get_band_lims_gpoint(); }

        Array<Real,2> get_band_lims_wavenumber() const
        { return this->kdist->get_band_lims_wavenumber(); }

    private:
        std::unique_ptr<Gas_optics_rrtmgp> kdist;
        std::unique_ptr<Cloud_optics> cloud_optics;
};

class Radiation_solver_shortwave
{
    public:
        Radiation_solver_shortwave(
                const Gas_concs& gas_concs,
                const std::string& file_name_gas,
                const std::string& file_name_cloud);

        void solve(
                const bool switch_fluxes,
                const bool switch_cloud_optics,
                const bool switch_output_optical,
                const bool switch_output_bnd_fluxes,
                const Gas_concs& gas_concs,
                const Array<Real,2>& p_lay, const Array<Real,2>& p_lev,
                const Array<Real,2>& t_lay, const Array<Real,2>& t_lev,
                const Array<Real,2>& col_dry,
                const Array<Real,2>& sfc_alb_dir, const Array<Real,2>& sfc_alb_dif,
                const Array<Real,1>& tsi_scaling, const Array<Real,1>& mu0,
                const Array<Real,2>& lwp, const Array<Real,2>& iwp,
                const Array<Real,2>& rel, const Array<Real,2>& rei,
                Array<Real,3>& tau, Array<Real,3>& ssa, Array<Real,3>& g,
                Array<Real,2>& toa_src,
                Array<Real,2>& sw_flux_up, Array<Real,2>& sw_flux_dn,
                Array<Real,2>& sw_flux_dn_dir, Array<Real,2>& sw_flux_net,
                Array<Real,3>& sw_bnd_flux_up, Array<Real,3>& sw_bnd_flux_dn,
                Array<Real,3>& sw_bnd_flux_dn_dir, Array<Real,3>& sw_bnd_flux_net) const;

        int get_n_gpt() const { return this->kdist->get_ngpt(); };
        int get_n_bnd() const { return this->kdist->get_nband(); };

        Real get_tsi() const { return this->kdist->get_tsi(); };

        Array<int,2> get_band_lims_gpoint() const
        { return this->kdist->get_band_lims_gpoint(); }

        Array<Real,2> get_band_lims_wavenumber() const
        { return this->kdist->get_band_lims_wavenumber(); }

    private:
        std::unique_ptr<Gas_optics> kdist;
        std::unique_ptr<Cloud_optics> cloud_optics;
};
#endif
