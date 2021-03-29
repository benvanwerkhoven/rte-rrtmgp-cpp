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

#ifndef RTE_LW_H
#define RTE_LW_H

#include <memory>
#include "Types.h"

// Forward declarations.
template<typename, int> class Array;
template<typename> class Optical_props_arry;
class Source_func_lw;
class Fluxes_broadband;

class Rte_lw
{
    public:
        static void rte_lw(
                const std::unique_ptr<Optical_props_arry<Real>>& optical_props,
                const Bool top_at_1,
                const Source_func_lw& sources,
                const Array<Real,2>& sfc_emis,
                const Array<Real,2>& inc_flux,
                Array<Real,3>& gpt_flux_up,
                Array<Real,3>& gpt_flux_dn,
                const int n_gauss_angles);

        static void expand_and_transpose(
                const std::unique_ptr<Optical_props_arry<Real>>& ops,
                const Array<Real,2> arr_in,
                Array<Real,2>& arr_out);
};
#endif
