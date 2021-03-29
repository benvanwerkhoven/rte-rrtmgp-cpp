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

#ifndef CLOUD_OPTICS_H
#define CLOUD_OPTICS_H

#include "Types.h"
#include "Array.h"
#include "Optical_props.h"


// Forward declarations.
template<typename> class Optical_props;

class Cloud_optics : public Optical_props<Real>
{
    public:
        Cloud_optics(
                const Array<Real,2>& band_lims_wvn,
                const Real radliq_lwr, const Real radliq_upr, const Real radliq_fac,
                const Real radice_lwr, const Real radice_upr, const Real radice_fac,
                const Array<Real,2>& lut_extliq, const Array<Real,2>& lut_ssaliq, const Array<Real,2>& lut_asyliq,
                const Array<Real,3>& lut_extice, const Array<Real,3>& lut_ssaice, const Array<Real,3>& lut_asyice);

        void cloud_optics(
                const Array<Real,2>& clwp, const Array<Real,2>& ciwp,
                const Array<Real,2>& reliq, const Array<Real,2>& reice,
                Optical_props_1scl<Real>& optical_props);

        void cloud_optics(
                const Array<Real,2>& clwp, const Array<Real,2>& ciwp,
                const Array<Real,2>& reliq, const Array<Real,2>& reice,
                Optical_props_2str<Real>& optical_props);

    private:
        int liq_nsteps;
        int ice_nsteps;
        Real liq_step_size;
        Real ice_step_size;

        // Lookup table constants.
        Real radliq_lwr;
        Real radliq_upr;
        Real radice_lwr;
        Real radice_upr;

        // Lookup table coefficients.
        Array<Real,2> lut_extliq;
        Array<Real,2> lut_ssaliq;
        Array<Real,2> lut_asyliq;
        Array<Real,2> lut_extice;
        Array<Real,2> lut_ssaice;
        Array<Real,2> lut_asyice;
};
#endif
