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

#ifndef GAS_OPTICS_H
#define GAS_OPTICS_H

#include <string>

#include "Array.h"
#include "Optical_props.h"

// Forward declarations.
class Gas_concs;
class Source_func_lw;
template<typename TF> class Source_func_lw_gpu;

class Gas_optics : public Optical_props
{
    public:
        Gas_optics(
                const Array<Real,2>& band_lims_wvn,
                const Array<int,2>& band_lims_gpt) :
            Optical_props(band_lims_wvn, band_lims_gpt)
        {}

        virtual ~Gas_optics() {};

        virtual bool source_is_internal() const = 0;
        virtual bool source_is_external() const = 0;

        virtual Real get_press_ref_min() const = 0;
        virtual Real get_press_ref_max() const = 0;

        virtual Real get_temp_min() const = 0;
        virtual Real get_temp_max() const = 0;

        // Longwave variant.
        virtual void gas_optics(
                const Array<Real,2>& play,
                const Array<Real,2>& plev,
                const Array<Real,2>& tlay,
                const Array<Real,1>& tsfc,
                const Gas_concs& gas_desc,
                std::unique_ptr<Optical_props_arry>& optical_props,
                Source_func_lw& sources,
                const Array<Real,2>& col_dry,
                const Array<Real,2>& tlev) const = 0;

        // Shortwave variant.
        virtual void gas_optics(
                const Array<Real,2>& play,
                const Array<Real,2>& plev,
                const Array<Real,2>& tlay,
                const Gas_concs& gas_desc,
                std::unique_ptr<Optical_props_arry>& optical_props,
                Array<Real,2>& toa_src,
                const Array<Real,2>& col_dry) const = 0;

        virtual Real get_tsi() const = 0;
};

#ifdef USECUDA
template<typename TF>
class Gas_optics_gpu : public Optical_props_gpu<TF>
{
    public:
        Gas_optics_gpu(
                const Array<TF,2>& band_lims_wvn,
                const Array<int,2>& band_lims_gpt) :
            Optical_props_gpu<TF>(band_lims_wvn, band_lims_gpt)
        {}

        virtual ~Gas_optics_gpu() {};

        virtual bool source_is_internal() const = 0;
        virtual bool source_is_external() const = 0;

        virtual TF get_press_ref_min() const = 0;
        virtual TF get_press_ref_max() const = 0;

        virtual TF get_temp_min() const = 0;
        virtual TF get_temp_max() const = 0;

        // Longwave variant.
        virtual void gas_optics(
                const Array_gpu<TF,2>& play,
                const Array_gpu<TF,2>& plev,
                const Array_gpu<TF,2>& tlay,
                const Array_gpu<TF,1>& tsfc,
                const Gas_concs_gpu<TF>& gas_desc,
                std::unique_ptr<Optical_props_arry_gpu<TF>>& optical_props,
                Source_func_lw_gpu<TF>& sources,
                const Array_gpu<TF,2>& col_dry,
                const Array_gpu<TF,2>& tlev) const = 0;

        // Shortwave variant.
        virtual void gas_optics(
                const Array_gpu<TF,2>& play,
                const Array_gpu<TF,2>& plev,
                const Array_gpu<TF,2>& tlay,
                const Gas_concs_gpu<TF>& gas_desc,
                std::unique_ptr<Optical_props_arry_gpu<TF>>& optical_props,
                Array_gpu<TF,2>& toa_src,
                const Array_gpu<TF,2>& col_dry) const = 0;

       virtual TF get_tsi() const = 0;
};
#endif


#endif
