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

#ifndef OPTICAL_PROPS_H
#define OPTICAL_PROPS_H

#include <memory>

#include "Types.h"
#include "Array.h"


class Optical_props
{
    public:
        Optical_props(
                const Array<Real,2>& band_lims_wvn,
                const Array<int,2>& band_lims_gpt);

        Optical_props(
                const Array<Real,2>& band_lims_wvn);

        virtual ~Optical_props() {};

        Optical_props(const Optical_props&) = default;

        Array<int,1> get_gpoint_bands() const { return this->gpt2band; }
        int get_nband() const { return this->band2gpt.dim(2); }
        int get_ngpt() const { return this->band2gpt.max(); }
        Array<int,2> get_band_lims_gpoint() const { return this->band2gpt; }
        Array<Real,2> get_band_lims_wavenumber() const { return this->band_lims_wvn; }

    private:
        Array<int,2> band2gpt;     // (begin g-point, end g-point) = band2gpt(2,band)
        Array<int,1> gpt2band;     // band = gpt2band(g-point)
        Array<Real,2> band_lims_wvn; // (upper and lower wavenumber by band) = band_lims_wvn(2,band)
};


// Base class for 1scl and 2str solvers fully implemented in header.
class Optical_props_arry : public Optical_props
{
    public:
        Optical_props_arry(const Optical_props& optical_props) :
            Optical_props(optical_props)
        {}

        virtual ~Optical_props_arry() {};

        virtual Array<Real,3>& get_tau() = 0;
        virtual Array<Real,3>& get_ssa() = 0;
        virtual Array<Real,3>& get_g  () = 0;

        virtual const Array<Real,3>& get_tau() const = 0;
        virtual const Array<Real,3>& get_ssa() const = 0;
        virtual const Array<Real,3>& get_g  () const = 0;

        // Optional argument.
        virtual void delta_scale(const Array<Real,3>& forward_frac=Array<Real,3>()) = 0;

        virtual void set_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e) = 0;

        virtual void get_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e) = 0;

        virtual int get_ncol() const = 0;
        virtual int get_nlay() const = 0;
};


class Optical_props_1scl : public Optical_props_arry
{
    public:
        // Initializer constructor.
        Optical_props_1scl(
                const int ncol,
                const int nlay,
                const Optical_props& optical_props);

        void set_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e);

        void get_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e);

        int get_ncol() const { return tau.dim(1); }
        int get_nlay() const { return tau.dim(2); }

        Array<Real,3>& get_tau() { return tau; }
        Array<Real,3>& get_ssa() { throw std::runtime_error("ssa is not available in this class"); }
        Array<Real,3>& get_g  () { throw std::runtime_error("g is available in this class"); }

        const Array<Real,3>& get_tau() const { return tau; }
        const Array<Real,3>& get_ssa() const { throw std::runtime_error("ssa is not available in this class"); }
        const Array<Real,3>& get_g  () const { throw std::runtime_error("g is available in this class"); }

        void delta_scale(const Array<Real,3>& forward_frac=Array<Real,3>()) {}

    private:
        Array<Real,3> tau;
};


class Optical_props_2str : public Optical_props_arry
{
    public:
        Optical_props_2str(
                const int ncol,
                const int nlay,
                const Optical_props& optical_props);

        void set_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e);

        void get_subset(
                const std::unique_ptr<Optical_props_arry>& optical_props_sub,
                const int col_s, const int col_e);

        int get_ncol() const { return tau.dim(1); }
        int get_nlay() const { return tau.dim(2); }

        Array<Real,3>& get_tau() { return tau; }
        Array<Real,3>& get_ssa() { return ssa; }
        Array<Real,3>& get_g  () { return g; }

        const Array<Real,3>& get_tau() const { return tau; }
        const Array<Real,3>& get_ssa() const { return ssa; }
        const Array<Real,3>& get_g  () const { return g; }

        void delta_scale(const Array<Real,3>& forward_frac=Array<Real,3>());

    private:
        Array<Real,3> tau;
        Array<Real,3> ssa;
        Array<Real,3> g;
};


void add_to(Optical_props_1scl& op_inout, const Optical_props_1scl& op_in);
void add_to(Optical_props_2str& op_inout, const Optical_props_2str& op_in);
#endif
