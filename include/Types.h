#ifndef TYPES_H
#define TYPES_H

#ifdef RTE_RRTMGP_USE_CBOOL
using Bool = signed char;
#else
using Bool = int;
#endif

#ifdef RTE_RRTMGP_SINGLE_PRECISION
using Real = float;
#else
using Real = double;
#endif

using Real_ptr = Real* const __restrict__;
#endif
