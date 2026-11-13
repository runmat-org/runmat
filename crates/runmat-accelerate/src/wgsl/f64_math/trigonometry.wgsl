// Translated from rust-lang/libm 0.2.16 sin.rs, cos.rs and tan.rs.
// Origins: FreeBSD /usr/src/lib/msun/src/s_sin.c, s_cos.c, s_tan.c.
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
// See UPSTREAM_TRIG.md. Floating exception flags are not exposed by WGSL.
// Tiny inputs return directly, including both signed zeros and subnormals.

fn runmat_math64_sin(x: f64) -> f64 {
    let ix = runmat_f64_high(x) & 0x7fffffffu;
    if (ix <= 0x3fe921fbu) {
        if (ix < 0x3e500000u) { return x; }
        return runmat_math64_k_sin(x, f64(0.0), 0);
    }
    if (ix >= 0x7ff00000u) { return x - x; }
    let reduced = runmat_math64_rem_pio2(x);
    switch (reduced.n & 3) {
        case 0: { return runmat_math64_k_sin(reduced.hi, reduced.lo, 1); }
        case 1: { return runmat_math64_k_cos(reduced.hi, reduced.lo); }
        case 2: { return -runmat_math64_k_sin(reduced.hi, reduced.lo, 1); }
        default: { return -runmat_math64_k_cos(reduced.hi, reduced.lo); }
    }
}

fn runmat_math64_cos(x: f64) -> f64 {
    let ix = runmat_f64_high(x) & 0x7fffffffu;
    if (ix <= 0x3fe921fbu) {
        if (ix < 0x3e46a09eu) { return f64(1.0); }
        return runmat_math64_k_cos(x, f64(0.0));
    }
    if (ix >= 0x7ff00000u) { return x - x; }
    let reduced = runmat_math64_rem_pio2(x);
    switch (reduced.n & 3) {
        case 0: { return runmat_math64_k_cos(reduced.hi, reduced.lo); }
        case 1: { return -runmat_math64_k_sin(reduced.hi, reduced.lo, 1); }
        case 2: { return -runmat_math64_k_cos(reduced.hi, reduced.lo); }
        default: { return runmat_math64_k_sin(reduced.hi, reduced.lo, 1); }
    }
}

fn runmat_math64_tan(x: f64) -> f64 {
    let ix = runmat_f64_high(x) & 0x7fffffffu;
    if (ix <= 0x3fe921fbu) {
        if (ix < 0x3e400000u) { return x; }
        return runmat_math64_k_tan(x, f64(0.0), 0);
    }
    if (ix >= 0x7ff00000u) { return x - x; }
    let reduced = runmat_math64_rem_pio2(x);
    return runmat_math64_k_tan(reduced.hi, reduced.lo, reduced.n & 1);
}
