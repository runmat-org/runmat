// Adapted for WGSL from rust-lang/libm 0.2.16 src/math/exp.rs.
// Origin: FreeBSD /usr/src/lib/msun/src/e_exp.c.
// Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// See UPSTREAM.md for source identity and the accompanying MIT permission.
// Modifications: encoded exceptional values, WGSL control flow and exact
// integer exponent scaling; floating exception flags are not exposed.

fn runmat_math64_exp(argument: f64) -> f64 {
    const ln2_hi: f64 = 6.93147180369123816490e-01;
    const ln2_lo: f64 = 1.90821492927058770002e-10;
    const inv_ln2: f64 = 1.44269504088896338700e+00;
    const p1: f64 = 1.66666666666666019037e-01;
    const p2: f64 = -2.77777777770155933842e-03;
    const p3: f64 = 6.61375632143793436117e-05;
    const p4: f64 = -1.65339022054652515390e-06;
    const p5: f64 = 4.13813679705723846039e-08;
    let high = runmat_f64_high(argument);
    let magnitude = high & 0x7fffffffu;
    let negative = (high >> 31u) != 0u;
    if (magnitude >= 0x7ff00000u) {
        if ((magnitude & 0x000fffffu) != 0u || runmat_f64_low(argument) != 0u) {
            return argument;
        }
        if (negative) { return f64(0.0); }
        return argument;
    }
    if (argument > f64(709.782712893383973096)) {
        return runmat_f64_from_words(0u, 0x7ff00000u);
    }
    if (argument < f64(-745.13321910194110842)) { return f64(0.0); }
    var x = argument;
    var hi = f64(0.0);
    var lo = f64(0.0);
    var k = 0;
    if (magnitude > 0x3fd62e42u) {
        if (magnitude >= 0x3ff0a2b2u) {
            let half = select(f64(0.5), f64(-0.5), negative);
            k = i32(inv_ln2 * x + half);
        } else {
            k = select(1, -1, negative);
        }
        hi = x - f64(k) * ln2_hi;
        lo = f64(k) * ln2_lo;
        x = hi - lo;
    } else if (magnitude > 0x3e300000u) {
        hi = x;
    } else {
        return f64(1.0) + x;
    }
    let square = x * x;
    let c = x - square * (p1 + square * (p2 + square * (p3 + square * (p4 + square * p5))));
    let y = f64(1.0) + (x * c / (f64(2.0) - c) - lo + hi);
    if (k == 0) { return y; }
    return runmat_math64_scalbn(y, k);
}
