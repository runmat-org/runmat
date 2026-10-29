// Adapted for WGSL from rust-lang/libm 0.2.16 src/math/log.rs.
// Origin: FreeBSD /usr/src/lib/msun/src/e_log.c.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// See UPSTREAM.md for source identity and the accompanying MIT permission.
// Modifications: encoded exceptional values and exact subnormal scaling;
// floating exception flags are not exposed.

fn runmat_math64_log(argument: f64) -> f64 {
    const ln2_hi: f64 = 6.93147180369123816490e-01;
    const ln2_lo: f64 = 1.90821492927058770002e-10;
    const lg1: f64 = 6.666666666666735130e-01;
    const lg2: f64 = 3.999999999940941908e-01;
    const lg3: f64 = 2.857142874366239149e-01;
    const lg4: f64 = 2.222219843214978396e-01;
    const lg5: f64 = 1.818357216161805012e-01;
    const lg6: f64 = 1.531383769920937332e-01;
    const lg7: f64 = 1.479819860511658591e-01;
    var x = argument;
    var high = runmat_f64_high(x);
    let low = runmat_f64_low(x);
    if ((high & 0x7fffffffu) == 0u && low == 0u) {
        return runmat_f64_from_words(0u, 0xfff00000u);
    }
    if ((high & 0x7fffffffu) > 0x7ff00000u ||
        ((high & 0x7fffffffu) == 0x7ff00000u && low != 0u)) {
        return argument;
    }
    if ((high >> 31u) != 0u) { return runmat_f64_from_words(0u, 0x7ff80000u); }
    if (high >= 0x7ff00000u) { return argument; }
    if (high == 0x3ff00000u && low == 0u) { return f64(0.0); }
    var k = 0;
    if (high < 0x00100000u) {
        k = -54;
        x = runmat_math64_scalbn(x, 54);
        high = runmat_f64_high(x);
    }
    high = high + 0x3ff00000u - 0x3fe6a09eu;
    k = k + i32(high >> 20u) - 0x3ff;
    high = (high & 0x000fffffu) + 0x3fe6a09eu;
    x = runmat_f64_from_words(runmat_f64_low(x), high);
    let f = x - f64(1.0);
    let half_square = f64(0.5) * f * f;
    let s = f / (f64(2.0) + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (lg2 + w * (lg4 + w * lg6));
    let t2 = z * (lg1 + w * (lg3 + w * (lg5 + w * lg7)));
    let r = t2 + t1;
    let dk = f64(k);
    return s * (half_square + r) + dk * ln2_lo - half_square + f + dk * ln2_hi;
}
