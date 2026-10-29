// Translated from rust-lang/libm 0.2.16 (pinned registry source).
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// See UPSTREAM_ELEMENTARY.md for source hashes, MIT permission and changes.
// Requires controlled binary64 evaluation; floating exception flags omitted.

const runmat_math64_log1p_LN2_HI: f64 = 6.93147180369123816490e-01;
const runmat_math64_log1p_LN2_LO: f64 = 1.90821492927058770002e-10;
const runmat_math64_log1p_LG1: f64 = 6.666666666666735130e-01;
const runmat_math64_log1p_LG2: f64 = 3.999999999940941908e-01;
const runmat_math64_log1p_LG3: f64 = 2.857142874366239149e-01;
const runmat_math64_log1p_LG4: f64 = 2.222219843214978396e-01;
const runmat_math64_log1p_LG5: f64 = 1.818357216161805012e-01;
const runmat_math64_log1p_LG6: f64 = 1.531383769920937332e-01;
const runmat_math64_log1p_LG7: f64 = 1.479819860511658591e-01;

fn runmat_math64_log1p(x: f64) -> f64 {
    var ui: u64 = bitcast<u64>(x);
    var hfsq: f64;
    var f: f64 = 0.;
    var c: f64 = 0.;
    var s: f64;
    var z: f64;
    var r: f64;
    var w: f64;
    var t1: f64;
    var t2: f64;
    var dk: f64;
    var hx: u32;
    var hu: u32;
    var k: i32;

    hx = u32((ui >> 32));
    k = 1;
    if hx < 0x3fda827a || (hx >> 31) > 0 {

        if hx >= 0xbff00000 {

            if x == -1. {
                return runmat_f64_from_words(0u, 0xfff00000u);
            }
            return runmat_f64_from_words(0u, 0x7ff80000u);
        }
        if hx << 1 < 0x3ca00000 << 1 {

            if (hx & 0x7ff00000) == 0 {

            }
            return x;
        }
        if hx <= 0xbfd2bec4 {

            k = 0;
            c = 0.;
            f = x;
        }
    } else if hx >= 0x7ff00000 {
        return x;
    }
    if k > 0 {
        ui = bitcast<u64>(1. + x);
        hu = u32((ui >> 32));
        hu += 0x3ff00000 - 0x3fe6a09e;
        k = i32((hu >> 20)) - 0x3ff;

        if k < 54 {
            if k >= 2 { c = 1. - (bitcast<f64>(ui) - x); } else { c = x - (bitcast<f64>(ui) - 1.); }
            c /= bitcast<f64>(ui);
        } else {
            c = 0.;
        }

        hu = (hu & 0x000fffff) + 0x3fe6a09e;
        ui = ((u64(hu)) << 32) | (ui & 0xffffffff);
        f = bitcast<f64>(ui) - 1.;
    }
    hfsq = 0.5 * f * f;
    s = f / (2.0 + f);
    z = s * s;
    w = z * z;
    t1 = w * (runmat_math64_log1p_LG2 + w * (runmat_math64_log1p_LG4 + w * runmat_math64_log1p_LG6));
    t2 = z * (runmat_math64_log1p_LG1 + w * (runmat_math64_log1p_LG3 + w * (runmat_math64_log1p_LG5 + w * runmat_math64_log1p_LG7)));
    r = t2 + t1;
    dk = f64(k);
    return s * (hfsq + r) + (dk * runmat_math64_log1p_LN2_LO + c) - hfsq + f + dk * runmat_math64_log1p_LN2_HI;
}
