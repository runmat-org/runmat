// Translated from rust-lang/libm 0.2.16 (pinned registry source).
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// See UPSTREAM_ELEMENTARY.md for source hashes, MIT permission and changes.
// Requires controlled binary64 evaluation; floating exception flags omitted.

const runmat_math64_log2_IVLN2HI: f64 = 1.44269504072144627571e+00;
const runmat_math64_log2_IVLN2LO: f64 = 1.67517131648865118353e-10;
const runmat_math64_log2_LG1: f64 = 6.666666666666735130e-01;
const runmat_math64_log2_LG2: f64 = 3.999999999940941908e-01;
const runmat_math64_log2_LG3: f64 = 2.857142874366239149e-01;
const runmat_math64_log2_LG4: f64 = 2.222219843214978396e-01;
const runmat_math64_log2_LG5: f64 = 1.818357216161805012e-01;
const runmat_math64_log2_LG6: f64 = 1.531383769920937332e-01;
const runmat_math64_log2_LG7: f64 = 1.479819860511658591e-01;

fn runmat_math64_log2(argument_x: f64) -> f64 {
var x = argument_x;
    var x1p54 = runmat_f64_from_words(0x0u, 0x43500000u);

    var ui: u64 = bitcast<u64>(x);
    var hfsq: f64;
    var f: f64;
    var s: f64;
    var z: f64;
    var r: f64;
    var w: f64;
    var t1: f64;
    var t2: f64;
    var y: f64;
    var hi: f64;
    var lo: f64;
    var val_hi: f64;
    var val_lo: f64;
    var hx: u32;
    var k: i32;

    hx = u32((ui >> 32));
    k = 0;
    if hx < 0x00100000 || (hx >> 31) > 0 {
        if ui << 1 == 0 {
            return runmat_f64_from_words(0u, 0xfff00000u);
        }
        if (hx >> 31) > 0 {
            return runmat_f64_from_words(0u, 0x7ff80000u);
        }

        k -= 54;
        x = runmat_math64_scalbn(x, 54);
        ui = bitcast<u64>(x);
        hx = u32((ui >> 32));
    } else if hx >= 0x7ff00000 {
        return x;
    } else if hx == 0x3ff00000 && ui << 32 == 0 {
        return f64(0.0);
    }

    hx += 0x3ff00000 - 0x3fe6a09e;
    k += i32((hx >> 20)) - 0x3ff;
    hx = (hx & 0x000fffff) + 0x3fe6a09e;
    ui = ((u64(hx)) << 32) | (ui & 0xffffffff);
    x = bitcast<f64>(ui);

    f = x - 1.0;
    hfsq = 0.5 * f * f;
    s = f / (2.0 + f);
    z = s * s;
    w = z * z;
    t1 = w * (runmat_math64_log2_LG2 + w * (runmat_math64_log2_LG4 + w * runmat_math64_log2_LG6));
    t2 = z * (runmat_math64_log2_LG1 + w * (runmat_math64_log2_LG3 + w * (runmat_math64_log2_LG5 + w * runmat_math64_log2_LG7)));
    r = t2 + t1;

    hi = f - hfsq;
    ui = bitcast<u64>(hi);
    ui &= (((u64(0xffffffffu) << 32u) | u64(0xffffffffu))) << 32;
    hi = bitcast<f64>(ui);
    lo = f - hi - hfsq + s * (hfsq + r);

    val_hi = hi * runmat_math64_log2_IVLN2HI;
    val_lo = (lo + hi) * runmat_math64_log2_IVLN2LO + lo * runmat_math64_log2_IVLN2HI;

    y = f64(k);
    w = y + val_hi;
    val_lo += (y - w) + val_hi;
    val_hi = w;

    return val_lo + val_hi;
}
