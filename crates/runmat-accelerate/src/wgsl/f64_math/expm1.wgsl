// Translated from rust-lang/libm 0.2.16 (pinned registry source).
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// See UPSTREAM_ELEMENTARY.md for source hashes, MIT permission and changes.
// Requires controlled binary64 evaluation; floating exception flags omitted.

const runmat_math64_expm1_O_THRESHOLD: f64 = 7.09782712893383973096e+02;
const runmat_math64_expm1_LN2_HI: f64 = 6.93147180369123816490e-01;
const runmat_math64_expm1_LN2_LO: f64 = 1.90821492927058770002e-10;
const runmat_math64_expm1_INVLN2: f64 = 1.44269504088896338700e+00;

const runmat_math64_expm1_Q1: f64 = -3.33333333333331316428e-02;
const runmat_math64_expm1_Q2: f64 = 1.58730158725481460165e-03;
const runmat_math64_expm1_Q3: f64 = -7.93650757867487942473e-05;
const runmat_math64_expm1_Q4: f64 = 4.00821782732936239552e-06;
const runmat_math64_expm1_Q5: f64 = -2.01099218183624371326e-07;

fn runmat_math64_expm1(argument_x: f64) -> f64 {
var x = argument_x;
    var hi: f64;
    var lo: f64;
    var k: i32;
    var c: f64;
    var t: f64;
    var y: f64;

    var ui = bitcast<u64>(x);
    var hx = u32(((ui >> 32) & 0x7fffffff));
    var sign = i32((ui >> 63));

    if hx >= 0x4043687A {

        if ((runmat_f64_high(x) & 0x7fffffffu) > 0x7ff00000u || ((runmat_f64_high(x) & 0x7fffffffu) == 0x7ff00000u && runmat_f64_low(x) != 0u)) {
            return x;
        }
        if sign != 0 {
            return f64(-1.0);
        }
        if x > runmat_math64_expm1_O_THRESHOLD {
            return runmat_f64_from_words(0u, 0x7ff00000u);
        }
    }

    if hx > 0x3fd62e42 {

        if hx < 0x3FF0A2B2 {

            if sign == 0 {
                hi = x - runmat_math64_expm1_LN2_HI;
                lo = runmat_math64_expm1_LN2_LO;
                k = 1;
            } else {
                hi = x + runmat_math64_expm1_LN2_HI;
                lo = f64(-1.90821492927058770002e-10);
                k = -1;
            }
        } else {
            k = i32((runmat_math64_expm1_INVLN2 * x + select(f64(0.5), f64(-0.5), sign != 0)));
            t = f64(k);
            hi = x - t * runmat_math64_expm1_LN2_HI;
            lo = t * runmat_math64_expm1_LN2_LO;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if hx < 0x3c900000 {

        return x;
    } else {
        c = 0.0;
        k = 0;
    }

    var hfx = 0.5 * x;
    var hxs = x * hfx;
    var r1 = 1.0 + hxs * (runmat_math64_expm1_Q1 + hxs * (runmat_math64_expm1_Q2 + hxs * (runmat_math64_expm1_Q3 + hxs * (runmat_math64_expm1_Q4 + hxs * runmat_math64_expm1_Q5))));
    t = 3.0 - r1 * hfx;
    var e = hxs * ((r1 - t) / (6.0 - x * t));
    if k == 0 {

        return x - (x * e - hxs);
    }
    e = x * (e - c) - c;
    e -= hxs;

    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        if x < -0.25 {
            return -2.0 * (e - (x + 0.5));
        }
        return 1.0 + 2.0 * (x - e);
    }
    ui = (u64((0x3ff + k))) << 52;
    var twopk = bitcast<f64>(ui);
    if k < 0 || k > 56 {

        y = x - e + 1.0;
        if k == 1024 {
            y = y * 2.0 * runmat_f64_from_words(0x0u, 0x7fe00000u);
        } else {
            y = y * twopk;
        }
        return y - 1.0;
    }
    ui = (u64((0x3ff - k))) << 52;
    var uf = bitcast<f64>(ui);
    if k < 20 {
        y = (x - e + (1.0 - uf)) * twopk;
    } else {
        y = (x - (e + uf) + 1.0) * twopk;
    }
    return y;
}
