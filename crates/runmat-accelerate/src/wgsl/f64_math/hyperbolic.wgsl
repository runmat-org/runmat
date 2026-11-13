// Translated from rust-lang/libm 0.2.16 (pinned registry source).
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// See UPSTREAM_ELEMENTARY.md for source hashes, MIT permission and changes.
// Requires controlled binary64 evaluation; floating exception flags omitted.

// Hyperbolic libm algorithms preserve their region-specific formulas.
fn runmat_math64_sinh(x: f64) -> f64 {




    var uf: f64 = x;
    var ui: u64 = bitcast<u64>(uf);
    var w: u32;
    var t: f64;
    var h: f64;
    var absx: f64;

    h = 0.5;
    if ui >> 63 != 0 {
        h = -h;
    }

    ui &= ((u64(0x7fffffffu) << 32u) | u64(0xffffffffu));
    uf = bitcast<f64>(ui);
    absx = uf;
    w = u32((ui >> 32));


    if w < 0x40862e42 {
        t = runmat_math64_expm1(absx);
        if w < 0x3ff00000 {
            if w < 0x3ff00000 - (26 << 20) {


                return x;
            }
            return h * (2.0 * t - t * t / (t + 1.0));
        }

        return h * (t + t / (t + 1.0));
    }



    t = 2.0 * h * runmat_math64_hyperbolic_expo2(absx);
    return t;
}

fn runmat_math64_cosh(argument_x: f64) -> f64 {
var x = argument_x;

    var ix = bitcast<u64>(x);
    ix &= ((u64(0x7fffffffu) << 32u) | u64(0xffffffffu));
    x = bitcast<f64>(ix);
    var w = ix >> 32;


    if w < 0x3fe62e42 {
        if w < 0x3ff00000 - (26 << 20) {
            var x1p120 = runmat_f64_from_words(0x0u, 0x47700000u);

            return f64(1.0);
        }
        var t = runmat_math64_expm1(x);
        return 1. + t * t / (2. * (1. + t));
    }


    if w < 0x40862e42 {
        var t = runmat_math64_exp(x);

        return 0.5 * (t + 1. / t);
    }


    return runmat_math64_hyperbolic_expo2(x);
}

fn runmat_math64_tanh(argument_x: f64) -> f64 {
var x = argument_x;
    if (runmat_f64_high(x) & 0x7fffffffu) < 0x00100000u { return x; }
    var uf: f64 = x;
    var ui: u64 = bitcast<u64>(uf);

    var w: u32;
    var sign: bool;
    var t: f64;


    sign = ui >> 63 != 0;
    ui &= ((u64(0x7fffffffu) << 32u) | u64(0xffffffffu));
    uf = bitcast<f64>(ui);
    x = uf;
    w = u32((ui >> 32));

    if w > 0x3fe193ea {

        if w > 0x40340000 {


            if w > 0x7ff00000u || (w == 0x7ff00000u && runmat_f64_low(x) != 0u) { return x; }
            t = f64(1.0);
        } else {
            t = runmat_math64_expm1(2.0 * x);
            t = 1.0 - 2.0 / (t + 2.0);
        }
    } else if w > 0x3fd058ae {

        t = runmat_math64_expm1(2.0 * x);
        t = t / (t + 2.0);
    } else if w >= 0x00100000 {

        t = runmat_math64_expm1(-2.0 * x);
        t = -t / (t + 2.0);
    } else {



        t = x;
    }

    return select(t, -t, sign);
}

const runmat_math64_asinh_LN2: f64 = 0.693147180559945309417232121458176568;

fn runmat_math64_asinh(argument_x: f64) -> f64 {
var x = argument_x;
    if (runmat_f64_high(x) & 0x7fffffffu) < 0x3e500000u { return x; }
    var u = bitcast<u64>(x);
    var e = (u32((u >> 52))) & 0x7ff;
    var sign = (u >> 63) != 0;


    u &= ((u64(0x7fffffffu) << 32u) | u64(0xffffffffu));
    x = bitcast<f64>(u);

    if e >= 0x3ff + 26 {

        x = runmat_math64_log(x) + runmat_math64_asinh_LN2;
    } else if e >= 0x3ff + 1 {

        x = runmat_math64_log(2.0 * x + 1.0 / (sqrt(x * x + 1.0) + x));
    } else if e >= 0x3ff - 26 {

        x = runmat_math64_log1p(x + x * x / (sqrt(x * x + 1.0) + 1.0));
    } else {

        var x1p120 = runmat_f64_from_words(0x0u, 0x47700000u);

    }

    return select(x, -x, sign);
}

const runmat_math64_acosh_LN2: f64 = 0.693147180559945309417232121458176568;

fn runmat_math64_acosh(x: f64) -> f64 {
    // Admit the real domain by encoding before cancellation-prone region
    // formulas. Negative inputs cannot delegate domain rejection to log:
    // x + sqrt(x*x-1) can cancel and produce finite/zero intermediates.
    let high = runmat_f64_high(x);
    let low = runmat_f64_low(x);
    let magnitude = high & 0x7fffffffu;
    if magnitude > 0x7ff00000u || (magnitude == 0x7ff00000u && low != 0u) {
        return x;
    }
    if (high & 0x80000000u) != 0u || magnitude < 0x3ff00000u {
        return runmat_f64_from_words(0u, 0x7ff80000u);
    }
    if magnitude == 0x3ff00000u && low == 0u { return f64(0.0); }
    if magnitude == 0x7ff00000u { return x; }
    var u = bitcast<u64>(x);
    var e = (u32((u >> 52))) & 0x7ff;



    if e < 0x3ff + 1 {

        return runmat_math64_log1p(x - 1.0 + sqrt((x - 1.0) * (x - 1.0) + 2.0 * (x - 1.0)));
    }
    if e < 0x3ff + 26 {

        return runmat_math64_log(2.0 * x - 1.0 / (x + sqrt(x * x - 1.0)));
    }

    return runmat_math64_log(x) + runmat_math64_acosh_LN2;
}

fn runmat_math64_atanh(x: f64) -> f64 {
    if (runmat_f64_high(x) & 0x7fffffffu) < 0x3df00000u { return x; }
    var u = bitcast<u64>(x);
    var e = (u32((u >> 52))) & 0x7ff;
    var sign = (u >> 63) != 0;


    var y = bitcast<f64>(u & ((u64(0x7fffffffu) << 32u) | u64(0xffffffffu)));

    if e < 0x3ff - 1 {
        if e < 0x3ff - 32 {

            if e == 0 {

            }
        } else {

            y = 0.5 * runmat_math64_log1p(2.0 * y + 2.0 * y * y / (1.0 - y));
        }
    } else {

        y = 0.5 * runmat_math64_log1p(2.0 * (y / (1.0 - y)));
    }

    return select(y, -y, sign);
}

// libm expo2.rs / k_expo2.rs: split scaling avoids premature exp overflow.
fn runmat_math64_hyperbolic_expo2(x: f64) -> f64 {
    let k_ln2 = runmat_f64_from_words(0x151add8bu, 0x40962066u);
    let scale = runmat_f64_from_words(0u, 0x7fc00000u);
    return runmat_math64_exp(x - k_ln2) * scale * scale;
}
