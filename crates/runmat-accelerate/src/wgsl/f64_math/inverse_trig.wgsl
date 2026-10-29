// Translated from rust-lang/libm 0.2.16 (pinned registry source).
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// See UPSTREAM_ELEMENTARY.md for source hashes, MIT permission and changes.
// Requires controlled binary64 evaluation; floating exception flags omitted.

fn runmat_math64_inverse_set_low(value: f64, low: u32) -> f64 {
    return runmat_f64_from_words(low, runmat_f64_high(value));
}
const runmat_math64_asin_PIO2_HI: f64 = 1.57079632679489655800e+00;
const runmat_math64_asin_PIO2_LO: f64 = 6.12323399573676603587e-17;

const runmat_math64_asin_P_S0: f64 = 1.66666666666666657415e-01;
const runmat_math64_asin_P_S1: f64 = -3.25565818622400915405e-01;
const runmat_math64_asin_P_S2: f64 = 2.01212532134862925881e-01;
const runmat_math64_asin_P_S3: f64 = -4.00555345006794114027e-02;
const runmat_math64_asin_P_S4: f64 = 7.91534994289814532176e-04;
const runmat_math64_asin_P_S5: f64 = 3.47933107596021167570e-05;
const runmat_math64_asin_Q_S1: f64 = -2.40339491173441421878e+00;
const runmat_math64_asin_Q_S2: f64 = 2.02094576023350569471e+00;
const runmat_math64_asin_Q_S3: f64 = -6.88283971605453293030e-01;
const runmat_math64_asin_Q_S4: f64 = 7.70381505559019352791e-02;

fn runmat_math64_asin_r(z: f64) -> f64 {
    var p = z * (runmat_math64_asin_P_S0 + z * (runmat_math64_asin_P_S1 + z * (runmat_math64_asin_P_S2 + z * (runmat_math64_asin_P_S3 + z * (runmat_math64_asin_P_S4 + z * runmat_math64_asin_P_S5)))));
    var q = 1.0 + z * (runmat_math64_asin_Q_S1 + z * (runmat_math64_asin_Q_S2 + z * (runmat_math64_asin_Q_S3 + z * runmat_math64_asin_Q_S4)));
    return p / q;
}

fn runmat_math64_asin(argument_x: f64) -> f64 {
var x = argument_x;
    var pio2_hi = runmat_math64_asin_PIO2_HI;
    var pio2_lo = runmat_math64_asin_PIO2_LO;
    var z: f64;
    var r: f64;
    var s: f64;
    var hx: u32;
    var ix: u32;

    hx = runmat_f64_high(x);
    ix = hx & 0x7fffffff;

    if ix >= 0x3ff00000 {
        var lx: u32;
        lx = runmat_f64_low(x);
        if ((ix - 0x3ff00000) | lx) == 0 {

            return x * pio2_hi + runmat_f64_from_words(0x0u, 0x38700000u);
        } else {
            return runmat_f64_from_words(0u, 0x7ff80000u);
        }
    }

    if ix < 0x3fe00000 {

        if ix < 0x3e500000u {
            return x;
        } else {
            return x + x * runmat_math64_asin_r(x * x);
        }
    }

    z = (1.0 - abs(x)) * 0.5;
    s = sqrt(z);
    r = runmat_math64_asin_r(z);
    if ix >= 0x3fef3333 {

        x = pio2_hi - (2. * (s + s * r) - pio2_lo);
    } else {
        var f: f64;
        var c: f64;

        f = runmat_math64_inverse_set_low(s, 0);
        c = (z - f * f) / (s + f);
        x = 0.5 * pio2_hi - (2.0 * s * r - (pio2_lo - 2.0 * c) - (0.5 * pio2_hi - 2.0 * f));
    }
    return select(x, -x, hx >> 31u != 0u);
}
const runmat_math64_acos_PIO2_HI: f64 = 1.57079632679489655800e+00;
const runmat_math64_acos_PIO2_LO: f64 = 6.12323399573676603587e-17;
const runmat_math64_acos_PS0: f64 = 1.66666666666666657415e-01;
const runmat_math64_acos_PS1: f64 = -3.25565818622400915405e-01;
const runmat_math64_acos_PS2: f64 = 2.01212532134862925881e-01;
const runmat_math64_acos_PS3: f64 = -4.00555345006794114027e-02;
const runmat_math64_acos_PS4: f64 = 7.91534994289814532176e-04;
const runmat_math64_acos_PS5: f64 = 3.47933107596021167570e-05;
const runmat_math64_acos_QS1: f64 = -2.40339491173441421878e+00;
const runmat_math64_acos_QS2: f64 = 2.02094576023350569471e+00;
const runmat_math64_acos_QS3: f64 = -6.88283971605453293030e-01;
const runmat_math64_acos_QS4: f64 = 7.70381505559019352791e-02;

fn runmat_math64_acos_r(z: f64) -> f64 {
    var p: f64 = z * (runmat_math64_acos_PS0 + z * (runmat_math64_acos_PS1 + z * (runmat_math64_acos_PS2 + z * (runmat_math64_acos_PS3 + z * (runmat_math64_acos_PS4 + z * runmat_math64_acos_PS5)))));
    var q: f64 = 1.0 + z * (runmat_math64_acos_QS1 + z * (runmat_math64_acos_QS2 + z * (runmat_math64_acos_QS3 + z * runmat_math64_acos_QS4)));
    return p / q;
}

fn runmat_math64_acos(x: f64) -> f64 {
    var pio2_hi = runmat_math64_acos_PIO2_HI;
    var pio2_lo = runmat_math64_acos_PIO2_LO;
    var x1p_120f = runmat_f64_from_words(0x0u, 0x38700000u);
    var z: f64;
    var w: f64;
    var s: f64;
    var c: f64;
    var df: f64;
    var hx: u32;
    var ix: u32;

    hx = u32((bitcast<u64>(x) >> 32));
    ix = hx & 0x7fffffff;

    if ix >= 0x3ff00000 {
        var lx: u32 = u32(bitcast<u64>(x));

        if ((ix - 0x3ff00000) | lx) == 0 {

            if (hx >> 31) != 0 {
                return 2. * pio2_hi + x1p_120f;
            }
            return f64(0.0);
        }
        return runmat_f64_from_words(0u, 0x7ff80000u);
    }

    if ix < 0x3fe00000 {
        if ix <= 0x3c600000 {

            return pio2_hi + x1p_120f;
        }
        return pio2_hi - (x - (pio2_lo - x * runmat_math64_acos_r(x * x)));
    }

    if (hx >> 31) != 0 {
        z = (1.0 + x) * 0.5;
        s = sqrt(z);
        w = runmat_math64_acos_r(z) * s - pio2_lo;
        return 2. * (pio2_hi - (s + w));
    }

    z = (1.0 - x) * 0.5;
    s = sqrt(z);

    df = runmat_f64_from_words(0u, runmat_f64_high(s));

    c = (z - df * df) / (s + df);
    w = runmat_math64_acos_r(z) * s + c;
    return 2. * (df + w);
}
const runmat_math64_atan_ATANHI: array<f64, 4> = array<f64, 4>(
    4.63647609000806093515e-01,
    7.85398163397448278999e-01,
    9.82793723247329054082e-01,
    1.57079632679489655800e+00,
);

const runmat_math64_atan_ATANLO: array<f64, 4> = array<f64, 4>(
    2.26987774529616870924e-17,
    3.06161699786838301793e-17,
    1.39033110312309984516e-17,
    6.12323399573676603587e-17,
);

const runmat_math64_atan_AT: array<f64, 11> = array<f64, 11>(
    3.33333333333329318027e-01,
    -1.99999999998764832476e-01,
    1.42857142725034663711e-01,
    -1.11111104054623557880e-01,
    9.09088713343650656196e-02,
    -7.69187620504482999495e-02,
    6.66107313738753120669e-02,
    -5.83357013379057348645e-02,
    4.97687799461593236017e-02,
    -3.65315727442169155270e-02,
    1.62858201153657823623e-02,
);

fn runmat_math64_atan(argument: f64) -> f64 {
    var x = argument;
    let high = runmat_f64_high(x);
    let sign = high >> 31u;
    let ix = high & 0x7fffffffu;
    if ix >= 0x44100000u {
        if ix > 0x7ff00000u || (ix == 0x7ff00000u && runmat_f64_low(x) != 0u) { return x; }
        var z = runmat_math64_atan_ATANHI[3];
        return select(z, -z, sign != 0u);
    }
    var id = -1;
    if ix < 0x3fdc0000u {
        if ix < 0x3e400000u { return x; }
    } else {
        x = abs(x);
        if ix < 0x3ff30000u {
            if ix < 0x3fe60000u {
                x = (f64(2.0) * x - f64(1.0)) / (f64(2.0) + x);
                id = 0;
            } else {
                x = (x - f64(1.0)) / (x + f64(1.0));
                id = 1;
            }
        } else if ix < 0x40038000u {
            x = (x - f64(1.5)) / (f64(1.0) + f64(1.5) * x);
            id = 2;
        } else {
            x = f64(-1.0) / x;
            id = 3;
        }
    }
    let z = x * x;
    let w = z * z;
    let s1 = z * (runmat_math64_atan_AT[0] + w * (runmat_math64_atan_AT[2] + w * (runmat_math64_atan_AT[4] + w * (runmat_math64_atan_AT[6] + w * (runmat_math64_atan_AT[8] + w * runmat_math64_atan_AT[10])))));
    let s2 = w * (runmat_math64_atan_AT[1] + w * (runmat_math64_atan_AT[3] + w * (runmat_math64_atan_AT[5] + w * (runmat_math64_atan_AT[7] + w * runmat_math64_atan_AT[9]))));
    if id < 0 { return x - x * (s1 + s2); }
    let result = runmat_math64_atan_ATANHI[u32(id)] - (x * (s1 + s2) - runmat_math64_atan_ATANLO[u32(id)] - x);
    return select(result, -result, sign != 0u);
}

fn runmat_math64_atan2(y: f64, x: f64) -> f64 {
    var pi: f64 = 3.1415926535897931160;
    var pi_lo: f64 = 1.2246467991473531772e-16;
    let hx = runmat_f64_high(x);
    let hy = runmat_f64_high(y);
    let lx = runmat_f64_low(x);
    let ly = runmat_f64_low(y);
    let ix = hx & 0x7fffffffu;
    let iy = hy & 0x7fffffffu;
    if ix > 0x7ff00000u || (ix == 0x7ff00000u && lx != 0u) { return x; }
    if iy > 0x7ff00000u || (iy == 0x7ff00000u && ly != 0u) { return y; }
    if hx == 0x3ff00000u && lx == 0u { return runmat_math64_atan(y); }
    let m = ((hy >> 31u) & 1u) | ((hx >> 30u) & 2u);
    if (iy | ly) == 0u {
        if m < 2u { return y; }
        return select(pi, -pi, m == 3u);
    }
    if (ix | lx) == 0u { return select(pi / f64(2.0), -pi / f64(2.0), (m & 1u) != 0u); }
    if ix == 0x7ff00000u {
        if iy == 0x7ff00000u {
            switch m {
                case 0u: { return pi / f64(4.0); }
                case 1u: { return -pi / f64(4.0); }
                case 2u: { return f64(3.0) * pi / f64(4.0); }
                default: { return f64(-3.0) * pi / f64(4.0); }
            }
        }
        switch m {
            case 0u: { return f64(0.0); }
            case 1u: { return runmat_f64_from_words(0u, 0x80000000u); }
            case 2u: { return pi; }
            default: { return -pi; }
        }
    }
    if ix + (64u << 20u) < iy || iy == 0x7ff00000u {
        return select(pi / f64(2.0), -pi / f64(2.0), (m & 1u) != 0u);
    }
    var z = f64(0.0);
    if !((m & 2u) != 0u && iy + (64u << 20u) < ix) {
        // Normalize both operands by encoding so a subnormal input or ratio
        // never enters floating arithmetic. The quotient mantissa is normal.
        var ax = runmat_f64_from_words(lx, ix);
        var ay = runmat_f64_from_words(ly, iy);
        var ex = i32(ix >> 20u) - 1023;
        var ey = i32(iy >> 20u) - 1023;
        if ix < 0x00100000u {
            ax = runmat_math64_scalbn(ax, 54);
            ex = i32(runmat_f64_high(ax) >> 20u) - 1023 - 54;
        }
        if iy < 0x00100000u {
            ay = runmat_math64_scalbn(ay, 54);
            ey = i32(runmat_f64_high(ay) >> 20u) - 1023 - 54;
        }
        ax = runmat_f64_from_words(runmat_f64_low(ax), (runmat_f64_high(ax) & 0x000fffffu) | 0x3ff00000u);
        ay = runmat_f64_from_words(runmat_f64_low(ay), (runmat_f64_high(ay) & 0x000fffffu) | 0x3ff00000u);
        let mantissa = ay / ax;
        let difference = ey - ex;
        if difference < -27 {
            if (m & 2u) == 0u {
                let ratio = runmat_math64_scalbn(mantissa, difference);
                return runmat_f64_from_words(runmat_f64_low(ratio), runmat_f64_high(ratio) | (hy & 0x80000000u));
            }
            // For negative x the small angle still matters near pi until
            // it is below the original libm 2^-64 ratio cutoff.
            if difference >= -64 { z = runmat_math64_scalbn(mantissa, difference); }
        } else {
            z = runmat_math64_atan(runmat_math64_scalbn(mantissa, difference));
        }
    }
    switch m {
        case 0u: { return z; }
        case 1u: { return -z; }
        case 2u: { return pi - (z - pi_lo); }
        default: { return (z - pi_lo) - pi; }
    }
}
