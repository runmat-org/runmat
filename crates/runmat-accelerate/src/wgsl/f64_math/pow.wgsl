// Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// Translated from pinned rust-lang/libm 0.2.16 pow.rs; see UPSTREAM_ELEMENTARY.md.
// Split logarithm/multiprecision product/polynomial preserved; NOT exp(y*log(x)).
// Existing RunMat policy wrappers own NaN-first power and complex promotion.

// Binary64 value normalization by encoding, followed by normal arithmetic.
// These special powers must not feed subnormals into native mul/div.
fn runmat_math64_pow_square(x: f64) -> f64 {
    let high = runmat_f64_high(x) & 0x7fffffffu;
    let low = runmat_f64_low(x);
    if high >= 0x7ff00000u {
        if high > 0x7ff00000u || low != 0u { return x; }
        return runmat_f64_from_words(0u, 0x7ff00000u);
    }
    if (high | low) == 0u { return f64(0.0); }
    var value = runmat_f64_from_words(low, high);
    var exponent = i32(high >> 20u) - 1023;
    if high < 0x00100000u {
        value = runmat_math64_scalbn(value, 54);
        exponent = i32(runmat_f64_high(value) >> 20u) - 1023 - 54;
    }
    let mantissa = runmat_f64_from_words(runmat_f64_low(value), (runmat_f64_high(value) & 0x000fffffu) | 0x3ff00000u);
    return runmat_math64_scalbn(mantissa * mantissa, 2 * exponent);
}
fn runmat_math64_pow_reciprocal(x: f64) -> f64 {
    let original_high = runmat_f64_high(x);
    let high = original_high & 0x7fffffffu;
    let sign = original_high & 0x80000000u;
    let low = runmat_f64_low(x);
    if high >= 0x7ff00000u {
        if high > 0x7ff00000u || low != 0u { return x; }
        return runmat_f64_from_words(0u, sign);
    }
    if (high | low) == 0u { return runmat_f64_from_words(0u, sign | 0x7ff00000u); }
    var value = runmat_f64_from_words(low, high);
    var exponent = i32(high >> 20u) - 1023;
    if high < 0x00100000u {
        value = runmat_math64_scalbn(value, 54);
        exponent = i32(runmat_f64_high(value) >> 20u) - 1023 - 54;
    }
    let mantissa = runmat_f64_from_words(runmat_f64_low(value), (runmat_f64_high(value) & 0x000fffffu) | 0x3ff00000u);
    let result = runmat_math64_scalbn(f64(1.0) / mantissa, -exponent);
    return runmat_f64_from_words(runmat_f64_low(result), runmat_f64_high(result) | sign);
}


fn runmat_math64_pow_set_low(value: f64, low: u32) -> f64 {
    return runmat_f64_from_words(low, runmat_f64_high(value));
}
fn runmat_math64_pow_set_high(value: f64, high: u32) -> f64 {
    return runmat_f64_from_words(runmat_f64_low(value), high);
}
fn runmat_math64_pow_signed_inf(sign: f64) -> f64 {
    return runmat_f64_from_words(0u, 0x7ff00000u | (runmat_f64_high(sign) & 0x80000000u));
}
fn runmat_math64_pow_signed_zero(sign: f64) -> f64 {
    return runmat_f64_from_words(0u, runmat_f64_high(sign) & 0x80000000u);
}
const runmat_math64_pow_BP: array<f64, 2> = array<f64, 2>(1.0, 1.5);
const runmat_math64_pow_DP_H: array<f64, 2> = array<f64, 2>(0.0, 5.84962487220764160156e-01);
const runmat_math64_pow_DP_L: array<f64, 2> = array<f64, 2>(0.0, 1.35003920212974897128e-08);
const runmat_math64_pow_TWO53: f64 = 9007199254740992.0;
const runmat_math64_pow_HUGE: f64 = 1.0e300;
const runmat_math64_pow_TINY: f64 = 1.0e-300;

const runmat_math64_pow_L1: f64 = 5.99999999999994648725e-01;
const runmat_math64_pow_L2: f64 = 4.28571428578550184252e-01;
const runmat_math64_pow_L3: f64 = 3.33333329818377432918e-01;
const runmat_math64_pow_L4: f64 = 2.72728123808534006489e-01;
const runmat_math64_pow_L5: f64 = 2.30660745775561754067e-01;
const runmat_math64_pow_L6: f64 = 2.06975017800338417784e-01;
const runmat_math64_pow_P1: f64 = 1.66666666666666019037e-01;
const runmat_math64_pow_P2: f64 = -2.77777777770155933842e-03;
const runmat_math64_pow_P3: f64 = 6.61375632143793436117e-05;
const runmat_math64_pow_P4: f64 = -1.65339022054652515390e-06;
const runmat_math64_pow_P5: f64 = 4.13813679705723846039e-08;
const runmat_math64_pow_LG2: f64 = 6.93147180559945286227e-01;
const runmat_math64_pow_LG2_H: f64 = 6.93147182464599609375e-01;
const runmat_math64_pow_LG2_L: f64 = -1.90465429995776804525e-09;
const runmat_math64_pow_OVT: f64 = 8.0085662595372944372e-017;
const runmat_math64_pow_CP: f64 = 9.61796693925975554329e-01;
const runmat_math64_pow_CP_H: f64 = 9.61796700954437255859e-01;
const runmat_math64_pow_CP_L: f64 = -7.02846165095275826516e-09;
const runmat_math64_pow_IVLN2: f64 = 1.44269504088896338700e+00;
const runmat_math64_pow_IVLN2_H: f64 = 1.44269502162933349609e+00;
const runmat_math64_pow_IVLN2_L: f64 = 1.92596299112661746887e-08;

fn runmat_math64_pow(x: f64, y: f64) -> f64 {
    var t1: f64;
    var t2: f64;

    var hx: i32 = bitcast<i32>(runmat_f64_high(x)); var lx: u32 = runmat_f64_low(x);
    var hy: i32 = bitcast<i32>(runmat_f64_high(y)); var ly: u32 = runmat_f64_low(y);

    var ix: i32 = hx & 0x7fffffff;
    var iy: i32 = hy & 0x7fffffff;


    if ((u32(iy)) | ly) == 0 {
        return f64(1.0);
    }


    if hx == 0x3ff00000 && lx == 0 {
        return f64(1.0);
    }


    if ix > 0x7ff00000
        || (ix == 0x7ff00000 && lx != 0)
        || iy > 0x7ff00000
        || (iy == 0x7ff00000 && ly != 0)
    {
        return x + y;
    }


    var yisint: i32 = 0;
    var k: i32;
    var j: i32;
    if hx < 0 {
        if iy >= 0x43400000 {
            yisint = 2;
        } else if iy >= 0x3ff00000 {
            k = (iy >> 20) - 0x3ff;

            if k > 20 {
                j = i32(ly >> u32(52 - k));

                if (j << u32(52 - k)) == (bitcast<i32>(ly)) {
                    yisint = 2 - (j & 1);
                }
            } else if ly == 0 {
                j = iy >> u32(20 - k);

                if (j << u32(20 - k)) == iy {
                    yisint = 2 - (j & 1);
                }
            }
        }
    }

    if ly == 0 {

        if iy == 0x7ff00000 {



            if ((ix - 0x3ff00000) | (bitcast<i32>(lx))) == 0 { return f64(1.0); }
            if ix >= 0x3ff00000 { if hy >= 0 { return y; } return f64(0.0); }
            if hy >= 0 { return f64(0.0); } return -y;
        }

        if iy == 0x3ff00000 {

            if hy >= 0 { return x; } return runmat_math64_pow_reciprocal(x);
        }

        if hy == 0x40000000 {

            return runmat_math64_pow_square(x);
        }

        if hy == 0x3fe00000 {

            if hx >= 0 {

                if ix < 0x00100000 && (u32(ix) | lx) != 0u {
                    return runmat_math64_scalbn(sqrt(runmat_math64_scalbn(x, 54)), -27);
                }
                return sqrt(x);
            }
        }
    }

    var ax: f64 = runmat_f64_from_words(lx, u32(ix));
    if lx == 0 {

        if ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000 {

            var z: f64 = ax;

            if hy < 0 {

                z = 1.0 / z;
            }

            if hx < 0 {
                if ((ix - 0x3ff00000) | yisint) == 0 {
                    z = runmat_f64_from_words(0u, 0x7ff80000u);
                } else if yisint == 1 {
                    z = -z;
                }
            }

            return z;
        }
    }

    var s: f64 = 1.0;
    if hx < 0 {
        if yisint == 0 {

            return runmat_f64_from_words(0u, 0x7ff80000u);
        }

        if yisint == 1 {

            s = -1.0;
        }
    }

    // Even max |log(x)| times this y is below half an ulp at one.
    // Avoid consuming a subnormal exponent in native floating arithmetic.
    if iy < 0x3be00000 { return f64(1.0); }


    if iy > 0x41e00000 {

        if iy > 0x43f00000 {

            if ix <= 0x3fefffff {
                if hy < 0 { return runmat_f64_from_words(0u, 0x7ff00000u); } return f64(0.0);
            }

            if ix >= 0x3ff00000 {
                if hy > 0 { return runmat_f64_from_words(0u, 0x7ff00000u); } return f64(0.0);
            }
        }


        if ix < 0x3fefffff {
            if hy < 0 { return runmat_math64_pow_signed_inf(s); } return runmat_math64_pow_signed_zero(s);
        }
        if ix > 0x3ff00000 {
            if hy > 0 { return runmat_math64_pow_signed_inf(s); } return runmat_math64_pow_signed_zero(s);
        }


        var t: f64 = ax - 1.0;
        var w: f64 = (t * t) * (0.5 - t * (0.3333333333333333333333 - t * 0.25));
        var u: f64 = runmat_math64_pow_IVLN2_H * t;
        var v: f64 = t * runmat_math64_pow_IVLN2_L - w * runmat_math64_pow_IVLN2;
        t1 = runmat_math64_pow_set_low(u + v, 0);
        t2 = v - (t1 - u);
    } else {

        var n: i32 = 0;

        if ix < 0x00100000 {

            ax = runmat_math64_scalbn(ax, 53);
            n -= 53;
            ix = bitcast<i32>(runmat_f64_high(ax));
        }

        n += (ix >> 20) - 0x3ff;
        j = ix & 0x000fffff;


        var k: i32;
        ix = j | 0x3ff00000;
        if j <= 0x3988E {

            k = 0;
        } else if j < 0xBB67A {

            k = 1;
        } else {
            k = 0;
            n += 1;
            ix -= 0x00100000;
        }
        ax = runmat_math64_pow_set_high(ax, bitcast<u32>(ix));


        var u: f64 = ax - runmat_math64_pow_BP[u32(k)];
        var v: f64 = 1.0 / (ax + runmat_math64_pow_BP[u32(k)]);
        var ss: f64 = u * v;
        var s_h = runmat_math64_pow_set_low(ss, 0);


        var t_h: f64 = runmat_math64_pow_set_high(
            f64(0.0),
            ((bitcast<u32>(ix) >> 1) | 0x20000000) + 0x00080000 + ((u32(k)) << 18),
        );
        var t_l: f64 = ax - (t_h - runmat_math64_pow_BP[u32(k)]);
        var s_l: f64 = v * ((u - s_h * t_h) - s_h * t_l);


        var s2: f64 = ss * ss;
        var r: f64 = s2 * s2 * (runmat_math64_pow_L1 + s2 * (runmat_math64_pow_L2 + s2 * (runmat_math64_pow_L3 + s2 * (runmat_math64_pow_L4 + s2 * (runmat_math64_pow_L5 + s2 * runmat_math64_pow_L6)))));
        r += s_l * (s_h + ss);
        var s2_hi: f64 = s_h * s_h;
        var t_h_sum: f64 = runmat_math64_pow_set_low(3.0 + s2_hi + r, 0);
        var t_l_sum: f64 = r - ((t_h_sum - 3.0) - s2_hi);


        var u_product: f64 = s_h * t_h_sum;
        var v_product: f64 = s_l * t_h_sum + t_l_sum * ss;


        var p_h: f64 = runmat_math64_pow_set_low(u_product + v_product, 0);
        var p_l = v_product - (p_h - u_product);
        var z_h: f64 = runmat_math64_pow_CP_H * p_h;
        var z_l: f64 = runmat_math64_pow_CP_L * p_h + p_l * runmat_math64_pow_CP + runmat_math64_pow_DP_L[u32(k)];


        var t: f64 = f64(n);
        t1 = runmat_math64_pow_set_low(((z_h + z_l) + runmat_math64_pow_DP_H[u32(k)]) + t, 0);
        t2 = z_l - (((t1 - t) - runmat_math64_pow_DP_H[u32(k)]) - z_h);
    }


    var y1: f64 = runmat_math64_pow_set_low(y, 0);
    var p_l: f64 = (y - y1) * t1 + y * t2;
    var p_h: f64 = y1 * t1;
    var z: f64 = p_l + p_h;
    j = bitcast<i32>(runmat_f64_high(z));
    var fraction_i: i32 = bitcast<i32>(runmat_f64_low(z));


    if j >= 0x40900000 {

        if ((j - 0x40900000) | fraction_i) != 0 {

            return runmat_math64_pow_signed_inf(s);
        }

        if p_l + runmat_math64_pow_OVT > z - p_h {
            return runmat_math64_pow_signed_inf(s);
        }
    } else if (j & 0x7fffffff) >= 0x4090cc00 {



        if (((bitcast<u32>(j)) - 0xc090cc00) | (bitcast<u32>(fraction_i))) != 0 {

            return runmat_math64_pow_signed_zero(s);
        }

        if p_l <= z - p_h {
            return runmat_math64_pow_signed_zero(s);
        }
    }


    var i: i32 = j & 0x7fffffff;
    k = (i >> 20) - 0x3ff;
    var n: i32 = 0;

    if i > 0x3fe00000 {

        n = j + (0x00100000i >> u32(k + 1));
        k = ((n & 0x7fffffff) >> 20) - 0x3ff;
        var t: f64 = runmat_math64_pow_set_high(0.0, u32((n & ~(0x000fffffi >> u32(k)))));
        n = ((n & 0x000fffff) | 0x00100000) >> u32(20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }

    var t: f64 = runmat_math64_pow_set_low(p_l + p_h, 0);
    var u: f64 = t * runmat_math64_pow_LG2_H;
    var v: f64 = (p_l - (t - p_h)) * runmat_math64_pow_LG2 + t * runmat_math64_pow_LG2_L;
    z = u + v;
    var w: f64 = v - (z - u);
    var square: f64 = z * z;
    var kernel_t1: f64 = z - square * (runmat_math64_pow_P1 + square * (runmat_math64_pow_P2 + square * (runmat_math64_pow_P3 + square * (runmat_math64_pow_P4 + square * runmat_math64_pow_P5))));
    var r: f64 = (z * kernel_t1) / (kernel_t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    j = bitcast<i32>(runmat_f64_high(z));
    j += n << 20;

    if (j >> 20) <= 0 {

        z = runmat_math64_scalbn(z, n);
    } else {
        z = runmat_math64_pow_set_high(z, bitcast<u32>(j));
    }

    return runmat_f64_from_words(runmat_f64_low(z), runmat_f64_high(z) | (runmat_f64_high(s) & 0x80000000u));
}
