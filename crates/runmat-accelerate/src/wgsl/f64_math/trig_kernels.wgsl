// Translated from rust-lang/libm 0.2.16 k_sin.rs, k_cos.rs, k_tan.rs.
// Origins: FreeBSD /usr/src/lib/msun/src/k_sin.c, k_cos.c, k_tan.c.
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
// k_tan.c notice:
// ====================================================
// Copyright 2004 Sun Microsystems, Inc.  All Rights Reserved.
//
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
// See UPSTREAM_TRIG.md for source identity and MIT permission.
// Arithmetic grouping is intentional; requires non-contracted binary64 ops.

fn runmat_math64_k_sin(x: f64, y: f64, iy: i32) -> f64 {
    const s1: f64 = -1.66666666666666324348e-01;
    const s2: f64 = 8.33333333332248946124e-03;
    const s3: f64 = -1.98412698298579493134e-04;
    const s4: f64 = 2.75573137070700676789e-06;
    const s5: f64 = -2.50507602534068634195e-08;
    const s6: f64 = 1.58969099521155010221e-10;
    let z = x * x;
    let w = z * z;
    let r = s2 + z * (s3 + z * s4) + z * w * (s5 + z * s6);
    let v = z * x;
    if (iy == 0) { return x + v * (s1 + z * r); }
    return x - ((z * (f64(0.5) * y - v * r) - y) - v * s1);
}

fn runmat_math64_k_cos(x: f64, y: f64) -> f64 {
    const c1: f64 = 4.16666666666666019037e-02;
    const c2: f64 = -1.38888888888741095749e-03;
    const c3: f64 = 2.48015872894767294178e-05;
    const c4: f64 = -2.75573143513906633035e-07;
    const c5: f64 = 2.08757232129817482790e-09;
    const c6: f64 = -1.13596475577881948265e-11;
    let z = x * x;
    let w = z * z;
    let r = z * (c1 + z * (c2 + z * c3)) + w * w * (c4 + z * (c5 + z * c6));
    let hz = f64(0.5) * z;
    let head = f64(1.0) - hz;
    return head + (((f64(1.0) - head) - hz) + (z * r - x * y));
}

fn runmat_math64_k_tan(argument: f64, tail: f64, odd: i32) -> f64 {
    const t: array<f64, 13> = array<f64, 13>(
        3.33333333333334091986e-01, 1.33333333333201242699e-01,
        5.39682539762260521377e-02, 2.18694882948595424599e-02,
        8.86323982359930005737e-03, 3.59207910759131235356e-03,
        1.45620945432529025516e-03, 5.88041240820264096874e-04,
        2.46463134818469906812e-04, 7.81794442939557092300e-05,
        7.14072491382608190305e-05, -1.85586374855275456654e-05,
        2.59073051863633712884e-05);
    const pio4: f64 = 7.85398163397448278999e-01;
    const pio4_lo: f64 = 3.06161699786838301793e-17;
    var x = argument;
    var y = tail;
    let hx = runmat_f64_high(x);
    let big = (hx & 0x7fffffffu) >= 0x3fe59428u;
    if (big) {
        if ((hx >> 31u) != 0u) { x = -x; y = -y; }
        x = (pio4 - x) + (pio4_lo - y);
        y = f64(0.0);
    }
    let z = x * x;
    let w = z * z;
    let r0 = t[1] + w * (t[3] + w * (t[5] + w * (t[7] + w * (t[9] + w * t[11]))));
    let v0 = z * (t[2] + w * (t[4] + w * (t[6] + w * (t[8] + w * (t[10] + w * t[12])))));
    let s0 = z * x;
    let r = y + z * (s0 * (r0 + v0) + y) + s0 * t[0];
    let result = x + r;
    if (big) {
        let s = f64(1.0) - f64(2.0) * f64(odd);
        let v = s - f64(2.0) * (x + (r - result * result / (result + s)));
        if ((hx >> 31u) != 0u) { return -v; }
        return v;
    }
    if (odd == 0) { return result; }
    // Compensated reciprocal: a simple -1/result can be two ULP away.
    let w0 = runmat_f64_from_words(0u, runmat_f64_high(result));
    let v = r - (w0 - x);
    let a = f64(-1.0) / result;
    let a0 = runmat_f64_from_words(0u, runmat_f64_high(a));
    return a0 + a * (f64(1.0) + a0 * w0 + a0 * v);
}
