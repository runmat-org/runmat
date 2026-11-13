// Translated from rust-lang/libm 0.2.16 rem_pio2.rs, rem_pio2_large.rs.
// Origins: FreeBSD /usr/src/lib/msun/src/e_rem_pio2.c, k_rem_pio2.c.
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
// k_rem_pio2.c notice:
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
// Optimized by Bruce D. Evans. See UPSTREAM_TRIG.md for source identity.
// Specialization of the upstream reducer to binary64 (prec=1, jk=jp=4).
// This retains the base-2^24 Payne–Hanek convolution, cancellation-driven
// recomputation and compensated two-component remainder. It is NOT modulo
// a rounded pi constant. All finite binary64 exponents fit the 66-word table.

struct RunmatMath64Reduction {
    n: i32,
    hi: f64,
    lo: f64,
}

fn runmat_math64_rem_pio2_large(x: array<f64, 3>, nx: i32, e0: i32) -> RunmatMath64Reduction {
    var ipio2 = array<i32, 66>(
        0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62,
        0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A,
        0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
        0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
        0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8,
        0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
        0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
        0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
        0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
        0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880,
        0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B);
    var pio2 = array<f64, 8>(
        1.57079625129699707031e+00, 7.54978941586159635335e-08,
        5.39030252995776476554e-15, 3.28200341580791294123e-22,
        1.27065575308067607349e-29, 1.22933308981111328932e-36,
        2.73370053816464559624e-44, 2.16741683877804819444e-51);
    const two24: f64 = 16777216.0;
    const twon24: f64 = 5.9604644775390625e-8;
    // Local addressable input permits dynamic indexing in all Naga backends.
    var tx = x;
    var f: array<f64, 20>;
    var fq: array<f64, 20>;
    var q: array<f64, 20>;
    var iq: array<i32, 20>;
    let jk = 4;
    let jx = nx - 1;
    let jv = max(0, (e0 - 3) / 24);
    var q0 = e0 - 24 * (jv + 1);
    var j = jv - jx;
    for (var i = 0; i <= jx + jk; i = i + 1) {
        if (j < 0) { f[i] = f64(0.0); }
        else { f[i] = f64(ipio2[j]); }
        j = j + 1;
    }
    for (var i = 0; i <= jk; i = i + 1) {
        var fw = f64(0.0);
        for (var k = 0; k <= jx; k = k + 1) {
            fw = fw + tx[k] * f[jx + i - k];
        }
        q[i] = fw;
    }
    var jz = jk;
    var z = f64(0.0);
    var n = 0;
    var ih = 0;
    loop {
        // Distill integral convolution chunks in reverse order.
        var i = 0;
        z = q[jz];
        for (var k = jz; k >= 1; k = k - 1) {
            let fw = f64(i32(twon24 * z));
            iq[i] = i32(z - two24 * fw);
            z = q[k - 1] + fw;
            i = i + 1;
        }
        z = runmat_math64_scalbn(z, q0);
        z = z - f64(8.0) * floor(z * f64(0.125));
        n = i32(z);
        z = z - f64(n);
        ih = 0;
        if (q0 > 0) {
            i = iq[jz - 1] >> u32(24 - q0);
            n = n + i;
            iq[jz - 1] = iq[jz - 1] - (i << u32(24 - q0));
            ih = iq[jz - 1] >> u32(23 - q0);
        } else if (q0 == 0) {
            ih = iq[jz - 1] >> 23u;
        } else if (z >= f64(0.5)) {
            ih = 2;
        }
        if (ih > 0) {
            n = n + 1;
            var carry = 0;
            for (var k = 0; k < jz; k = k + 1) {
                let digit = iq[k];
                if (carry == 0) {
                    if (digit != 0) { carry = 1; iq[k] = 0x1000000 - digit; }
                } else { iq[k] = 0xffffff - digit; }
            }
            if (q0 == 1) { iq[jz - 1] = iq[jz - 1] & 0x7fffff; }
            if (q0 == 2) { iq[jz - 1] = iq[jz - 1] & 0x3fffff; }
            if (ih == 2) {
                z = f64(1.0) - z;
                if (carry != 0) { z = z - runmat_math64_scalbn(f64(1.0), q0); }
            }
        }
        if (z == f64(0.0)) {
            var accumulated = 0;
            for (var k = jz - 1; k >= jk; k = k - 1) {
                accumulated = accumulated | iq[k];
            }
            if (accumulated == 0) {
                // Cancellation exposes further 2/pi digits; do not truncate.
                var extra = 1;
                while (iq[jk - extra] == 0) { extra = extra + 1; }
                for (var k = jz + 1; k <= jz + extra; k = k + 1) {
                    f[jx + k] = f64(ipio2[jv + k]);
                    var fw = f64(0.0);
                    for (var l = 0; l <= jx; l = l + 1) {
                        fw = fw + tx[l] * f[jx + k - l];
                    }
                    q[k] = fw;
                }
                jz = jz + extra;
                continue;
            }
        }
        break;
    }
    if (z == f64(0.0)) {
        jz = jz - 1;
        q0 = q0 - 24;
        while (iq[jz] == 0) { jz = jz - 1; q0 = q0 - 24; }
    } else {
        z = runmat_math64_scalbn(z, -q0);
        if (z >= two24) {
            let fw = f64(i32(twon24 * z));
            iq[jz] = i32(z - two24 * fw);
            jz = jz + 1;
            q0 = q0 + 24;
            iq[jz] = i32(fw);
        } else { iq[jz] = i32(z); }
    }
    var fw = runmat_math64_scalbn(f64(1.0), q0);
    for (var i = jz; i >= 0; i = i - 1) {
        q[i] = fw * f64(iq[i]);
        fw = fw * twon24;
    }
    for (var i = jz; i >= 0; i = i - 1) {
        fw = f64(0.0);
        var k = 0;
        while (k <= jk && k <= jz - i) {
            fw = fw + pio2[k] * q[i + k];
            k = k + 1;
        }
        fq[jz - i] = fw;
    }
    fw = f64(0.0);
    for (var i = jz; i >= 0; i = i - 1) { fw = fw + fq[i]; }
    var hi = fw;
    fw = fq[0] - fw;
    for (var i = 1; i <= jz; i = i + 1) { fw = fw + fq[i]; }
    if (ih != 0) { hi = -hi; fw = -fw; }
    return RunmatMath64Reduction(n & 7, hi, fw);
}

fn runmat_math64_rem_pio2_medium(x: f64, ix: u32) -> RunmatMath64Reduction {
    const inv_pio2: f64 = 6.36619772367581382433e-01;
    const to_int: f64 = 6755399441055744.0;
    const pio2_1: f64 = 1.57079632673412561417e+00;
    const pio2_1t: f64 = 6.07710050650619224932e-11;
    const pio2_2: f64 = 6.07710050630396597660e-11;
    const pio2_2t: f64 = 2.02226624879595063154e-21;
    const pio2_3: f64 = 2.02226624871116645580e-21;
    const pio2_3t: f64 = 8.47842766036889956997e-32;
    let tmp = x * inv_pio2 + to_int;
    let fnn = tmp - to_int;
    let n = i32(fnn);
    var r = x - fnn * pio2_1;
    var w = fnn * pio2_1t;
    var hi = r - w;
    let ey = i32((runmat_f64_high(hi) >> 20u) & 0x7ffu);
    let ex = i32(ix >> 20u);
    if (ex - ey > 16) {
        let t = r;
        w = fnn * pio2_2;
        r = t - w;
        w = fnn * pio2_2t - ((t - r) - w);
        hi = r - w;
        let ey2 = i32((runmat_f64_high(hi) >> 20u) & 0x7ffu);
        if (ex - ey2 > 49) {
            let t2 = r;
            w = fnn * pio2_3;
            r = t2 - w;
            w = fnn * pio2_3t - ((t2 - r) - w);
            hi = r - w;
        }
    }
    return RunmatMath64Reduction(n, hi, (r - hi) - w);
}

fn runmat_math64_rem_pio2(x: f64) -> RunmatMath64Reduction {
    const pio2_1: f64 = 1.57079632673412561417e+00;
    const pio2_1t: f64 = 6.07710050650619224932e-11;
    let high = runmat_f64_high(x);
    let sign = high >> 31u;
    let ix = high & 0x7fffffffu;
    // The caller handles the unreduced interval. Preserve the upstream fast
    // 1..4 quadrant cases, including its cancellation escape to medium.
    var quick_n = 0;
    if (ix <= 0x400f6a7au) {
        if ((ix & 0xfffffu) == 0x921fbu) {
            return runmat_math64_rem_pio2_medium(x, ix);
        }
        quick_n = select(2, 1, ix <= 0x4002d97cu);
    } else if (ix <= 0x401c463bu) {
        if (ix <= 0x4015fdbcu) {
            if (ix == 0x4012d97cu) { return runmat_math64_rem_pio2_medium(x, ix); }
            quick_n = 3;
        } else {
            if (ix == 0x401921fbu) { return runmat_math64_rem_pio2_medium(x, ix); }
            quick_n = 4;
        }
    }
    if (quick_n != 0) {
        if (sign != 0u) { quick_n = -quick_n; }
        let z = x - f64(quick_n) * pio2_1;
        let correction = f64(quick_n) * pio2_1t;
        let hi = z - correction;
        return RunmatMath64Reduction(quick_n, hi, (z - hi) - correction);
    }
    if (ix < 0x413921fbu) { return runmat_math64_rem_pio2_medium(x, ix); }
    if (ix >= 0x7ff00000u) { return RunmatMath64Reduction(0, x - x, x - x); }
    var z = runmat_f64_from_words(runmat_f64_low(x), (ix & 0xfffffu) | 0x41600000u);
    var tx: array<f64, 3>;
    for (var i = 0; i < 2; i = i + 1) {
        tx[i] = f64(i32(z));
        z = (z - tx[i]) * f64(16777216.0);
    }
    tx[2] = z;
    var last = 2;
    while (last != 0 && tx[last] == f64(0.0)) { last = last - 1; }
    let reduced = runmat_math64_rem_pio2_large(tx, last + 1, i32(ix >> 20u) - (0x3ff + 23));
    if (sign != 0u) { return RunmatMath64Reduction(-reduced.n, -reduced.hi, -reduced.lo); }
    return reduced;
}
