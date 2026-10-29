// Binary64 angle conversions, with exact integer multiplication for the tiny
// range. Native devices may flush subnormal operands/results. Scaling a native
// rounded product down would also introduce double rounding; instead retain
// all 106 product bits and round once, including at the subnormal boundary.
// Requires SHADER_INT64 and shared binary64 word helpers, but not trig kernels.

// Precondition: finite |x| < 2^-767, and positive normal factor near unity.
// This helper is deliberately limited to the radians/degrees constants.
fn runmat_math64_angle_tiny_product(x: f64, factor: f64) -> f64 {
    let raw = bitcast<u64>(x);
    let cbits = bitcast<u64>(factor);
    let mask32 = u64(0xffffffffu);
    let fraction = (u64(0xfffffu) << 32u) | mask32;
    let hidden = u64(1u) << 52u;
    let sign = raw & (u64(0x80000000u) << 32u);
    let encoded = i32((raw >> 52u) & u64(0x7ffu));
    var a = raw & fraction;
    var ex = -1022;
    if (encoded != 0) { a = a | hidden; ex = encoded - 1023; }
    if (a == u64(0u)) { return x; }
    let b = (cbits & fraction) | hidden;
    let ec = i32((cbits >> 52u) & u64(0x7ffu)) - 1023;
    let p0 = (a & mask32) * (b & mask32);
    let p1 = (a >> 32u) * (b & mask32);
    let p2 = (a & mask32) * (b >> 32u);
    let p3 = (a >> 32u) * (b >> 32u);
    let middle = (p0 >> 32u) + (p1 & mask32) + (p2 & mask32);
    let low = (p0 & mask32) | (middle << 32u);
    let high = p3 + (p1 >> 32u) + (p2 >> 32u) + (middle >> 32u);
    var top = 0;
    if (u32(high >> 32u) != 0u) {
        top = 127 - i32(countLeadingZeros(u32(high >> 32u)));
    } else if (u32(high) != 0u) {
        top = 95 - i32(countLeadingZeros(u32(high)));
    } else {
        // factor has 53 significand bits, so even minimum subnormal x
        // produces at least 53 product bits.
        top = 63 - i32(countLeadingZeros(u32(low >> 32u)));
    }
    var exponent = ex + ec - 104 + top;
    let shift = u32(max(top - 52, -970 - ex - ec));
    // The two angle constants bound shift to [47, 58], hence both shifts
    // below are strictly between zero and 64 and no wide-shift UB occurs.
    var rounded = (high << (64u - shift)) | (low >> shift);
    let remainder = low & ((u64(1u) << shift) - u64(1u));
    let halfway = u64(1u) << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & u64(1u)) != u64(0u))) {
        rounded = rounded + u64(1u);
    }
    if (exponent < -1022) {
        // Rounded may equal 2^52 (the smallest normal), already its encoding.
        return bitcast<f64>(sign | rounded);
    }
    if (rounded >= (hidden << 1u)) { rounded = rounded >> 1u; exponent = exponent + 1; }
    return bitcast<f64>(sign | (u64(u32(exponent + 1023)) << 52u) | (rounded & fraction));
}

fn runmat_math64_radians(x: f64) -> f64 {
    const factor: f64 = 0.017453292519943295769236907684886127;
    if ((runmat_f64_high(x) & 0x7fffffffu) < 0x10000000u) {
        return runmat_math64_angle_tiny_product(x, factor);
    }
    return x * factor;
}

fn runmat_math64_degrees(x: f64) -> f64 {
    const factor: f64 = 57.29577951308232087679815481410517;
    if ((runmat_f64_high(x) & 0x7fffffffu) < 0x10000000u) {
        return runmat_math64_angle_tiny_product(x, factor);
    }
    return x * factor;
}
