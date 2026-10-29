// Exact binary64 exponent scaling, including round-to-nearest-even at the
// subnormal boundary. Integer packing avoids a flushed floating intermediate.
// Requires the shared binary64 word helpers and SHADER_INT64.
fn runmat_math64_scalbn(value: f64, requested_exponent: i32) -> f64 {
    let bits = bitcast<u64>(value);
    let sign = bits & (u64(0x80000000u) << 32u);
    let fraction_mask = (u64(0x000fffffu) << 32u) | u64(0xffffffffu);
    let hidden = u64(1u) << 52u;
    let encoded_exponent = u32((bits >> 52u) & u64(0x7ffu));
    var significand = bits & fraction_mask;
    if (encoded_exponent == 0x7ffu || (encoded_exponent == 0u && significand == u64(0u))) {
        return value;
    }
    if (requested_exponent > 4096) {
        return bitcast<f64>(sign | (u64(0x7ff00000u) << 32u));
    }
    if (requested_exponent < -4096) { return bitcast<f64>(sign); }
    var exponent = i32(encoded_exponent) - 1023;
    if (encoded_exponent == 0u) {
        let high = u32(significand >> 32u);
        var shift = 0u;
        if (high != 0u) {
            shift = countLeadingZeros(high) - 11u;
        } else {
            shift = countLeadingZeros(u32(significand)) + 21u;
        }
        significand = significand << shift;
        exponent = -1022 - i32(shift);
    } else {
        significand = significand | hidden;
    }
    exponent = exponent + requested_exponent;
    if (exponent > 1023) {
        return bitcast<f64>(sign | (u64(0x7ff00000u) << 32u));
    }
    if (exponent >= -1022) {
        return bitcast<f64>(sign | (u64(u32(exponent + 1023)) << 52u) |
                           (significand & fraction_mask));
    }
    if (exponent < -1075) { return bitcast<f64>(sign); }
    let shift = u32(-1022 - exponent);
    let mask = (u64(1u) << shift) - u64(1u);
    let remainder = significand & mask;
    let halfway = u64(1u) << (shift - 1u);
    var rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (rounded & u64(1u)) != u64(0u))) {
        rounded = rounded + u64(1u);
    }
    // Rounding may produce the smallest normal value; its encoding is already
    // the hidden bit, so this also handles that transition without another op.
    return bitcast<f64>(sign | rounded);
}
