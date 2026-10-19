//! Decimal scaling around the shared ties-away operation.
use runmat_accelerate_api::ProviderPrecision;

pub(super) fn functions(precision: ProviderPrecision) -> String {
    let (ty, maximum, subnormal, normalization, normalization_log, minimum, classify, normalize, pack, negative, overflow) = match precision {
        ProviderPrecision::F32 => (
            // Nearest Single representation of ln(2^23).
            "f32", 38, 45, "8388608.0", "f32(15.942384719848633)", "1.1754943508222875e-38",
            "let bits = bitcast<u32>(a); let exponent = bits & 0x7f800000u; let fraction = bits & 0x007fffffu; let special = exponent == 0x7f800000u; let zero = exponent == 0u && fraction == 0u; let tiny = exponent == 0u;",
            "f32(fraction) * f32(MINIMUM)",
            "return bitcast<f32>(u32(units) | (bitcast<u32>(a) & 0x80000000u));",
            "(bits & 0x80000000u) != 0u",
            "(bitcast<u32>(scaled) & 0x7f800000u) == 0x7f800000u",
        ),
        ProviderPrecision::F64 => (
            // Nearest Double representation of ln(2^52); Naga cannot
            // evaluate the equivalent constant f64 log expression.
            "f64", 308, 324, "4503599627370496.0", "f64(36.04365338911715)", "2.2250738585072014e-308",
            "let low = runmat_f64_low(a); let high = runmat_f64_high(a); let exponent = high & 0x7ff00000u; let fraction_high = high & 0x000fffffu; let special = exponent == 0x7ff00000u; let zero = exponent == 0u && fraction_high == 0u && low == 0u; let tiny = exponent == 0u;",
            "(f64(fraction_high) * f64(4294967296.0) + f64(low)) * f64(MINIMUM)",
            "let upper = u32(floor(units / f64(4294967296.0))); let lower = u32(units - f64(upper) * f64(4294967296.0)); return runmat_f64_from_words(lower, upper | (runmat_f64_high(a) & 0x80000000u));",
            "(high & 0x80000000u) != 0u",
            "(runmat_f64_high(scaled) & 0x7ff00000u) == 0x7ff00000u",
        ),
    };
    let normalize = normalize.replace("MINIMUM", minimum);
    let name = super::entrypoint(precision);
    let roundtrip = match precision {
        ProviderPrecision::F32 => 9,
        ProviderPrecision::F64 => 17,
    };
    format!(
        r#"
fn rm_round_divide_{ty}(numerator: {ty}, denominator: {ty}) -> {ty} {{
    let quotient = numerator / denominator;
    let residual = fma(-quotient, denominator, numerator);
    return quotient + residual / denominator;
}}

fn rm_round_power10_{ty}(exponent: i32) -> {ty} {{
    var count = u32(abs(exponent));
    var base = {ty}(10.0);
    var value = {ty}(1.0);
    loop {{
        if count == 0u {{ break; }}
        if (count & 1u) != 0u {{ value = value * base; }}
        count = count >> 1u;
        if count != 0u {{ base = base * base; }}
    }}
    return value;
}}

fn rm_round_digits_{ty}(a: {ty}, requested: i32, significant: bool) -> {ty} {{
    {classify}
    if special {{ return a; }}
    if zero {{
        if significant {{ return {ty}(0.0); }}
        return a;
    }}
    var magnitude = abs(a);
    if tiny {{ magnitude = {normalize}; }}
    var order = floor(log(magnitude) * {ty}(0.43429448190325182765));
    let LN_NORMALIZATION: {ty} = {normalization_log};
    if tiny {{ order = floor(log(magnitude) * {ty}(0.43429448190325182765) - LN_NORMALIZATION * {ty}(0.43429448190325182765)); }}
    var digits = requested;
    if significant {{
        // Bound the control before subtraction, including for i32::MAX.
        if requested >= {roundtrip} {{ return a; }}
        digits = requested - 1 - i32(order);
    }}
    if digits >= {roundtrip} - 1 - i32(order) {{ return a; }}
    if digits > {subnormal} {{ return a; }}
    if digits < -{maximum} {{
        let units = {ty}(0.0);
        {pack}
    }}
    if digits == 0 {{ return {name}(a); }}
    let first = min(digits, {maximum});
    let factor = rm_round_power10_{ty}(abs(first));
    let tail = rm_round_power10_{ty}(digits - first);
    var scaled = (magnitude * factor) * tail;
    if first < 0 {{ scaled = rm_round_divide_{ty}(magnitude, factor); }}
    if tiny {{ scaled = scaled / {ty}({normalization}); }}
    // Overflow means the quantum is below the represented input's spacing.
    if {overflow} {{ return a; }}
    let rounded = {name}(scaled);
    var result = rm_round_divide_{ty}(rm_round_divide_{ty}(rounded, factor), tail);
    if first < 0 {{ result = rounded * factor; }}
    if first > 0 && result < {ty}({minimum}) {{
        // Construct subnormal results from normal arithmetic intermediates.
        let units = round(rm_round_divide_{ty}(rm_round_divide_{ty}(rounded, factor * {ty}({minimum})), tail) * {ty}({normalization}));
        {pack}
    }}
    if {negative} {{ return -result; }}
    return result;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_logs_keep_the_represented_inputs_and_order_formula() {
        let single = functions(ProviderPrecision::F32);
        let double = functions(ProviderPrecision::F64);
        assert_eq!(15.942_385_f32.to_bits(), 0x417f1402);
        assert_eq!(36.04365338911715_f64.to_bits(), 0x404205966f2b4f12);
        assert!((15.942_385_f32 - (2.0_f32.powi(23)).ln()).abs() <= 8.0 * f32::EPSILON);
        assert!((36.04365338911715_f64 - (2.0_f64.powi(52)).ln()).abs() <= 32.0 * f64::EPSILON);
        for (source, ty, typed_constant) in [
            (&single, "f32", "f32(15.942384719848633)"),
            (&double, "f64", "f64(36.04365338911715)"),
        ] {
            assert!(source.contains(&format!("let LN_NORMALIZATION: {ty} = {typed_constant};")));
            assert!(source.contains("- LN_NORMALIZATION *"));
            assert!(!source.contains("log(f64(4503599627370496.0))"));
        }
    }
}
