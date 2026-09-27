//! Decimal scaling around the shared ties-away operation.
use runmat_accelerate_api::ProviderPrecision;

pub(super) fn functions(precision: ProviderPrecision) -> String {
    let (ty, maximum, subnormal, normalization, minimum, classify, normalize, pack, negative, overflow) = match precision {
        ProviderPrecision::F32 => (
            "f32", 38, 45, "8388608.0", "1.1754943508222875e-38",
            "let bits = bitcast<u32>(a); let exponent = bits & 0x7f800000u; let fraction = bits & 0x007fffffu; let special = exponent == 0x7f800000u; let zero = exponent == 0u && fraction == 0u; let tiny = exponent == 0u;",
            "f32(fraction) * f32(MINIMUM)",
            "return bitcast<f32>(u32(units) | (bitcast<u32>(a) & 0x80000000u));",
            "(bits & 0x80000000u) != 0u",
            "(bitcast<u32>(scaled) & 0x7f800000u) == 0x7f800000u",
        ),
        ProviderPrecision::F64 => (
            "f64", 308, 324, "4503599627370496.0", "2.2250738585072014e-308",
            "let bits = bitcast<vec2<u32>>(a); let exponent = bits.y & 0x7ff00000u; let fraction_high = bits.y & 0x000fffffu; let special = exponent == 0x7ff00000u; let zero = exponent == 0u && fraction_high == 0u && bits.x == 0u; let tiny = exponent == 0u;",
            "(f64(fraction_high) * f64(4294967296.0) + f64(bits.x)) * f64(MINIMUM)",
            "let upper = u32(floor(units / f64(4294967296.0))); let lower = u32(units - f64(upper) * f64(4294967296.0)); return bitcast<f64>(vec2<u32>(lower, upper | (bitcast<vec2<u32>>(a).y & 0x80000000u)));",
            "(bits.y & 0x80000000u) != 0u",
            "(bitcast<vec2<u32>>(scaled).y & 0x7ff00000u) == 0x7ff00000u",
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
    if tiny {{ order = floor(log(magnitude) * {ty}(0.43429448190325182765) - log({ty}({normalization})) * {ty}(0.43429448190325182765)); }}
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
