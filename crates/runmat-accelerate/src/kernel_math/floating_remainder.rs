//! Encoded floating reduction shared by standalone and fused mod/rem.
use runmat_accelerate_api::ProviderPrecision;
use runmat_runtime::builtins::common::spec::{FusionMathHelper, ScalarType};

mod format;

/// Static work bound for one represented pair, including normalization,
/// whole-number checks, compensation, reduction/alignment, and packing.
/// The reduction and opposite-sign alignment paths are mutually exclusive.
pub(crate) fn scalar_work_units(precision: ProviderPrecision) -> u64 {
    let (fraction_bits, max_gap) = match precision {
        ProviderPrecision::F32 => (23_u64, 276_u64),
        ProviderPrecision::F64 => (52_u64, 2097_u64),
    };
    64 + (max_gap + 7 * fraction_bits) * 8
}

pub(crate) fn entrypoint(precision: ProviderPrecision) -> &'static str {
    FusionMathHelper::FloatingRemainder
        .entrypoint(match precision {
            ProviderPrecision::F32 => ScalarType::F32,
            ProviderPrecision::F64 => ScalarType::F64,
        })
        .expect("floating remainder precision")
}

#[cfg(any(feature = "wgpu", test))]
pub(crate) fn functions(precision: ProviderPrecision) -> String {
    let mut source = String::new();
    write_functions(precision, &mut source).expect("String writer is infallible");
    source
}

pub(crate) fn write_functions(
    precision: ProviderPrecision,
    writer: &mut impl std::fmt::Write,
) -> std::fmt::Result {
    let format = format::Format::new(precision);
    let ty = format.ty;
    let fraction_bits = format.fraction_bits;
    let bias = format.bias;
    let implicit = format.implicit;
    let decode = format.decode;
    let encode = format.encode;
    let nan = format.nan;
    let nan_bits = match precision {
        ProviderPrecision::F32 => "let nan_bits: u32 = 0x7fc00000u;",
        ProviderPrecision::F64 => "",
    };
    let name = entrypoint(precision);
    write!(
        writer,
        r#"
struct RmRemainderParts_{ty} {{
    significand: {ty},
    exponent: i32,
    negative: bool,
    nan: bool,
    infinite: bool,
}};

fn rm_remainder_decode_{ty}(value: {ty}) -> RmRemainderParts_{ty} {{
    {decode}
    var exponent = i32(encoded_exponent) - {bias} - {fraction_bits};
    var significand = fraction;
    if encoded_exponent == 0u {{ exponent = 1 - {bias} - {fraction_bits}; }}
    else {{ significand = significand + {ty}({implicit}); }}
    if !special && significand != {ty}(0.0) {{
        loop {{
            if significand >= {ty}({implicit}) {{ break; }}
            significand = significand * {ty}(2.0);
            exponent = exponent - 1;
        }}
    }}
    return RmRemainderParts_{ty}(significand, exponent, negative, special && fraction != {ty}(0.0), special && fraction == {ty}(0.0));
}}

fn rm_remainder_pack_{ty}(significand_input: {ty}, exponent_input: i32, negative: bool) -> {ty} {{
    var significand = significand_input;
    var exponent = exponent_input;
    if significand != {ty}(0.0) {{
        loop {{
            if significand >= {ty}({implicit}) {{ break; }}
            significand = significand * {ty}(2.0);
            exponent = exponent - 1;
        }}
    }}
    var encoded_exponent = 0u;
    if significand != {ty}(0.0) {{
        let encoded = exponent + {fraction_bits} + {bias};
        if encoded > 0 {{
            encoded_exponent = u32(encoded);
            significand = significand - {ty}({implicit});
        }} else {{
            // Only normal significands are divided. No arithmetic result is
            // subnormal: the final subnormal is constructed from its bits.
            var shift = 1 - encoded;
            loop {{
                if shift == 0 {{ break; }}
                significand = significand * {ty}(0.5);
                shift = shift - 1;
            }}
        }}
    }}
    {encode}
}}

fn rm_remainder_whole_{ty}(parts: RmRemainderParts_{ty}) -> bool {{
    if parts.exponent >= 0 {{ return true; }}
    if parts.exponent < -{fraction_bits} {{ return false; }}
    var value = parts.significand;
    var shift = -parts.exponent;
    loop {{
        if shift == 0 {{ break; }}
        value = value * {ty}(0.5);
        shift = shift - 1;
    }}
    return value == trunc(value);
}}

fn rm_remainder_compensated_zero_{ty}(a: RmRemainderParts_{ty}, b: RmRemainderParts_{ty}) -> bool {{
    if rm_remainder_whole_{ty}(a) && rm_remainder_whole_{ty}(b) {{ return false; }}
    let gap = a.exponent - b.exponent;
    if gap < 0 || gap > {fraction_bits} {{ return false; }}
    var divisor = b.significand;
    for (var shift = 0; shift < gap; shift = shift + 1) {{ divisor = divisor * {ty}(0.5); }}
    let candidate = round(a.significand / divisor);
    return candidate < {ty}({implicit}) && candidate * divisor == a.significand;
}}

fn {name}(left: {ty}, right: {ty}, modulus: bool) -> {ty} {{
    {nan_bits}
    let a = rm_remainder_decode_{ty}(left);
    let b = rm_remainder_decode_{ty}(right);
    if a.nan || b.nan {{ return {nan}; }}
    if b.significand == {ty}(0.0) {{
        if modulus {{ return left; }}
        return {nan};
    }}
    if a.infinite {{ return {nan}; }}
    if a.significand == {ty}(0.0) {{ return {ty}(0.0); }}
    if b.infinite {{
        if modulus && a.negative != b.negative {{ return right; }}
        return left;
    }}
    if rm_remainder_compensated_zero_{ty}(a, b) {{ return {ty}(0.0); }}
    var residue = a.significand;
    var exponent = a.exponent;
    if a.exponent >= b.exponent {{
        var shift = a.exponent - b.exponent;
        loop {{
            if residue >= b.significand {{ residue = residue - b.significand; }}
            if shift == 0 || residue == {ty}(0.0) {{ break; }}
            residue = residue * {ty}(2.0);
            shift = shift - 1;
        }}
        exponent = b.exponent;
    }}
    if residue == {ty}(0.0) {{ return {ty}(0.0); }}
    if modulus && a.negative != b.negative {{
        // Match the divisor's normal significand scale before subtracting.
        var shift = b.exponent - exponent;
        loop {{
            if shift <= 0 {{ break; }}
            residue = residue * {ty}(0.5);
            shift = shift - 1;
        }}
        return rm_remainder_pack_{ty}(b.significand - residue, b.exponent, b.negative);
    }}
    return rm_remainder_pack_{ty}(residue, exponent, a.negative);
}}
"#
    )
}
