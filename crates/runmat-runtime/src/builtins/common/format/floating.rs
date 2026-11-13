//! Floating conversion shared by sprintf and fprintf.

use super::{apply_width, group_float_mantissa, FloatStyle, FormatFlags};
use crate::BuiltinResult;

pub(super) fn format_float(
    value: f64,
    style: FloatStyle,
    flags: FormatFlags,
    width: Option<isize>,
    precision: Option<isize>,
    alternate: bool,
) -> BuiltinResult<String> {
    let mut sign = String::new();
    let mut magnitude = value;

    if value.is_nan() {
        return apply_width(
            String::new(),
            String::new(),
            "NaN".to_string(),
            flags,
            width,
            false,
        );
    }

    if value.is_infinite() {
        if value.is_sign_negative() {
            sign.push('-');
        } else if flags.sign_plus {
            sign.push('+');
        } else if flags.sign_space {
            sign.push(' ');
        }
        return apply_width(sign, String::new(), "Inf".to_string(), flags, width, false);
    }

    if value.is_sign_negative() {
        sign.push('-');
        magnitude = -value;
    } else if flags.sign_plus {
        sign.push('+');
    } else if flags.sign_space {
        sign.push(' ');
    }

    let prec = precision.unwrap_or(6).max(0) as usize;
    let mut body = match style {
        FloatStyle::FixedLower | FloatStyle::FixedUpper => format!("{magnitude:.prec$}"),
        FloatStyle::ExponentLower => normalize_exponent(format!("{magnitude:.prec$e}"), false),
        FloatStyle::ExponentUpper => normalize_exponent(format!("{magnitude:.prec$E}"), true),
        FloatStyle::GeneralLower => format_general(magnitude, prec, false, alternate),
        FloatStyle::GeneralUpper => format_general(magnitude, prec, true, alternate),
    };

    if alternate {
        ensure_decimal_point(&mut body);
    }

    if flags.grouping && style.permits_grouping_or_decimal() {
        body = group_float_mantissa(&body);
    }

    let zero_pad_allowed = flags.zero_pad && !flags.left_align;
    apply_width(sign, String::new(), body, flags, width, zero_pad_allowed)
}

fn format_general(value: f64, precision: usize, uppercase: bool, alternate: bool) -> String {
    // General precision counts significant digits; zero precision means one.
    let significant = precision.max(1);
    // Determine notation from the rounded scientific representation, not the
    // unrounded logarithm: rounding may move a value across the threshold.
    let scientific = format!("{value:.digits$e}", digits = significant - 1);
    let Some((mantissa, raw_exponent)) = scientific.split_once('e') else {
        return scientific;
    };
    let Ok(exponent) = raw_exponent.parse::<i32>() else {
        return scientific;
    };
    let use_exponent = exponent < -4 || exponent >= significant as i32;

    if use_exponent {
        let mut digits = mantissa.to_string();
        if !alternate {
            trim_fractional_zeros(&mut digits);
        }
        let symbol = if uppercase { 'E' } else { 'e' };
        format!("{digits}{symbol}{exponent:+03}")
    } else {
        let fractional = (significant as i64 - i64::from(exponent) - 1).max(0) as usize;
        let mut digits = format!("{value:.fractional$}");
        if !alternate {
            trim_fractional_zeros(&mut digits);
        }
        digits
    }
}

fn trim_fractional_zeros(mantissa: &mut String) {
    if mantissa.contains('.') {
        while mantissa.ends_with('0') {
            mantissa.pop();
        }
        if mantissa.ends_with('.') {
            mantissa.pop();
        }
    }
}

fn ensure_decimal_point(text: &mut String) {
    let exponent = text.find(['e', 'E']).unwrap_or(text.len());
    if !text[..exponent].contains('.') {
        text.insert(exponent, '.');
    }
}

fn normalize_exponent(text: String, uppercase: bool) -> String {
    let marker = if uppercase { 'E' } else { 'e' };
    let Some((mantissa, raw_exponent)) = text.split_once(marker) else {
        return text;
    };
    let Ok(exponent) = raw_exponent.parse::<i32>() else {
        return text;
    };
    format!("{mantissa}{marker}{exponent:+03}")
}

/// Numeric text-file conversion retains the scalar's storage class until its
/// general-format precision is selected. Integers never pass through a float.
pub(crate) fn format_delimited_numeric(value: runmat_value::NumericScalar) -> String {
    if let Some(integer) = value.into_int_value() {
        return integer.decimal_string();
    }
    let significant = if matches!(value, runmat_value::NumericScalar::F32(_)) {
        7
    } else {
        15
    };
    let value = value.materialize_f64();
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Inf"
        } else {
            "Inf"
        }
        .to_string();
    }
    let mut body = format_general(value.abs(), significant, false, false);
    if value.is_sign_negative() {
        body.insert(0, '-');
    }
    body
}

#[cfg(test)]
mod delimited_tests {
    use super::*;
    use runmat_value::NumericScalar::*;

    #[test]
    fn class_precision_rounding_carry_and_nonfinite_text_are_bounded() {
        for (value, expected) in [
            (F64(1.693061224489796), "1.6930612244898"),
            (F32(1.2345679), "1.234568"),
            (F64(1e-20), "1e-20"),
            (F32(1e-8), "1e-08"),
            (F64(1e20), "1e+20"),
            (F64(9.999999999999996e14), "1e+15"),
            (F64(0.00009999999999999996), "0.0001"),
            (F32(9.999999e6), "9999999"),
            (F64(-0.0), "-0"),
            (F32(-0.0), "-0"),
            (F64(f64::NAN), "NaN"),
            (F64(f64::INFINITY), "Inf"),
            (F32(f32::NEG_INFINITY), "-Inf"),
            (I64(i64::MIN), "-9223372036854775808"),
            (U64(u64::MAX), "18446744073709551615"),
        ] {
            assert_eq!(format_delimited_numeric(value), expected, "{value:?}");
        }
    }
}
