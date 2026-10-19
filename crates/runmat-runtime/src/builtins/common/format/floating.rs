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
