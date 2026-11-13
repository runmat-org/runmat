//! Allocation-free per-cell formatting into the caller's admitted scratch.
//! Keep the spelling and rounding policy paired with the ordinary formatter.

use std::fmt::Write as _;

use super::{CustomFormat, CustomKind, FormatOptions, FormatSpec, DEFAULT_PRECISION};
use crate::builtins::strings::core::numeric_format_work::{FormatBuffer, FormatCapacityExceeded};

type Result<T> = std::result::Result<T, FormatCapacityExceeded>;

fn trim_trailing_zeros(output: &mut FormatBuffer<'_>) {
    let Some(dot) = output.as_str().find('.') else {
        return;
    };
    let mut end = output.len();
    while end > dot + 1 && output.as_str().as_bytes()[end - 1] == b'0' {
        end -= 1;
    }
    if end > dot && output.as_str().as_bytes()[end - 1] == b'.' {
        end -= 1;
    }
    output.truncate(end);
}

fn normalize_negative_zero(output: &mut FormatBuffer<'_>) -> Result<()> {
    if output.as_str().starts_with('-') && output.as_str()[1..].bytes().all(|byte| byte == b'0') {
        output.clear();
        output.ascii(b'0')?;
    }
    Ok(())
}

fn general(
    value: f64,
    digits: usize,
    uppercase: bool,
    output: &mut FormatBuffer<'_>,
) -> Result<()> {
    if value.is_nan() {
        return output.append("NaN");
    }
    if value.is_infinite() {
        return output.append(if value.is_sign_negative() {
            "-Inf"
        } else {
            "Inf"
        });
    }
    if value == 0.0 {
        return output.append("0");
    }
    let significant = digits.max(1);
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= significant as i32 {
        let precision = significant.saturating_sub(1);
        if uppercase {
            output.checked_format(format_args!("{:.*E}", precision, value))?;
        } else {
            output.checked_format(format_args!("{:.*e}", precision, value))?;
        }
        let marker = if uppercase { 'E' } else { 'e' };
        if let Some(index) = output.as_str().find(marker) {
            let mut end = index;
            while end > 0 && output.as_str().as_bytes()[end - 1] == b'0' {
                end -= 1;
            }
            if end > 0 && output.as_str().as_bytes()[end - 1] == b'.' {
                end -= 1;
            }
            output.remove_range(end, index);
            // The ordinary implementation normalizes the mantissa before
            // appending the exponent. A zero mantissa is relevant only after
            // decimal rounding; preserve the same policy here.
            let marker_index = output.as_str().find(marker).expect("exponent retained");
            if output.as_str()[..marker_index].starts_with('-')
                && output.as_str()[1..marker_index]
                    .bytes()
                    .all(|byte| byte == b'0')
            {
                output.remove_range(0, marker_index);
                output.insert_ascii(0, b'0')?;
            }
        }
    } else {
        let decimals = if significant as i32 - 1 - exponent < 0 {
            0
        } else {
            (significant as i32 - 1 - exponent) as usize
        };
        output.checked_format(format_args!("{:.*}", decimals, value))?;
        trim_trailing_zeros(output);
        normalize_negative_zero(output)?;
    }
    Ok(())
}

fn flags(output: &mut FormatBuffer<'_>, format: &CustomFormat) -> Result<()> {
    if format.sign_always
        && !output.as_str().starts_with('-')
        && !output.as_str().starts_with('+')
        && output.as_str() != "NaN"
    {
        output.insert_ascii(0, b'+')?;
    }
    let width = format.width.unwrap_or(0);
    if output.len() >= width {
        return Ok(());
    }
    let pad = width - output.len();
    if format.left_align {
        return output.repeat_ascii(b' ', pad);
    }
    if format.zero_pad {
        let signed = output.as_str().starts_with('+') || output.as_str().starts_with('-');
        if !signed {
            // Preserve the existing formatter's unsigned-width behavior:
            // zero padding is applied only after an explicit sign.
            return output.pad_left(pad, b' ');
        }
        output.pad_left(pad, b'0')?;
        // Move the original sign across the newly inserted zero padding.
        output.swap_ascii(0, pad);
        return Ok(());
    }
    output.pad_left(pad, b' ')
}

fn custom(value: f64, format: &CustomFormat, output: &mut FormatBuffer<'_>) -> Result<()> {
    if value.is_nan() {
        return output.append("NaN");
    }
    if value.is_infinite() {
        return output.append(if value.is_sign_negative() {
            "-Inf"
        } else {
            "Inf"
        });
    }
    let precision = format.precision.unwrap_or(match format.kind {
        CustomKind::Fixed | CustomKind::Exponent => 6,
        CustomKind::General => DEFAULT_PRECISION,
    });
    match format.kind {
        CustomKind::Fixed => output.checked_format(format_args!("{:.*}", precision, value))?,
        CustomKind::Exponent => {
            output.checked_format(format_args!("{:.*e}", precision, value))?;
            if format.uppercase {
                output.upper_ascii();
            }
        }
        CustomKind::General => general(value, precision.max(1), format.uppercase, output)?,
    }
    if format.kind != CustomKind::Fixed {
        trim_trailing_zeros(output);
        normalize_negative_zero(output)?;
    }
    flags(output, format)
}

pub(super) fn real(
    value: f64,
    options: &FormatOptions,
    output: &mut FormatBuffer<'_>,
) -> Result<()> {
    output.clear();
    match &options.spec {
        FormatSpec::General { digits } => general(value, *digits, false, output),
        FormatSpec::Custom(format) => custom(value, format, output),
    }
}

fn integer_exponent(
    negative: bool,
    digits: &str,
    significant: usize,
    uppercase: bool,
    output: &mut FormatBuffer<'_>,
) -> Result<()> {
    let significant = significant.max(1);
    let mut exponent = digits.len() - 1;
    let mut rounded = [0u8; 64];
    let count = digits.len().min(significant);
    rounded[..count].copy_from_slice(&digits.as_bytes()[..count]);
    if digits.len() > significant && digits.as_bytes()[significant] >= b'5' {
        for position in (0..count).rev() {
            if rounded[position] < b'9' {
                rounded[position] += 1;
                break;
            }
            rounded[position] = b'0';
            if position == 0 {
                rounded[0] = b'1';
                exponent += 1;
            }
        }
    }
    if negative {
        output.ascii(b'-')?;
    }
    output.ascii(rounded[0])?;
    let fraction = &rounded[1..count];
    let fraction_end = fraction
        .iter()
        .rposition(|byte| *byte != b'0')
        .map_or(0, |index| index + 1);
    if fraction_end > 0 {
        output.ascii(b'.')?;
        output.append(std::str::from_utf8(&fraction[..fraction_end]).expect("ASCII digits"))?;
    }
    output.ascii(if uppercase { b'E' } else { b'e' })?;
    output.ascii(b'+')?;
    write!(output, "{exponent}").map_err(|_| FormatCapacityExceeded)
}

pub(super) fn integer(
    raw: &str,
    options: &FormatOptions,
    output: &mut FormatBuffer<'_>,
) -> Result<()> {
    output.clear();
    let (negative, digits) = raw
        .strip_prefix('-')
        .map_or((false, raw), |digits| (true, digits));
    match &options.spec {
        FormatSpec::General { digits: precision } => {
            if digits.len() <= (*precision).max(1) {
                if negative {
                    output.ascii(b'-')?;
                }
                output.append(digits)?;
            } else {
                integer_exponent(negative, digits, *precision, false, output)?;
            }
        }
        FormatSpec::Custom(format) => {
            let precision = format.precision.unwrap_or(match format.kind {
                CustomKind::Fixed | CustomKind::Exponent => 6,
                CustomKind::General => DEFAULT_PRECISION,
            });
            match format.kind {
                CustomKind::Fixed => {
                    if negative {
                        output.ascii(b'-')?;
                    }
                    output.append(digits)?;
                    if precision > 0 {
                        output.ascii(b'.')?;
                        output.repeat_ascii(b'0', precision)?;
                    }
                }
                CustomKind::Exponent => {
                    integer_exponent(negative, digits, precision + 1, format.uppercase, output)?
                }
                CustomKind::General => {
                    if digits.len() <= precision.max(1) {
                        if negative {
                            output.ascii(b'-')?;
                        }
                        output.append(digits)?;
                    } else {
                        integer_exponent(
                            negative,
                            digits,
                            precision.max(1),
                            format.uppercase,
                            output,
                        )?;
                    }
                }
            }
            flags(output, format)?;
        }
    }
    Ok(())
}

pub(super) fn complex(
    real_part: f64,
    imag_part: f64,
    options: &FormatOptions,
    output: &mut FormatBuffer<'_>,
    secondary: &mut FormatBuffer<'_>,
) -> Result<()> {
    real(real_part, options, output)?;
    let abs_imag = if imag_part == 0.0 {
        0.0
    } else {
        imag_part.abs()
    };
    real(abs_imag, options, secondary)?;
    if abs_imag == 0.0 && !imag_part.is_nan() {
        return Ok(());
    }
    if real_part == 0.0 && !real_part.is_sign_negative() && !real_part.is_nan() {
        output.clear();
        if imag_part.is_sign_negative()
            && !imag_part.is_nan()
            && !secondary.as_str().starts_with('-')
        {
            output.ascii(b'-')?;
        }
        output.append(secondary.as_str())?;
        output.ascii(b'i')?;
        return Ok(());
    }
    output.append(if imag_part.is_sign_negative() {
        " - "
    } else {
        " + "
    })?;
    output.append(secondary.as_str())?;
    output.ascii(b'i')
}
