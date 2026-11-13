//! Significant decimal rendering of existing binary64 values and exact rationals.
use super::MAX_DIGITS;
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
pub(super) fn significant_decimal_from_f64(value: f64, digits: usize) -> String {
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
    if value == 0.0 {
        return "0".to_string();
    }
    let precision = digits.saturating_sub(1).min(MAX_DIGITS);
    let scientific = format!("{value:.precision$e}");
    scientific_to_decimal(&scientific)
}

pub(super) fn significant_decimal_from_rational(
    numerator: &BigInt,
    denominator: &BigInt,
    digits: usize,
) -> String {
    let negative = numerator.is_negative() ^ denominator.is_negative();
    let numerator = numerator.abs();
    let denominator = denominator.abs();
    if denominator.is_zero() {
        return "NaN".to_string();
    }
    let integer = &numerator / &denominator;
    let mut remainder = numerator % &denominator;
    let mut out = String::new();
    if negative && (!integer.is_zero() || !remainder.is_zero()) {
        out.push('-');
    }
    let integer_text = integer.to_string();
    out.push_str(&integer_text);
    let mut significant = if integer.is_zero() {
        0
    } else {
        integer_text.len()
    };
    let mut significant_positions: Vec<usize> = if integer.is_zero() {
        Vec::new()
    } else {
        (out.len() - integer_text.len()..out.len()).collect()
    };
    if significant > digits || remainder.is_zero() {
        round_to_significant_digits(&mut out, &significant_positions, digits);
        return out;
    }
    out.push('.');
    while !remainder.is_zero() && significant < digits.saturating_add(1) {
        remainder *= 10;
        let digit = &remainder / &denominator;
        remainder %= &denominator;
        let digit = digit.to_u8().unwrap_or(0);
        out.push(char::from(b'0' + digit));
        if digit != 0 || significant > 0 || !integer.is_zero() {
            significant_positions.push(out.len() - 1);
            significant += 1;
        }
    }
    round_to_significant_digits(&mut out, &significant_positions, digits);
    out
}

fn round_to_significant_digits(text: &mut String, significant_positions: &[usize], digits: usize) {
    if significant_positions.len() <= digits {
        return;
    }
    let guard = significant_positions[digits];
    let round_up = text.as_bytes()[guard] >= b'5';
    let integer_end = text.find('.').unwrap_or(text.len());
    let integer_zeros = integer_end.saturating_sub(guard);
    text.truncate(guard);
    if !round_up {
        text.extend(std::iter::repeat_n('0', integer_zeros));
        trim_trailing_decimal_zeros(text);
        return;
    }
    for &pos in significant_positions[..digits].iter().rev() {
        let digit = text.as_bytes()[pos];
        if digit < b'9' {
            text.replace_range(pos..=pos, &char::from(digit + 1).to_string());
            text.extend(std::iter::repeat_n('0', integer_zeros));
            trim_trailing_decimal_zeros(text);
            return;
        }
        text.replace_range(pos..=pos, "0");
    }
    let insert_at = if text.starts_with('-') { 1 } else { 0 };
    let unsigned = &text[insert_at..];
    if unsigned.starts_with("0.") {
        let decimal = insert_at + 1;
        let first_significant = significant_positions[0];
        text.truncate(insert_at);
        if first_significant == decimal + 1 {
            text.push('1');
        } else {
            text.push_str("0.");
            for _ in 0..first_significant.saturating_sub(decimal + 2) {
                text.push('0');
            }
            text.push('1');
        }
        return;
    }
    text.insert(insert_at, '1');
    text.extend(std::iter::repeat_n('0', integer_zeros));
    trim_trailing_decimal_zeros(text);
}

fn scientific_to_decimal(scientific: &str) -> String {
    let Some((mantissa, exponent)) = scientific.split_once('e') else {
        return scientific.to_string();
    };
    let exponent = exponent.parse::<i32>().unwrap_or(0);
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.trim_start_matches('-').trim_start_matches('+');
    let mut digits = mantissa.replace('.', "");
    while digits.starts_with('0') && digits.len() > 1 {
        digits.remove(0);
    }

    let decimal_pos = 1i32 + exponent;
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if decimal_pos <= 0 {
        out.push_str("0.");
        for _ in 0..decimal_pos.unsigned_abs() {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decimal_pos as usize >= digits.len() {
        out.push_str(&digits);
        for _ in 0..(decimal_pos as usize - digits.len()) {
            out.push('0');
        }
    } else {
        let split = decimal_pos as usize;
        out.push_str(&digits[..split]);
        out.push('.');
        out.push_str(&digits[split..]);
    }
    out
}

fn trim_trailing_decimal_zeros(text: &mut String) {
    if let Some(dot) = text.find('.') {
        while text.len() > dot + 1 && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
}
