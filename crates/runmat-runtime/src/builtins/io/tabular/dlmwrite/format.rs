use super::*;

pub(super) fn format_numeric(value: f64, precision: &PrecisionSpec) -> BuiltinResult<String> {
    if value.is_nan() {
        return Ok("NaN".to_string());
    }
    if value.is_infinite() {
        return Ok(if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        });
    }
    match precision {
        PrecisionSpec::Significant(digits) => {
            if *digits == 0 {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_FORMAT,
                    "dlmwrite: precision must be positive",
                ));
            }
            let fmt = format!("%.{digits}g");
            let mut rendered = c_format(value, &fmt)?;
            if value == 0.0 || rendered == "-0" {
                rendered = "0".to_string();
            }
            Ok(rendered)
        }
        PrecisionSpec::Format(spec) => {
            let rendered = c_format(value, spec)?;
            if value == 0.0 && rendered.starts_with("-") {
                Ok(rendered.trim_start_matches('-').to_string())
            } else {
                Ok(rendered)
            }
        }
    }
}

pub(super) fn format_tensor_value(
    tensor: &Tensor,
    idx: usize,
    options: &DlmWriteOptions,
) -> BuiltinResult<String> {
    let value = tensor
        .numeric_value_at(idx)
        .expect("index within authoritative numeric storage");
    if let Some(value) = value.into_int_value() {
        if !options.precision_explicit {
            return Ok(value.decimal_string());
        }
        return format_numeric(value.to_f64(), &options.precision);
    }
    format_numeric(value.materialize_f64(), &options.precision)
}

pub(super) fn c_format(value: f64, spec: &str) -> BuiltinResult<String> {
    format_float(value, spec)
}

pub(super) fn format_float(value: f64, spec: &str) -> BuiltinResult<String> {
    let parsed = ParsedFormat::parse(spec)?;
    Ok(parsed.render(value))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FloatSpecifier {
    Fixed,
    Exponent,
    General,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SignFlag {
    None,
    Plus,
    Space,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ParsedFormat {
    specifier: FloatSpecifier,
    uppercase: bool,
    alternate: bool,
    sign: SignFlag,
    left_adjust: bool,
    zero_pad: bool,
    width: Option<usize>,
    precision: Option<usize>,
}

impl ParsedFormat {
    pub(super) fn parse(input: &str) -> BuiltinResult<Self> {
        use std::iter::Peekable;
        use std::str::Chars;

        fn parse_number(
            chars: &mut Peekable<Chars<'_>>,
            label: &str,
        ) -> BuiltinResult<Option<usize>> {
            let mut value: usize = 0;
            let mut saw_digit = false;
            while let Some(&ch) = chars.peek() {
                if ch.is_ascii_digit() {
                    saw_digit = true;
                    value = value
                        .checked_mul(10)
                        .and_then(|v| v.checked_add((ch as u8 - b'0') as usize))
                        .ok_or_else(|| {
                            dlmwrite_error_with(
                                &DLMWRITE_ERROR_FORMAT,
                                format!("dlmwrite: {label} too large in precision format"),
                            )
                        })?;
                    chars.next();
                } else {
                    break;
                }
            }
            if saw_digit && value > MAX_NUMERIC_FORMAT_FIELD {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_FORMAT,
                    format!("dlmwrite: {label} exceeds maximum supported precision format size"),
                ));
            }
            Ok(if saw_digit { Some(value) } else { None })
        }

        let mut chars = input.chars().peekable();
        match chars.next() {
            Some('%') => {}
            _ => {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_FORMAT,
                    "dlmwrite: precision format must start with '%'",
                ));
            }
        }

        let mut left_adjust = false;
        let mut sign = SignFlag::None;
        let mut zero_pad = false;
        let mut alternate = false;
        while let Some(&ch) = chars.peek() {
            match ch {
                '-' => {
                    left_adjust = true;
                    zero_pad = false;
                    chars.next();
                }
                '+' => {
                    sign = SignFlag::Plus;
                    chars.next();
                }
                ' ' => {
                    if sign != SignFlag::Plus {
                        sign = SignFlag::Space;
                    }
                    chars.next();
                }
                '0' => {
                    if !left_adjust {
                        zero_pad = true;
                    }
                    chars.next();
                }
                '#' => {
                    alternate = true;
                    chars.next();
                }
                _ => break,
            }
        }

        let width = parse_number(&mut chars, "field width")?;
        let precision = if matches!(chars.peek(), Some('.')) {
            chars.next();
            parse_number(&mut chars, "precision")?.or(Some(0))
        } else {
            None
        };

        if matches!(chars.peek(), Some('l' | 'L' | 'h')) {
            return Err(dlmwrite_error_with(
                &DLMWRITE_ERROR_FORMAT,
                "dlmwrite: length modifiers are not supported in precision formats",
            ));
        }

        let spec_ch = chars.next().ok_or_else(|| {
            dlmwrite_error_with(
                &DLMWRITE_ERROR_FORMAT,
                "dlmwrite: incomplete precision format",
            )
        })?;
        if chars.next().is_some() {
            return Err(dlmwrite_error_with(
                &DLMWRITE_ERROR_FORMAT,
                "dlmwrite: unexpected trailing characters in precision format",
            ));
        }

        let (specifier, uppercase) = match spec_ch {
            'f' => (FloatSpecifier::Fixed, false),
            'F' => (FloatSpecifier::Fixed, true),
            'e' => (FloatSpecifier::Exponent, false),
            'E' => (FloatSpecifier::Exponent, true),
            'g' => (FloatSpecifier::General, false),
            'G' => (FloatSpecifier::General, true),
            other => {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_FORMAT,
                    format!("dlmwrite: unsupported precision format specifier '{other}'"),
                ));
            }
        };

        Ok(Self {
            specifier,
            uppercase,
            alternate,
            sign,
            left_adjust,
            zero_pad: zero_pad && !left_adjust,
            width,
            precision,
        })
    }

    pub(super) fn render(&self, value: f64) -> String {
        let negative = value.is_sign_negative();
        let magnitude = if negative { -value } else { value };
        let mut body = match self.specifier {
            FloatSpecifier::Fixed => {
                format_fixed_body(magnitude, self.precision.unwrap_or(6), self.alternate)
            }
            FloatSpecifier::Exponent => format_exponential_body(
                magnitude,
                self.precision.unwrap_or(6),
                self.alternate,
                self.uppercase,
            ),
            FloatSpecifier::General => format_general_body(
                magnitude,
                self.precision.unwrap_or(6),
                self.alternate,
                self.uppercase,
            ),
        };
        if self.uppercase {
            body.make_ascii_uppercase();
        }

        let mut prefix = String::new();
        if negative {
            prefix.push('-');
        } else {
            match self.sign {
                SignFlag::Plus => prefix.push('+'),
                SignFlag::Space => prefix.push(' '),
                SignFlag::None => {}
            }
        }

        let total_len = prefix.len() + body.len();
        if let Some(width) = self.width {
            if width > total_len {
                let pad = width - total_len;
                if self.left_adjust {
                    let mut result = prefix;
                    result.push_str(&body);
                    result.extend(std::iter::repeat_n(' ', pad));
                    return result;
                } else if self.zero_pad {
                    let mut result = String::with_capacity(width);
                    result.push_str(&prefix);
                    result.extend(std::iter::repeat_n('0', pad));
                    result.push_str(&body);
                    return result;
                } else {
                    let mut result = String::with_capacity(width);
                    result.extend(std::iter::repeat_n(' ', pad));
                    result.push_str(&prefix);
                    result.push_str(&body);
                    return result;
                }
            }
        }

        let mut result = prefix;
        result.push_str(&body);
        result
    }
}

pub(super) fn format_fixed_body(value: f64, precision: usize, alternate: bool) -> String {
    let mut s = format!("{:.*}", precision, value);
    if precision == 0 && alternate && !s.contains('.') {
        s.push('.');
    }
    s
}

pub(super) fn format_exponential_body(
    value: f64,
    precision: usize,
    alternate: bool,
    uppercase: bool,
) -> String {
    let mut s = format!("{:.*e}", precision, value);
    normalize_exponent_notation(&mut s);
    if uppercase {
        s.make_ascii_uppercase();
    }
    if precision == 0 && alternate {
        insert_decimal_point(&mut s);
    }
    s
}

pub(super) fn format_general_body(
    value: f64,
    precision: usize,
    alternate: bool,
    uppercase: bool,
) -> String {
    let effective_precision = precision.max(1);
    let abs_val = value.abs();
    if abs_val == 0.0 {
        return if alternate {
            let mut s = "0.".to_string();
            s.extend(std::iter::repeat_n('0', effective_precision - 1));
            s
        } else {
            "0".to_string()
        };
    }

    let exponent = abs_val.log10().floor() as i32;
    let use_exponent = exponent < -4 || exponent >= effective_precision as i32;
    let mut s = if use_exponent {
        let frac = effective_precision.saturating_sub(1);
        let mut out = format!("{:.*e}", frac, abs_val);
        normalize_exponent_notation(&mut out);
        if uppercase {
            out.make_ascii_uppercase();
        }
        out
    } else {
        let frac = {
            let diff = effective_precision as isize - (exponent + 1) as isize;
            if diff < 0 {
                0
            } else {
                diff as usize
            }
        };
        let mut out = format!("{:.*}", frac, abs_val);
        if uppercase {
            out.make_ascii_uppercase();
        }
        out
    };

    if alternate {
        insert_decimal_point(&mut s);
    } else {
        trim_trailing_zeros(&mut s);
    }
    s
}

pub(super) fn insert_decimal_point(s: &mut String) {
    if s.contains('.') {
        return;
    }
    if let Some(idx) = find_exponent_index(s) {
        s.insert(idx, '.');
    } else {
        s.push('.');
    }
}

pub(super) fn trim_trailing_zeros(s: &mut String) {
    if let Some(idx) = find_exponent_index(s) {
        let exponent = s[idx..].to_string();
        let mut mantissa = s[..idx].to_string();
        trim_fraction(&mut mantissa);
        s.clear();
        s.push_str(&mantissa);
        s.push_str(&exponent);
        normalize_exponent_notation(s);
    } else {
        trim_fraction(s);
    }
}

pub(super) fn trim_fraction(s: &mut String) {
    if let Some(dot_idx) = s.find('.') {
        let mut idx = s.len();
        while idx > dot_idx + 1 && matches!(s.as_bytes().get(idx - 1), Some(b'0')) {
            idx -= 1;
        }
        if idx == dot_idx + 1 {
            idx -= 1;
        }
        s.truncate(idx);
    }
}

pub(super) fn find_exponent_index(s: &str) -> Option<usize> {
    s.find('e').or_else(|| s.find('E'))
}

pub(super) fn normalize_exponent_notation(s: &mut String) {
    if let Some(idx) = find_exponent_index(s) {
        let marker = s.as_bytes()[idx] as char;
        let suffix = &s[idx + 1..];
        let (sign, digits) = if let Some(first) = suffix.chars().next() {
            if first == '+' || first == '-' {
                (first, suffix.get(1..).unwrap_or("").to_string())
            } else {
                ('+', suffix.to_string())
            }
        } else {
            ('+', String::from("0"))
        };
        let mut normalized_digits = if digits.is_empty() {
            String::from("0")
        } else {
            digits
        };
        if normalized_digits.is_empty() {
            normalized_digits.push('0');
        }
        if normalized_digits.len() < 2 {
            normalized_digits = format!("{:0>2}", normalized_digits);
        }
        let mut rebuilt = String::with_capacity(idx + 1 + 1 + normalized_digits.len());
        rebuilt.push_str(&s[..idx]);
        rebuilt.push(marker);
        rebuilt.push(sign);
        rebuilt.push_str(&normalized_digits);
        *s = rebuilt;
    }
}
