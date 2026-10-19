use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FormatItem {
    pub(super) kind: FormatKind,
    pub(super) skip: bool,
    pub(super) width: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FormatKind {
    Float(NumericDType),
    SignedInt(NumericDType),
    UnsignedInt(NumericDType),
    RadixInt { dtype: NumericDType, radix: u32 },
    String,
    QuotedString,
    Char,
    CharSet { chars: HashSet<char>, negated: bool },
}

impl FormatKind {
    pub(super) fn output_kind(&self) -> OutputKind {
        match self {
            FormatKind::Float(dtype)
            | FormatKind::SignedInt(dtype)
            | FormatKind::UnsignedInt(dtype)
            | FormatKind::RadixInt { dtype, .. } => OutputKind::Numeric(*dtype),
            FormatKind::String
            | FormatKind::QuotedString
            | FormatKind::Char
            | FormatKind::CharSet { .. } => OutputKind::Text,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OutputKind {
    Numeric(NumericDType),
    Text,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FormatElement {
    Conversion(FormatItem),
    Literal(String),
    Whitespace,
}

pub(super) fn parse_format_spec(format: &str) -> BuiltinResult<Vec<FormatElement>> {
    let mut elements = Vec::new();
    let mut conversion_count = 0usize;
    let chars: Vec<char> = format.chars().collect();
    let mut idx = 0usize;
    while idx < chars.len() {
        if chars[idx] != '%' {
            if chars[idx].is_whitespace() {
                while idx < chars.len() && chars[idx].is_whitespace() {
                    idx += 1;
                }
                if !matches!(elements.last(), Some(FormatElement::Whitespace)) {
                    elements.push(FormatElement::Whitespace);
                }
                continue;
            }
            let start = idx;
            while idx < chars.len() && chars[idx] != '%' && !chars[idx].is_whitespace() {
                idx += 1;
            }
            elements.push(FormatElement::Literal(chars[start..idx].iter().collect()));
            continue;
        }
        idx += 1;
        if idx < chars.len() && chars[idx] == '%' {
            elements.push(FormatElement::Literal("%".to_string()));
            idx += 1;
            continue;
        }
        let mut skip = false;
        if idx < chars.len() && chars[idx] == '*' {
            skip = true;
            idx += 1;
        }
        let width_start = idx;
        while idx < chars.len() && chars[idx].is_ascii_digit() {
            idx += 1;
        }
        let width = if idx > width_start {
            Some(
                chars[width_start..idx]
                    .iter()
                    .collect::<String>()
                    .parse::<usize>()
                    .map_err(|_| {
                        textscan_error_with(&TEXTSCAN_ERROR_FORMAT, "textscan: invalid field width")
                    })?,
            )
        } else {
            None
        };
        if idx >= chars.len() {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_FORMAT,
                "textscan: incomplete conversion specifier",
            ));
        }
        let kind = match chars[idx] {
            'f' | 'e' | 'E' | 'g' | 'G' | 'n' => {
                idx += 1;
                FormatKind::Float(parse_numeric_suffix(
                    &chars,
                    &mut idx,
                    NumericDType::F64,
                    true,
                )?)
            }
            'd' | 'i' => {
                idx += 1;
                FormatKind::SignedInt(parse_numeric_suffix(
                    &chars,
                    &mut idx,
                    NumericDType::I32,
                    false,
                )?)
            }
            'u' => {
                idx += 1;
                FormatKind::UnsignedInt(parse_unsigned_suffix(&chars, &mut idx)?)
            }
            'x' | 'b' => {
                let radix = if chars[idx] == 'x' { 16 } else { 2 };
                idx += 1;
                FormatKind::RadixInt {
                    dtype: parse_radix_suffix(&chars, &mut idx)?,
                    radix,
                }
            }
            's' => {
                idx += 1;
                FormatKind::String
            }
            'q' => {
                idx += 1;
                FormatKind::QuotedString
            }
            'c' => {
                idx += 1;
                FormatKind::Char
            }
            '[' => {
                idx += 1;
                let mut negated = false;
                if idx < chars.len() && chars[idx] == '^' {
                    negated = true;
                    idx += 1;
                }
                let mut set = HashSet::new();
                while idx < chars.len() && chars[idx] != ']' {
                    set.insert(chars[idx]);
                    idx += 1;
                }
                if idx >= chars.len() {
                    return Err(textscan_error_with(
                        &TEXTSCAN_ERROR_FORMAT,
                        "textscan: unterminated character set conversion",
                    ));
                }
                idx += 1;
                FormatKind::CharSet {
                    chars: set,
                    negated,
                }
            }
            other => {
                return Err(textscan_error_with(
                    &TEXTSCAN_ERROR_FORMAT,
                    format!("textscan: unsupported conversion '%{other}'"),
                ));
            }
        };
        elements.push(FormatElement::Conversion(FormatItem { kind, skip, width }));
        conversion_count += 1;
    }
    if conversion_count == 0 {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_FORMAT,
            "textscan: formatSpec must contain at least one conversion",
        ));
    }
    Ok(elements)
}

fn parse_numeric_suffix(
    chars: &[char],
    idx: &mut usize,
    default: NumericDType,
    floating: bool,
) -> BuiltinResult<NumericDType> {
    let start = *idx;
    while *idx < chars.len() && chars[*idx].is_ascii_digit() {
        *idx += 1;
    }
    if start == *idx {
        return Ok(default);
    }
    let suffix: String = chars[start..*idx].iter().collect();
    let dtype = if floating {
        match suffix.as_str() {
            "32" => NumericDType::F32,
            "64" => NumericDType::F64,
            _ => return Err(invalid_numeric_suffix(&suffix)),
        }
    } else {
        match suffix.as_str() {
            "8" => NumericDType::I8,
            "16" => NumericDType::I16,
            "32" => NumericDType::I32,
            "64" => NumericDType::I64,
            _ => return Err(invalid_numeric_suffix(&suffix)),
        }
    };
    Ok(dtype)
}

fn parse_unsigned_suffix(chars: &[char], idx: &mut usize) -> BuiltinResult<NumericDType> {
    Ok(
        match parse_numeric_suffix(chars, idx, NumericDType::I32, false)? {
            NumericDType::I8 => NumericDType::U8,
            NumericDType::I16 => NumericDType::U16,
            NumericDType::I32 => NumericDType::U32,
            NumericDType::I64 => NumericDType::U64,
            _ => unreachable!("integer suffix parser returned floating dtype"),
        },
    )
}

fn parse_radix_suffix(chars: &[char], idx: &mut usize) -> BuiltinResult<NumericDType> {
    let Some(sign) = chars
        .get(*idx)
        .copied()
        .filter(|ch| matches!(ch, 'u' | 's'))
    else {
        return Ok(NumericDType::U64);
    };
    let suffix_start = *idx;
    *idx += 1;
    let digits_start = *idx;
    while *idx < chars.len() && chars[*idx].is_ascii_digit() {
        *idx += 1;
    }
    if digits_start == *idx {
        *idx = suffix_start;
        return Ok(NumericDType::U64);
    }
    let width: String = chars[digits_start..*idx].iter().collect();
    match (sign, width.as_str()) {
        ('u', "8") => Ok(NumericDType::U8),
        ('u', "16") => Ok(NumericDType::U16),
        ('u', "32") => Ok(NumericDType::U32),
        ('u', "64") => Ok(NumericDType::U64),
        ('s', "8") => Ok(NumericDType::I8),
        ('s', "16") => Ok(NumericDType::I16),
        ('s', "32") => Ok(NumericDType::I32),
        ('s', "64") => Ok(NumericDType::I64),
        _ => Err(invalid_numeric_suffix(&format!("{sign}{width}"))),
    }
}

fn invalid_numeric_suffix(suffix: &str) -> RuntimeError {
    textscan_error_with(
        &TEXTSCAN_ERROR_FORMAT,
        format!("textscan: unsupported numeric output width '{suffix}'"),
    )
}
