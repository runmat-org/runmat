use super::*;

#[derive(Debug, Clone)]
pub(super) struct TextscanOptions {
    pub(super) delimiters: Vec<String>,
    pub(super) whitespace: String,
    pub(super) multiple_delims_as_one: bool,
    pub(super) header_lines: usize,
    pub(super) treat_as_empty: Vec<String>,
    pub(super) comment_style: CommentStyle,
    pub(super) collect_output: bool,
    pub(super) return_on_error: bool,
}

impl Default for TextscanOptions {
    fn default() -> Self {
        Self {
            delimiters: Vec::new(),
            whitespace: " \u{0008}\t".to_string(),
            multiple_delims_as_one: false,
            header_lines: 0,
            treat_as_empty: Vec::new(),
            comment_style: CommentStyle::None,
            collect_output: false,
            return_on_error: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CommentStyle {
    None,
    Line(Vec<String>),
    Block { start: String, end: String },
}

pub(super) fn parse_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(Option<usize>, TextscanOptions)> {
    let mut idx = 0usize;
    let repeat = if let Some(value) = args.first() {
        if is_numeric_scalar(value) {
            idx = 1;
            Some(nonnegative_usize(value, "repeat count")?)
        } else {
            None
        }
    } else {
        None
    };
    if !(args.len() - idx).is_multiple_of(2) {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            "textscan: options must be provided as name-value pairs",
        ));
    }
    let mut options = TextscanOptions::default();
    while idx < args.len() {
        let name = string_scalar(&args[idx], "option name")?;
        let value = &args[idx + 1];
        apply_option(&mut options, &name, value)?;
        idx += 2;
    }
    Ok((repeat, options))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextscanOptionName {
    Delimiter,
    MultipleDelimsAsOne,
    HeaderLines,
    TreatAsEmpty,
    CommentStyle,
    CollectOutput,
    ReturnOnError,
    Whitespace,
    Unimplemented,
}

impl TextscanOptionName {
    fn decode(name: &str) -> Option<Self> {
        Some(match normalize_name(name).as_str() {
            "delimiter" => Self::Delimiter,
            "multipledelimsasone" => Self::MultipleDelimsAsOne,
            "headerlines" => Self::HeaderLines,
            "treatasempty" => Self::TreatAsEmpty,
            "commentstyle" => Self::CommentStyle,
            "collectoutput" => Self::CollectOutput,
            "returnonerror" => Self::ReturnOnError,
            "whitespace" => Self::Whitespace,
            "emptyvalue" | "endofline" | "bufsize" | "expchars" | "texttype" | "datelocale" => {
                Self::Unimplemented
            }
            _ => return None,
        })
    }
}

pub(super) fn preflight_host_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Option<(Option<usize>, TextscanOptions)>> {
    if !args.iter().any(crate::dispatcher::value_contains_gpu) {
        return parse_args(args).map(Some);
    }
    let Some(first) = args.first() else {
        return parse_args(args).map(Some);
    };
    if crate::dispatcher::value_contains_gpu(first) {
        if !matches!(first, Value::GpuTensor(_)) {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                "textscan: option name must be scalar text",
            ));
        }
        return Ok(None);
    }
    let start = if is_numeric_scalar(first) {
        nonnegative_usize(first, "repeat count")?;
        1
    } else {
        0
    };
    if !(args.len() - start).is_multiple_of(2) {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            "textscan: options must be provided as name-value pairs",
        ));
    }
    let mut checked = TextscanOptions::default();
    for pair in args.range(start..)?.chunks_exact(2) {
        if !matches!(
            pair[0],
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        ) {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                "textscan: option name must be scalar text",
            ));
        }
        if crate::dispatcher::value_contains_gpu(&pair[0]) {
            continue;
        }
        let name = string_scalar(&pair[0], "option name")?;
        let option = TextscanOptionName::decode(&name);
        match option {
            Some(TextscanOptionName::Unimplemented) => {
                return Err(textscan_error_with(
                    &TEXTSCAN_ERROR_ARGUMENT,
                    format!("textscan: option '{name}' is not implemented yet"),
                ))
            }
            None => {
                return Err(textscan_error_with(
                    &TEXTSCAN_ERROR_ARGUMENT,
                    format!("textscan: unsupported option '{}'", normalize_name(&name)),
                ))
            }
            Some(_) => {}
        }
        if !crate::dispatcher::value_contains_gpu(&pair[1]) {
            apply_option(&mut checked, &name, &pair[1])?;
        }
    }
    Ok(None)
}

fn apply_option(options: &mut TextscanOptions, name: &str, value: &Value) -> BuiltinResult<()> {
    match TextscanOptionName::decode(name) {
        Some(TextscanOptionName::Delimiter) => options.delimiters = delimiter_list(value)?,
        Some(TextscanOptionName::MultipleDelimsAsOne) => {
            options.multiple_delims_as_one = bool_like(value, name)?
        }
        Some(TextscanOptionName::HeaderLines) => {
            options.header_lines = nonnegative_usize(value, name)?
        }
        Some(TextscanOptionName::TreatAsEmpty) => {
            options.treat_as_empty = string_list(value, name)?
        }
        Some(TextscanOptionName::CommentStyle) => {
            options.comment_style = parse_comment_style(value)?
        }
        Some(TextscanOptionName::CollectOutput) => options.collect_output = bool_like(value, name)?,
        Some(TextscanOptionName::ReturnOnError) => {
            options.return_on_error = bool_like(value, name)?
        }
        Some(TextscanOptionName::Whitespace) => options.whitespace = string_scalar(value, name)?,
        Some(TextscanOptionName::Unimplemented) => {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                format!("textscan: option '{name}' is not implemented yet"),
            ));
        }
        None => {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                format!("textscan: unsupported option '{}'", normalize_name(name)),
            ));
        }
    }
    Ok(())
}

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|ch| *ch != '_' && *ch != ' ')
        .flat_map(char::to_lowercase)
        .collect()
}

pub(super) fn string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            format!("textscan: expected {context} as a string scalar or character vector"),
        )
    })
}

fn string_list(value: &Value, context: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data().len());
            for row in 0..cell.rows {
                for col in 0..cell.cols {
                    out.push(string_scalar(
                        &cell.get(row, col).map_err(|err| {
                            textscan_error_with(
                                &TEXTSCAN_ERROR_ARGUMENT,
                                format!("textscan: {err}"),
                            )
                        })?,
                        context,
                    )?);
                }
            }
            Ok(out)
        }
        Value::StringArray(sa) if sa.data().iter().all(|text| !text.is_missing()) => Ok(sa
            .data()
            .iter()
            .map(|text| text.as_str().expect("checked nonmissing string").to_owned())
            .collect()),
        _ => Ok(vec![string_scalar(value, context)?]),
    }
}

fn delimiter_list(value: &Value) -> BuiltinResult<Vec<String>> {
    let mut delimiters = string_list(value, "Delimiter")?;
    for delimiter in &mut delimiters {
        *delimiter = match delimiter.as_str() {
            "\\t" => "\t".to_string(),
            "\\n" => "\n".to_string(),
            "\\r" => "\r".to_string(),
            other => other.to_string(),
        };
        if delimiter.is_empty() {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                "textscan: Delimiter entries must not be empty",
            ));
        }
    }
    Ok(delimiters)
}

fn parse_comment_style(value: &Value) -> BuiltinResult<CommentStyle> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) if s.eq_ignore_ascii_case("none") => {
            Ok(CommentStyle::None)
        }
        Value::CharArray(_) => {
            let text = string_scalar(value, "CommentStyle")?;
            if text.eq_ignore_ascii_case("none") {
                Ok(CommentStyle::None)
            } else {
                Ok(CommentStyle::Line(vec![text]))
            }
        }
        Value::Cell(cell) if cell.data().len() == 2 => {
            let first = string_scalar(
                &cell.get(0, 0).map_err(|err| {
                    textscan_error_with(&TEXTSCAN_ERROR_ARGUMENT, format!("textscan: {err}"))
                })?,
                "CommentStyle",
            )?;
            let second = if cell.rows == 1 {
                string_scalar(
                    &cell.get(0, 1).map_err(|err| {
                        textscan_error_with(&TEXTSCAN_ERROR_ARGUMENT, format!("textscan: {err}"))
                    })?,
                    "CommentStyle",
                )?
            } else {
                string_scalar(
                    &cell.get(1, 0).map_err(|err| {
                        textscan_error_with(&TEXTSCAN_ERROR_ARGUMENT, format!("textscan: {err}"))
                    })?,
                    "CommentStyle",
                )?
            };
            Ok(CommentStyle::Block {
                start: first,
                end: second,
            })
        }
        Value::Cell(cell) => {
            let mut markers = Vec::new();
            for row in 0..cell.rows {
                for col in 0..cell.cols {
                    markers.push(string_scalar(
                        &cell.get(row, col).map_err(|err| {
                            textscan_error_with(
                                &TEXTSCAN_ERROR_ARGUMENT,
                                format!("textscan: {err}"),
                            )
                        })?,
                        "CommentStyle",
                    )?);
                }
            }
            Ok(CommentStyle::Line(markers))
        }
        _ => {
            let text = string_scalar(value, "CommentStyle")?;
            if text.eq_ignore_ascii_case("none") {
                Ok(CommentStyle::None)
            } else {
                Ok(CommentStyle::Line(vec![text]))
            }
        }
    }
}

fn bool_like(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if (*value - 0.0).abs() < f64::EPSILON => Ok(false),
        Value::Num(value) if (*value - 1.0).abs() < f64::EPSILON => Ok(true),
        Value::Int(value) if value.to_i64() == 0 => Ok(false),
        Value::Int(value) if value.to_i64() == 1 => Ok(true),
        _ => match string_scalar(value, context)?
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "true" | "on" | "yes" | "1" => Ok(true),
            "false" | "off" | "no" | "0" => Ok(false),
            _ => Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                format!("textscan: {context} must be logical"),
            )),
        },
    }
}

pub(super) fn nonnegative_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer.try_to_usize().ok_or_else(|| {
            textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                format!("textscan: {context} must be a nonnegative integer scalar"),
            )
        });
    }

    let raw = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        _ => {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                format!("textscan: {context} must be a nonnegative integer scalar"),
            ));
        }
    };
    if !raw.is_finite() || raw < 0.0 || raw.fract() != 0.0 {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            format!("textscan: {context} must be a nonnegative integer scalar"),
        ));
    }
    if raw > usize::MAX.saturating_sub(1) as f64 {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            format!("textscan: {context} is too large"),
        ));
    }
    let parsed = raw.round() as usize;
    if parsed as f64 != raw || parsed == usize::MAX {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            format!("textscan: {context} is too large"),
        ));
    }
    Ok(parsed)
}

pub(super) fn numeric_fid(value: &Value) -> Option<i32> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer.try_to_i32();
    }

    let raw = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        _ => return None,
    };
    if raw.is_finite() && raw.fract() == 0.0 && raw >= i32::MIN as f64 && raw <= i32::MAX as f64 {
        Some(raw as i32)
    } else {
        None
    }
}

fn is_numeric_scalar(value: &Value) -> bool {
    numeric_fid(value).is_some()
}
