//! MATLAB text transformation compatibility helpers.

use regex::Regex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ResolveContext, StringArray, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::builtins::strings::core::compat::{
    broadcast_flat_index, broadcast_shape, logical_value, pattern_regex, scalar_text, text_items,
};
use crate::builtins::strings::text_analytics::documents::erase_punctuation_tokenized_document;
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};

const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
}];

const OUT_BOOL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result.",
}];

const IN_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text.",
}];

const IN_TEXT_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional arguments.",
    },
];

const REST: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arg",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Values to append.",
}];

const NO_ERRORS: [BuiltinErrorDescriptor; 0] = [];

macro_rules! descriptor {
    ($name:ident, $label:expr, $inputs:expr, $outputs:expr) => {
        const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: $inputs,
                outputs: $outputs,
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &NO_ERRORS,
        };
    };
}

descriptor!(APPEND_DESCRIPTOR, "s = append(arg1, ...)", &REST, &OUT_ANY);
descriptor!(REVERSE_DESCRIPTOR, "s = reverse(text)", &IN_TEXT, &OUT_ANY);
descriptor!(DEBLANK_DESCRIPTOR, "s = deblank(text)", &IN_TEXT, &OUT_ANY);
descriptor!(
    STRJUST_DESCRIPTOR,
    "s = strjust(text, side)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    SPLITLINES_DESCRIPTOR,
    "s = splitlines(text)",
    &IN_TEXT,
    &OUT_ANY
);
descriptor!(
    EXTRACT_BEFORE_DESCRIPTOR,
    "s = extractBefore(text, boundary)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    EXTRACT_AFTER_DESCRIPTOR,
    "s = extractAfter(text, boundary)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    INSERT_BEFORE_DESCRIPTOR,
    "s = insertBefore(text, boundary, newText)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    INSERT_AFTER_DESCRIPTOR,
    "s = insertAfter(text, boundary, newText)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    REPLACE_BETWEEN_DESCRIPTOR,
    "s = replaceBetween(text, start, end, newText)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    ERASE_URLS_DESCRIPTOR,
    "s = eraseURLs(text)",
    &IN_TEXT,
    &OUT_ANY
);
descriptor!(
    ERASE_PUNCTUATION_DESCRIPTOR,
    "out = erasePunctuation(textOrDocuments, Name, Value, ...)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    MATCHES_DESCRIPTOR,
    "tf = matches(text, pattern)",
    &IN_TEXT_REST,
    &OUT_BOOL
);

fn any_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

fn bool_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

fn transform_error(name: &str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn map_flow(name: &'static str) -> impl Fn(crate::RuntimeError) -> crate::RuntimeError {
    move |err| map_control_flow_with_builtin(err, name)
}

#[runtime_builtin(
    name = "append",
    category = "strings/transform",
    summary = "Append text inputs elementwise.",
    keywords = "append,string,char,text,concatenate",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::APPEND_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn append_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Ok(Value::String(String::new()));
    }
    let mut lists = Vec::with_capacity(rest.len());
    for value in rest {
        let value = gather_if_needed_async(&value)
            .await
            .map_err(map_flow("append"))?;
        lists.push(text_items(value, "append")?);
    }
    let shape = lists
        .iter()
        .skip(1)
        .try_fold(lists[0].shape.clone(), |shape, list| {
            broadcast_shape(&shape, &list.shape, "append")
        })?;
    let total: usize = shape.iter().product();
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let mut text = String::new();
        for list in &lists {
            let source = broadcast_flat_index(idx, &shape, &list.shape);
            if let Some(part) = &list.items[source] {
                text.push_str(part);
            }
        }
        out.push(text);
    }
    string_array_or_scalar(out, shape, "append")
}

#[runtime_builtin(
    name = "reverse",
    category = "strings/transform",
    summary = "Reverse characters in text values.",
    keywords = "reverse,string,char,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::REVERSE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn reverse_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("reverse"))?;
    map_text_preserve(text, "reverse", |s| s.chars().rev().collect())
}

#[runtime_builtin(
    name = "deblank",
    category = "strings/transform",
    summary = "Remove trailing whitespace from text.",
    keywords = "deblank,trailing whitespace,string,char,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::DEBLANK_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn deblank_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("deblank"))?;
    map_text_preserve(text, "deblank", |s| {
        s.trim_end_matches(char::is_whitespace).to_string()
    })
}

#[runtime_builtin(
    name = "strjust",
    category = "strings/transform",
    summary = "Justify character rows left, right, or center.",
    keywords = "strjust,justify,char,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::STRJUST_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn strjust_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("strjust"))?;
    let side = if let Some(value) = rest.first() {
        let value = gather_if_needed_async(value)
            .await
            .map_err(map_flow("strjust"))?;
        scalar_text(&value, "strjust")?.to_ascii_lowercase()
    } else {
        "right".to_string()
    };
    map_text_preserve(text, "strjust", |s| justify(s, &side))
}

#[runtime_builtin(
    name = "splitlines",
    category = "strings/transform",
    summary = "Split text at newline boundaries.",
    keywords = "splitlines,newline,string,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::SPLITLINES_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn splitlines_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("splitlines"))?;
    match text {
        Value::String(text) => strings_from_lines(&text),
        Value::StringArray(array) if array.data.len() == 1 => strings_from_lines(&array.data[0]),
        Value::CharArray(array) if array.rows <= 1 => {
            strings_from_lines(&char_row_to_string_slice(&array.data, array.cols, 0))
        }
        other => {
            let list = text_items(other, "splitlines")?;
            let values = list
                .items
                .into_iter()
                .map(|text| strings_from_lines(&text.unwrap_or_default()))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, list.shape).map_err(|e| transform_error("splitlines", e))
        }
    }
}

#[runtime_builtin(
    name = "extractBefore",
    category = "strings/transform",
    summary = "Extract text before a position or boundary.",
    keywords = "extractBefore,string,text,boundary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::EXTRACT_BEFORE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn extract_before_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    boundary_transform(text, rest, "extractBefore", |s, boundary| {
        let (start, _) = locate_boundary(s, boundary)?;
        Ok(s[..start].to_string())
    })
    .await
}

#[runtime_builtin(
    name = "extractAfter",
    category = "strings/transform",
    summary = "Extract text after a position or boundary.",
    keywords = "extractAfter,string,text,boundary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::EXTRACT_AFTER_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn extract_after_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    boundary_transform(text, rest, "extractAfter", |s, boundary| {
        let (_, end) = locate_boundary(s, boundary)?;
        Ok(s[end..].to_string())
    })
    .await
}

#[runtime_builtin(
    name = "insertBefore",
    category = "strings/transform",
    summary = "Insert text before a position or boundary.",
    keywords = "insertBefore,string,text,boundary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::INSERT_BEFORE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn insert_before_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    insert_transform(text, rest, "insertBefore", false).await
}

#[runtime_builtin(
    name = "insertAfter",
    category = "strings/transform",
    summary = "Insert text after a position or boundary.",
    keywords = "insertAfter,string,text,boundary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::INSERT_AFTER_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn insert_after_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    insert_transform(text, rest, "insertAfter", true).await
}

#[runtime_builtin(
    name = "replaceBetween",
    category = "strings/transform",
    summary = "Replace text between positions or boundary markers.",
    keywords = "replaceBetween,string,text,boundary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::REPLACE_BETWEEN_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn replace_between_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() < 3 {
        return Err(transform_error(
            "replaceBetween",
            "replaceBetween: expected start, end, and replacement text",
        ));
    }
    let start = gather_if_needed_async(&rest[0])
        .await
        .map_err(map_flow("replaceBetween"))?;
    let stop = gather_if_needed_async(&rest[1])
        .await
        .map_err(map_flow("replaceBetween"))?;
    let replacement = gather_if_needed_async(&rest[2])
        .await
        .map_err(map_flow("replaceBetween"))?;
    let replacement = scalar_text(&replacement, "replaceBetween")?;
    let start = Boundary::from_value(&start, "replaceBetween")?;
    let stop = Boundary::from_value(&stop, "replaceBetween")?;
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("replaceBetween"))?;
    map_text_try_preserve(text, "replaceBetween", |s| {
        let (replace_start, replace_end) = replacement_span_between(s, &start, &stop)?;
        Ok(format!(
            "{}{}{}",
            &s[..replace_start],
            replacement,
            &s[replace_end..]
        ))
    })
}

#[runtime_builtin(
    name = "eraseURLs",
    category = "strings/transform",
    summary = "Remove URL substrings from text.",
    keywords = "eraseURLs,url,text analytics,string",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::ERASE_URLS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn erase_urls_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("eraseURLs"))?;
    let regex = Regex::new(r"https?://[^\s]+|www\.[^\s]+")
        .map_err(|e| transform_error("eraseURLs", e.to_string()))?;
    map_text_preserve(text, "eraseURLs", |s| regex.replace_all(s, "").to_string())
}

#[runtime_builtin(
    name = "erasePunctuation",
    category = "strings/transform",
    summary = "Remove Unicode punctuation and symbol characters from text.",
    keywords = "erasePunctuation,punctuation,symbol,text analytics,string",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::ERASE_PUNCTUATION_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn erase_punctuation_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("erasePunctuation"))?;
    if let Value::Object(object) = text {
        let mut gathered = Vec::with_capacity(rest.len());
        for value in rest {
            gathered.push(
                gather_if_needed_async(&value)
                    .await
                    .map_err(map_flow("erasePunctuation"))?,
            );
        }
        return erase_punctuation_tokenized_document(object, gathered);
    }
    if !rest.is_empty() {
        return Err(transform_error(
            "erasePunctuation",
            "erasePunctuation: name-value options are only supported for tokenizedDocument input",
        ));
    }
    let regex = Regex::new(r"[\p{P}\p{S}]")
        .map_err(|e| transform_error("erasePunctuation", e.to_string()))?;
    map_text_preserve(text, "erasePunctuation", |s| {
        regex.replace_all(s, "").to_string()
    })
}

#[runtime_builtin(
    name = "matches",
    category = "strings/search",
    summary = "Return true when text fully matches a pattern.",
    keywords = "matches,string pattern,text,regular expression",
    accel = "sink",
    type_resolver(bool_type),
    descriptor(crate::builtins::strings::transform::compat::MATCHES_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn matches_builtin(text: Value, pattern: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("matches"))?;
    let pattern = gather_if_needed_async(&pattern)
        .await
        .map_err(map_flow("matches"))?;
    let pattern = pattern_regex(&pattern, "matches")?;
    let regex = Regex::new(&format!("^(?:{pattern})$"))
        .map_err(|e| transform_error("matches", e.to_string()))?;
    let list = text_items(text, "matches")?;
    let out = list
        .items
        .iter()
        .map(|text| u8::from(text.as_ref().is_some_and(|text| regex.is_match(text))))
        .collect::<Vec<_>>();
    logical_value(out, list.shape, "matches")
}

async fn boundary_transform(
    text: Value,
    rest: Vec<Value>,
    fn_name: &'static str,
    op: impl Fn(&str, &Boundary) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(transform_error(
            fn_name,
            format!("{fn_name}: expected a boundary argument"),
        ));
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow(fn_name))?;
    let boundary = gather_if_needed_async(&rest[0])
        .await
        .map_err(map_flow(fn_name))?;
    let boundary = Boundary::from_value(&boundary, fn_name)?;
    map_text_try_preserve(text, fn_name, |s| op(s, &boundary))
}

async fn insert_transform(
    text: Value,
    rest: Vec<Value>,
    fn_name: &'static str,
    after: bool,
) -> BuiltinResult<Value> {
    if rest.len() < 2 {
        return Err(transform_error(
            fn_name,
            format!("{fn_name}: expected boundary and new text"),
        ));
    }
    let boundary = gather_if_needed_async(&rest[0])
        .await
        .map_err(map_flow(fn_name))?;
    let new_text = gather_if_needed_async(&rest[1])
        .await
        .map_err(map_flow(fn_name))?;
    let boundary = Boundary::from_value(&boundary, fn_name)?;
    let new_text = scalar_text(&new_text, fn_name)?;
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow(fn_name))?;
    map_text_try_preserve(text, fn_name, |s| {
        let (start, end) = locate_boundary(s, &boundary)?;
        let idx = if after { end } else { start };
        Ok(format!("{}{}{}", &s[..idx], new_text, &s[idx..]))
    })
}

#[derive(Clone)]
enum Boundary {
    Position(usize),
    Text(String),
    Pattern(String),
}

impl Boundary {
    fn from_value(value: &Value, fn_name: &str) -> BuiltinResult<Self> {
        match value {
            Value::Num(n) if n.is_finite() && *n > 0.0 && n.fract() == 0.0 => {
                Ok(Self::Position(*n as usize))
            }
            Value::Num(_) => Err(transform_error(
                fn_name,
                format!("{fn_name}: numeric boundaries must be positive integer scalars"),
            )),
            Value::Object(_) => Ok(Self::Pattern(pattern_regex(value, fn_name)?)),
            _ => Ok(Self::Text(scalar_text(value, fn_name)?)),
        }
    }
}

fn locate_boundary(text: &str, boundary: &Boundary) -> BuiltinResult<(usize, usize)> {
    match boundary {
        Boundary::Position(pos) => {
            let char_len = text.chars().count();
            if *pos > char_len.saturating_add(1) {
                return Err(transform_error(
                    "text boundary",
                    format!("boundary position {pos} exceeds text length {char_len}"),
                ));
            }
            let start = byte_index_for_char_position(text, *pos);
            let end = if *pos > char_len {
                text.len()
            } else {
                byte_index_after_char_position(text, *pos)
            };
            Ok((start, end))
        }
        Boundary::Text(needle) => text
            .find(needle)
            .map(|idx| (idx, idx + needle.len()))
            .ok_or_else(|| transform_error("text boundary", "boundary not found")),
        Boundary::Pattern(pattern) => Regex::new(pattern)
            .map_err(|e| transform_error("text boundary", e.to_string()))?
            .find(text)
            .map(|m| (m.start(), m.end()))
            .ok_or_else(|| transform_error("text boundary", "boundary not found")),
    }
}

fn replacement_span_between(
    text: &str,
    start: &Boundary,
    stop: &Boundary,
) -> BuiltinResult<(usize, usize)> {
    match (start, stop) {
        (Boundary::Position(start), Boundary::Position(stop)) => {
            if start > stop {
                return Err(transform_error(
                    "replaceBetween",
                    "replaceBetween: start position must be less than or equal to end position",
                ));
            }
            let char_len = text.chars().count();
            if *stop > char_len {
                return Err(transform_error(
                    "replaceBetween",
                    format!("replaceBetween: end position {stop} exceeds text length {char_len}"),
                ));
            }
            Ok((
                byte_index_for_char_position(text, *start),
                byte_index_after_char_position(text, *stop),
            ))
        }
        _ => {
            let (_, start_end) = locate_boundary(text, start)?;
            let (stop_start, _) = locate_boundary(&text[start_end..], stop)
                .map(|(a, b)| (a + start_end, b + start_end))?;
            Ok((start_end, stop_start))
        }
    }
}

fn byte_index_for_char_position(text: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    text.char_indices()
        .nth(pos.saturating_sub(1))
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

fn byte_index_after_char_position(text: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    text.char_indices()
        .nth(pos)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

fn map_text_preserve(
    value: Value,
    fn_name: &str,
    map: impl Fn(&str) -> String + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            if is_missing_string(&text) {
                Ok(Value::String(text))
            } else {
                Ok(Value::String(map(&text)))
            }
        }
        Value::StringArray(array) => StringArray::new(
            array
                .data
                .into_iter()
                .map(|text| {
                    if is_missing_string(&text) {
                        text
                    } else {
                        map(&text)
                    }
                })
                .collect(),
            array.shape,
        )
        .map(Value::StringArray)
        .map_err(|e| transform_error(fn_name, e)),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| map(&char_row_to_string_slice(&array.data, array.cols, row)))
                .collect::<Vec<_>>();
            char_rows(rows, fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| map_text_preserve(value, fn_name, map))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape).map_err(|e| transform_error(fn_name, e))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

fn map_text_try_preserve(
    value: Value,
    fn_name: &str,
    map: impl Fn(&str) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            if is_missing_string(&text) {
                Ok(Value::String(text))
            } else {
                Ok(Value::String(map(&text)?))
            }
        }
        Value::StringArray(array) => StringArray::new(
            array
                .data
                .into_iter()
                .map(|text| {
                    if is_missing_string(&text) {
                        Ok(text)
                    } else {
                        map(&text)
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            array.shape,
        )
        .map(Value::StringArray)
        .map_err(|e| transform_error(fn_name, e)),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| map(&char_row_to_string_slice(&array.data, array.cols, row)))
                .collect::<BuiltinResult<Vec<_>>>()?;
            char_rows(rows, fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| map_text_try_preserve(value, fn_name, map))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape).map_err(|e| transform_error(fn_name, e))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

fn char_rows(rows: Vec<String>, fn_name: &str) -> BuiltinResult<Value> {
    let row_count = rows.len();
    let cols = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(row_count * cols);
    for row in rows {
        let mut chars = row.chars().collect::<Vec<_>>();
        chars.resize(cols, ' ');
        data.extend(chars);
    }
    CharArray::new(data, row_count, cols)
        .map(Value::CharArray)
        .map_err(|e| transform_error(fn_name, e))
}

fn string_array_or_scalar(
    values: Vec<String>,
    shape: Vec<usize>,
    fn_name: &str,
) -> BuiltinResult<Value> {
    if values.len() == 1 {
        Ok(Value::String(values.into_iter().next().unwrap()))
    } else {
        StringArray::new(values, shape)
            .map(Value::StringArray)
            .map_err(|e| transform_error(fn_name, e))
    }
}

fn strings_from_lines(text: &str) -> BuiltinResult<Value> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines = normalized
        .split('\n')
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = lines.len();
    StringArray::new(lines, vec![rows, 1])
        .map(Value::StringArray)
        .map_err(|e| transform_error("splitlines", e))
}

fn justify(text: &str, side: &str) -> String {
    let width = text.chars().count();
    let trimmed = text.trim().to_string();
    let pad = width.saturating_sub(trimmed.chars().count());
    match side {
        "left" => format!("{trimmed}{}", " ".repeat(pad)),
        "center" | "centre" => {
            let left = pad / 2;
            let right = pad - left;
            format!("{}{}{}", " ".repeat(left), trimmed, " ".repeat(right))
        }
        _ => format!("{}{}", " ".repeat(pad), trimmed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::CellArray;

    fn block(
        value: impl std::future::Future<Output = BuiltinResult<Value>>,
    ) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn append_extract_insert_and_replace_work() {
        assert_eq!(
            block(append_builtin(vec![
                Value::String("run".into()),
                Value::String("mat".into())
            ]))
            .unwrap(),
            Value::String("runmat".into())
        );
        assert_eq!(
            block(extract_before_builtin(
                Value::String("alpha.beta".into()),
                vec![Value::String(".".into())],
            ))
            .unwrap(),
            Value::String("alpha".into())
        );
        assert_eq!(
            block(insert_after_builtin(
                Value::String("run".into()),
                vec![Value::Num(3.0), Value::String("mat".into())],
            ))
            .unwrap(),
            Value::String("runmat".into())
        );
        assert_eq!(
            block(extract_after_builtin(
                Value::String("alpha.beta".into()),
                vec![Value::String(".".into())],
            ))
            .unwrap(),
            Value::String("beta".into())
        );
        assert_eq!(
            block(insert_before_builtin(
                Value::String("run".into()),
                vec![Value::Num(1.0), Value::String("pre".into())],
            ))
            .unwrap(),
            Value::String("prerun".into())
        );
        assert_eq!(
            block(replace_between_builtin(
                Value::String("a[old]b".into()),
                vec![
                    Value::String("[".into()),
                    Value::String("]".into()),
                    Value::String("new".into()),
                ],
            ))
            .unwrap(),
            Value::String("a[new]b".into())
        );
    }

    #[test]
    fn splitlines_reverse_deblank_and_matches_work() {
        assert_eq!(
            block(reverse_builtin(Value::String("abc".into()))).unwrap(),
            Value::String("cba".into())
        );
        assert_eq!(
            block(deblank_builtin(Value::CharArray(CharArray::new_row(
                "abc   "
            ))))
            .unwrap(),
            Value::CharArray(CharArray::new_row("abc"))
        );
        assert_eq!(
            block(matches_builtin(
                Value::String("A12".into()),
                crate::builtins::strings::core::compat::pattern_object(r"A\d+"),
            ))
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            block(erase_urls_builtin(Value::String(
                "see https://example.com now".into()
            )))
            .unwrap(),
            Value::String("see  now".into())
        );
        assert_eq!(
            block(erase_punctuation_builtin(
                Value::String("it's one and/or two.".into()),
                vec![]
            ))
            .unwrap(),
            Value::String("its one andor two".into())
        );
        assert_eq!(
            block(erase_punctuation_builtin(
                Value::CharArray(CharArray::new_row("cost: $5.00!")),
                vec![]
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("cost 500"))
        );
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("alpha,beta!")),
                Value::String("C++ and C#".into()),
            ],
            1,
            2,
        )
        .unwrap();
        match block(erase_punctuation_builtin(Value::Cell(cell), vec![])).unwrap() {
            Value::Cell(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(
                    out.data[0],
                    Value::CharArray(CharArray::new_row("alphabeta"))
                );
                assert_eq!(out.data[1], Value::String("C and C".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
        assert_eq!(
            block(strjust_builtin(
                Value::CharArray(CharArray::new_row("  x")),
                vec![Value::String("left".into())],
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("x  "))
        );
    }
}
