//! MATLAB text transformation compatibility helpers.

use regex::Regex;

#[cfg(test)]
use super::{
    extractafter::extract_after_builtin, extractbefore::extract_before_builtin,
    insertafter::insert_after_builtin, insertbefore::insert_before_builtin,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ResolveContext, Type, ERASE_PUNCTUATION_BROAD_CELL_EXTENSION,
    ERASE_PUNCTUATION_CHAR_MATRIX_EXTENSION, ERASE_PUNCTUATION_ERROR_INVALID_INPUT,
    ERASE_PUNCTUATION_ERROR_INVALID_OPTION, ERASE_URLS_BROAD_CELL_EXTENSION,
    ERASE_URLS_CHAR_MATRIX_EXTENSION, ERASE_URLS_ERROR_INVALID_INPUT, EXTRACT_AFTER_ERRORS,
    EXTRACT_BEFORE_ERRORS, INSERT_AFTER_BROAD_TEXT_EXTENSION,
    INSERT_AFTER_RESIDENT_POSITION_EXTENSION, INSERT_BEFORE_BROAD_TEXT_EXTENSION,
    INSERT_BEFORE_RESIDENT_POSITION_EXTENSION, INSERT_ERRORS,
};
#[cfg(test)]
use runmat_builtins::{
    EXTRACT_AFTER_DESCRIPTOR, EXTRACT_AFTER_RESIDENT_POSITION_EXTENSION,
    EXTRACT_BEFORE_CHAR_MATRIX_EXTENSION, EXTRACT_BEFORE_DESCRIPTOR,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, IntValue, NumericScalar, StringArray, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::common::{
    char_row_to_string_slice, contains_numeric_or_resident_text_input, is_missing_string,
};
use crate::builtins::strings::core::compat::{
    broadcast_flat_index, broadcast_shape, logical_value, pattern_regex, scalar_text, text_items,
};
use crate::builtins::strings::text_analytics::documents::{
    erase_punctuation_tokenized_document, erase_urls_tokenized_document,
};
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

descriptor!(REVERSE_DESCRIPTOR, "s = reverse(text)", &IN_TEXT, &OUT_ANY);
pub const REVERSE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "reverse accepts string arrays, character vectors, or cell arrays of character vectors. Numeric and integer inputs reject without implicit text conversion or provider access.",
};
descriptor!(
    STRJUST_DESCRIPTOR,
    "s = strjust(text, side)",
    &IN_TEXT_REST,
    &OUT_ANY
);
pub const STRJUST_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "strjust operates on character or string text and an optional textual alignment. Integer, numeric, and provider-resident inputs reject before provider access and are not treated as character codes.",
};
descriptor!(
    SPLITLINES_DESCRIPTOR,
    "s = splitlines(text)",
    &IN_TEXT,
    &OUT_ANY
);
pub const SPLITLINES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "splitlines accepts host string, character, or cell text and returns a string or cell text container. Numeric, logical, symbolic, and provider-resident values reject before gather or provider access.",
    };
descriptor!(
    REPLACE_BETWEEN_DESCRIPTOR,
    "s = replaceBetween(text, start, end, newText)",
    &IN_TEXT_REST,
    &OUT_ANY
);
const REPLACE_BETWEEN_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "replacebetween-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "replaceBetween host fallback for explicit gpuArray input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReplaceBetweenExplicitGpuInputExtension"),
    };
const REPLACE_BETWEEN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [REPLACE_BETWEEN_EXPLICIT_GPU_EXTENSION];
const REPLACE_BETWEEN_POSITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "startPos/endPos",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented numeric positions accept all eight integer classes and are read exactly as one-based inclusive boundaries.",
    }];
const REPLACE_BETWEEN_TEXT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "str/newText",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Source and replacement text reject integer and resident numeric values before gather or implicit formatting.",
    }];
pub const REPLACE_BETWEEN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "newStr = replaceBetween(str, integer_startPos, integer_endPos, newText)",
        inputs: &REPLACE_BETWEEN_POSITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Positions remain exact through range and ordering validation; output preserves the host text container. Automatic residency gathers transparently, while explicit gpuArray fallback is separately gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "newStr = replaceBetween(integer_str, start, end, integer_newText)",
        inputs: &REPLACE_BETWEEN_TEXT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer data is outside the public text roles and is never converted to decimal text.",
    },
];
descriptor!(
    MATCHES_DESCRIPTOR,
    "tf = matches(text, pattern)",
    &IN_TEXT_REST,
    &OUT_BOOL
);
pub const MATCHES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "matches accepts text or pattern inputs and a logical IgnoreCase option. Numeric and integer data has no documented role and rejects without implicit text conversion or provider access.",
};

fn any_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

fn bool_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

fn transform_error(name: &str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

#[derive(Clone, Copy)]
enum ExtractErrorKind {
    InvalidInput,
    InvalidBoundary,
    SizeMismatch,
}

fn extract_error(
    name: &str,
    kind: ExtractErrorKind,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let descriptor = match (name, kind) {
        ("extractBefore", ExtractErrorKind::InvalidInput) => &EXTRACT_BEFORE_ERRORS[0],
        ("extractBefore", ExtractErrorKind::InvalidBoundary) => &EXTRACT_BEFORE_ERRORS[1],
        ("extractBefore", ExtractErrorKind::SizeMismatch) => &EXTRACT_BEFORE_ERRORS[2],
        ("extractAfter", ExtractErrorKind::InvalidInput) => &EXTRACT_AFTER_ERRORS[0],
        ("extractAfter", ExtractErrorKind::InvalidBoundary) => &EXTRACT_AFTER_ERRORS[1],
        ("extractAfter", ExtractErrorKind::SizeMismatch) => &EXTRACT_AFTER_ERRORS[2],
        _ => return transform_error(name, message),
    };
    let mut builder = build_runtime_error(message).with_builtin(name);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_flow(name: &'static str) -> impl Fn(crate::RuntimeError) -> crate::RuntimeError {
    move |err| map_control_flow_with_builtin(err, name)
}

#[runtime_builtin(
    name = "append",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn append_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Ok(Value::String(String::new()));
    }
    let mut lists = Vec::with_capacity(rest.len());
    let mut output_kind = AppendOutputKind::Char;
    for value in rest {
        if matches!(value, Value::GpuTensor(_)) {
            return Err(transform_error(
                "append",
                "append: expected string, character vector, or cell array of character vectors",
            ));
        }
        let (list, kind) = append_text_items(value)?;
        output_kind = output_kind.combine(kind);
        lists.push(list);
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
    append_output(out, shape, output_kind)
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum AppendOutputKind {
    Char,
    Cell,
    String,
}

impl AppendOutputKind {
    fn combine(self, other: Self) -> Self {
        self.max(other)
    }
}

fn append_text_items(
    value: Value,
) -> BuiltinResult<(
    crate::builtins::strings::core::compat::TextList,
    AppendOutputKind,
)> {
    match value {
        Value::String(text) => Ok((
            text_items(Value::String(text), "append")?,
            AppendOutputKind::String,
        )),
        Value::StringArray(array) => Ok((
            text_items(Value::StringArray(array), "append")?,
            AppendOutputKind::String,
        )),
        Value::Cell(cell) => {
            let mut items = Vec::with_capacity(cell.data.len());
            for value in cell.data {
                match value {
                    Value::CharArray(array) if array.rows <= 1 || array.cols <= 1 => {
                        items.push(Some(array.data.into_iter().collect::<String>()));
                    }
                    _ => {
                        return Err(transform_error(
                            "append",
                            "append: cell inputs must contain only character vectors",
                        ));
                    }
                }
            }
            Ok((
                crate::builtins::strings::core::compat::TextList {
                    items,
                    shape: cell.shape,
                },
                AppendOutputKind::Cell,
            ))
        }
        Value::CharArray(array) if array.rows <= 1 || array.cols <= 1 => {
            let text = array.data.into_iter().collect::<String>();
            Ok((
                text_items(Value::String(text), "append")?,
                AppendOutputKind::Char,
            ))
        }
        Value::CharArray(_) => Err(transform_error(
            "append",
            "append: character inputs must be vectors",
        )),
        other => Err(transform_error(
            "append",
            format!(
                "append: expected string, character vector, or cell array of character vectors, got {other:?}"
            ),
        )),
    }
}

fn append_output(
    values: Vec<String>,
    shape: Vec<usize>,
    kind: AppendOutputKind,
) -> BuiltinResult<Value> {
    match kind {
        AppendOutputKind::String => string_array_or_scalar(values, shape, "append"),
        AppendOutputKind::Cell => {
            let values = values
                .into_iter()
                .map(|text| char_rows(vec![text], "append"))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, shape).map_err(|error| transform_error("append", error))
        }
        AppendOutputKind::Char => {
            let text = values.into_iter().next().unwrap_or_default();
            char_rows(vec![text], "append")
        }
    }
}

#[runtime_builtin(
    name = "reverse",
    category = "strings/transform",
    summary = "Reverse characters in text values.",
    keywords = "reverse,string,char,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::REVERSE_DESCRIPTOR),
    integer_audit(crate::builtins::strings::transform::compat::REVERSE_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn reverse_builtin(text: Value) -> BuiltinResult<Value> {
    if crate::dispatcher::value_contains_gpu(&text) {
        return Err(transform_error(
            "reverse",
            "reverse: expected string, character array, or cell array of character vectors",
        ));
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("reverse"))?;
    map_text_preserve(text, "reverse", |s| s.chars().rev().collect())
}

#[runtime_builtin(
    name = "deblank",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn deblank_builtin(text: Value) -> BuiltinResult<Value> {
    if crate::dispatcher::value_contains_gpu(&text) {
        return Err(transform_error(
            "deblank",
            "deblank: expected string, character array, or cell array of character vectors",
        ));
    }
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
    integer_audit(crate::builtins::strings::transform::compat::STRJUST_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn strjust_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if contains_numeric_or_resident_text_input(&text)
        || rest
            .first()
            .is_some_and(contains_numeric_or_resident_text_input)
    {
        return Err(transform_error(
            "strjust",
            "strjust: expected host text and a textual alignment",
        ));
    }
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
    integer_audit(crate::builtins::strings::transform::compat::SPLITLINES_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn splitlines_builtin(text: Value) -> BuiltinResult<Value> {
    if contains_numeric_or_resident_text_input(&text) {
        return Err(transform_error(
            "splitlines",
            "splitlines: expected host string, character array, or cell array of character vectors",
        ));
    }
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
    name = "replaceBetween",
    category = "strings/transform",
    summary = "Replace text between positions or boundary markers.",
    keywords = "replaceBetween,string,text,boundary",
    accel = "sink",
    extensions(REPLACE_BETWEEN_EXTENSIONS),
    integer_capabilities(REPLACE_BETWEEN_INTEGER_CAPABILITIES),
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
    if contains_numeric_or_resident_text_input(&text)
        || contains_numeric_or_resident_text_input(&rest[2])
    {
        return Err(transform_error(
            "replaceBetween",
            "replaceBetween: source and replacement must be text",
        ));
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&text)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &REPLACE_BETWEEN_EXPLICIT_GPU_EXTENSION,
            "replaceBetween",
        )?;
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
    let replacements = TextList::from_value(&replacement, "replaceBetween")?;
    let starts = BoundaryList::from_value(&start, "replaceBetween")?;
    let stops = BoundaryList::from_value(&stop, "replaceBetween")?;
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("replaceBetween"))?;
    let text_shape = boundary_text_shape(&text)
        .ok_or_else(|| transform_error("replaceBetween", "replaceBetween: expected text input"))?;
    if !shape_is_scalar_or_same(&starts.shape, &text_shape)
        || !shape_is_scalar_or_same(&stops.shape, &text_shape)
        || !shape_is_scalar_or_same(&replacements.shape, &text_shape)
    {
        return Err(transform_error(
            "replaceBetween",
            "replaceBetween: start, end, and newText must be scalar or match str",
        ));
    }
    map_text_with_replacements(text, &starts, &stops, &replacements)
}

#[runtime_builtin(
    name = "eraseURLs",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn erase_urls_builtin(text: Value) -> BuiltinResult<Value> {
    reject_resident_text_input(&text, "eraseURLs", &ERASE_URLS_ERROR_INVALID_INPUT)?;
    ensure_erase_text_extensions(
        &text,
        "eraseURLs",
        &ERASE_URLS_CHAR_MATRIX_EXTENSION,
        &ERASE_URLS_BROAD_CELL_EXTENSION,
    )?;
    if let Value::Object(object) = text {
        return erase_urls_tokenized_document(object);
    }
    if !is_host_text_tree(&text) {
        return Err(descriptor_transform_error(
            "eraseURLs",
            &ERASE_URLS_ERROR_INVALID_INPUT,
        ));
    }
    let regex = Regex::new(r"(?i)https?://[^\s]+")
        .map_err(|e| transform_error("eraseURLs", e.to_string()))?;
    map_text_preserve(text, "eraseURLs", |s| regex.replace_all(s, "").to_string())
}

#[runtime_builtin(
    name = "erasePunctuation",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn erase_punctuation_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    reject_resident_text_input(
        &text,
        "erasePunctuation",
        &ERASE_PUNCTUATION_ERROR_INVALID_INPUT,
    )?;
    for value in &rest {
        reject_resident_text_input(
            value,
            "erasePunctuation",
            &ERASE_PUNCTUATION_ERROR_INVALID_OPTION,
        )?;
    }
    ensure_erase_text_extensions(
        &text,
        "erasePunctuation",
        &ERASE_PUNCTUATION_CHAR_MATRIX_EXTENSION,
        &ERASE_PUNCTUATION_BROAD_CELL_EXTENSION,
    )?;
    if let Value::Object(object) = text {
        return erase_punctuation_tokenized_document(object, rest);
    }
    if !is_host_text_tree(&text) {
        return Err(descriptor_transform_error(
            "erasePunctuation",
            &ERASE_PUNCTUATION_ERROR_INVALID_INPUT,
        ));
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

fn reject_resident_text_input(
    value: &Value,
    name: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<()> {
    if crate::dispatcher::value_contains_gpu(value) {
        let mut builder = build_runtime_error(error.message).with_builtin(name);
        if let Some(identifier) = error.identifier {
            builder = builder.with_identifier(identifier);
        }
        return Err(builder.build());
    }
    Ok(())
}

fn descriptor_transform_error(
    name: &str,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn is_host_text_tree(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => true,
        Value::Cell(cell) => cell.data.iter().all(is_host_text_tree),
        _ => false,
    }
}

fn ensure_erase_text_extensions(
    value: &Value,
    builtin: &str,
    char_matrix: &'static BuiltinExtensionDescriptor,
    broad_cell: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    match value {
        Value::CharArray(array) if array.rows > 1 && array.cols > 1 => {
            crate::compatibility::ensure_builtin_extension_enabled(char_matrix, builtin)
        }
        Value::Cell(cell) if !cell.data.iter().all(is_cellstr_element) => {
            crate::compatibility::ensure_builtin_extension_enabled(broad_cell, builtin)
        }
        _ => Ok(()),
    }
}

fn is_cellstr_element(value: &Value) -> bool {
    matches!(value, Value::CharArray(array) if array.rows <= 1 || array.cols <= 1)
}

#[runtime_builtin(
    name = "matches",
    category = "strings/search",
    summary = "Return true when text fully matches a pattern.",
    keywords = "matches,string pattern,text,regular expression",
    accel = "sink",
    type_resolver(bool_type),
    descriptor(crate::builtins::strings::transform::compat::MATCHES_DESCRIPTOR),
    integer_audit(crate::builtins::strings::transform::compat::MATCHES_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn matches_builtin(text: Value, pattern: Value) -> BuiltinResult<Value> {
    if crate::dispatcher::value_contains_gpu(&text)
        || crate::dispatcher::value_contains_gpu(&pattern)
    {
        return Err(transform_error(
            "matches",
            "matches: expected host text or pattern inputs",
        ));
    }
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

pub(super) async fn boundary_transform(
    text: Value,
    rest: Vec<Value>,
    fn_name: &'static str,
    resident_extension: &'static BuiltinExtensionDescriptor,
    char_matrix_extension: &'static BuiltinExtensionDescriptor,
    string_cell_extension: &'static BuiltinExtensionDescriptor,
    op: impl Fn(&str, &Boundary) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    if rest.len() != 1 {
        return Err(extract_error(
            fn_name,
            ExtractErrorKind::InvalidInput,
            format!("{fn_name}: expected exactly text and one boundary argument"),
        ));
    }
    if !valid_extract_text_input(&text, false) {
        return Err(extract_error(
            fn_name,
            ExtractErrorKind::InvalidInput,
            format!("{fn_name}: expected text input"),
        ));
    }
    if !valid_raw_boundary_form(&rest[0]) {
        return Err(extract_error(
            fn_name,
            ExtractErrorKind::InvalidBoundary,
            format!("{fn_name}: expected a text, pattern, or numeric boundary"),
        ));
    }
    if matches!(rest[0], Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(resident_extension, fn_name)?;
    }
    if is_multirow_char(&text) || is_multirow_char(&rest[0]) {
        crate::compatibility::ensure_builtin_extension_enabled(char_matrix_extension, fn_name)?;
    }
    if contains_string_or_nested_cell(&text) || contains_string_or_nested_cell(&rest[0]) {
        crate::compatibility::ensure_builtin_extension_enabled(string_cell_extension, fn_name)?;
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow(fn_name))?;
    if !valid_extract_text_input(&text, false) {
        return Err(extract_error(
            fn_name,
            ExtractErrorKind::InvalidInput,
            format!("{fn_name}: expected a supported text container"),
        ));
    }
    let boundary = gather_if_needed_async(&rest[0])
        .await
        .map_err(map_flow(fn_name))?;
    let boundaries = BoundaryList::from_value(&boundary, fn_name).map_err(|error| {
        extract_error(
            fn_name,
            ExtractErrorKind::InvalidBoundary,
            error.to_string(),
        )
    })?;
    let text_shape = boundary_text_shape(&text).ok_or_else(|| {
        extract_error(
            fn_name,
            ExtractErrorKind::InvalidInput,
            format!("{fn_name}: expected text input"),
        )
    })?;
    if !shape_is_scalar_or_same(&boundaries.shape, &text_shape) {
        return Err(extract_error(
            fn_name,
            ExtractErrorKind::SizeMismatch,
            format!("{fn_name}: boundary size must be scalar or match the text input"),
        ));
    }
    map_text_with_boundaries(text, &boundaries, fn_name, op).map_err(|error| {
        extract_error(
            fn_name,
            ExtractErrorKind::InvalidBoundary,
            error.to_string(),
        )
    })
}

pub(super) async fn insert_transform(
    text: Value,
    rest: Vec<Value>,
    fn_name: &'static str,
    after: bool,
) -> BuiltinResult<Value> {
    if rest.len() != 2 {
        return Err(insert_error(
            fn_name,
            0,
            format!("{fn_name}: expected exactly three inputs"),
        ));
    }
    let (resident_extension, broad_extension) = if after {
        (
            &INSERT_AFTER_RESIDENT_POSITION_EXTENSION,
            &INSERT_AFTER_BROAD_TEXT_EXTENSION,
        )
    } else {
        (
            &INSERT_BEFORE_RESIDENT_POSITION_EXTENSION,
            &INSERT_BEFORE_BROAD_TEXT_EXTENSION,
        )
    };
    if !valid_extract_text_input(&text, false) || !valid_insert_replacement(&rest[1]) {
        return Err(insert_error(
            fn_name,
            0,
            format!("{fn_name}: source and inserted values must be host text"),
        ));
    }
    if !valid_raw_boundary_form(&rest[0]) {
        return Err(insert_error(
            fn_name,
            1,
            format!("{fn_name}: expected a text, pattern, or numeric boundary"),
        ));
    }
    if matches!(rest[0], Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(resident_extension, fn_name)?;
    }
    if [&text, &rest[0], &rest[1]]
        .into_iter()
        .any(|value| is_multirow_char(value) || contains_string_or_nested_cell(value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(broad_extension, fn_name)?;
    }
    let boundary = gather_if_needed_async(&rest[0])
        .await
        .map_err(map_flow(fn_name))?;
    let new_text = gather_if_needed_async(&rest[1])
        .await
        .map_err(map_flow(fn_name))?;
    let boundaries = BoundaryList::from_value(&boundary, fn_name)
        .map_err(|error| insert_error(fn_name, 1, error.to_string()))?;
    let replacements = TextList::from_value(&new_text, fn_name)
        .map_err(|error| insert_error(fn_name, 0, error.to_string()))?;
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow(fn_name))?;
    let text_shape = boundary_text_shape(&text)
        .ok_or_else(|| insert_error(fn_name, 0, format!("{fn_name}: expected text input")))?;
    if !shape_is_scalar_or_same(&boundaries.shape, &text_shape)
        || !shape_is_scalar_or_same(&replacements.shape, &text_shape)
    {
        return Err(insert_error(
            fn_name,
            2,
            format!("{fn_name}: boundary and newText must be scalar or match str"),
        ));
    }
    map_text_with_insertions(text, &boundaries, &replacements, fn_name, after)
}

fn insert_error(name: &str, error_index: usize, message: impl Into<String>) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(name);
    if let Some(identifier) = INSERT_ERRORS[error_index].identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone)]
pub(super) enum Boundary {
    Position(usize),
    Text(String),
    Pattern(String),
}

#[derive(Clone)]
struct BoundaryList {
    data: Vec<Boundary>,
    shape: Vec<usize>,
}

#[derive(Clone)]
struct TextList {
    data: Vec<String>,
    shape: Vec<usize>,
}

impl TextList {
    fn from_value(value: &Value, fn_name: &str) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Ok(Self {
                data: vec![text.clone()],
                shape: vec![1, 1],
            }),
            Value::StringArray(array) => Ok(Self {
                data: array.data.clone(),
                shape: array.shape.clone(),
            }),
            Value::CharArray(array) => {
                let data = if array.rows == 0 {
                    vec![String::new()]
                } else {
                    (0..array.rows)
                        .map(|row| char_row_to_string_slice(&array.data, array.cols, row))
                        .collect()
                };
                Ok(Self {
                    data,
                    shape: if array.rows <= 1 {
                        vec![1, 1]
                    } else {
                        vec![array.rows, 1]
                    },
                })
            }
            Value::Cell(cell) => Ok(Self {
                data: cell
                    .data
                    .iter()
                    .map(|value| scalar_text(value, fn_name))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                shape: cell.shape.clone(),
            }),
            other => Err(transform_error(
                fn_name,
                format!("{fn_name}: newText must be text, got {other:?}"),
            )),
        }
    }

    fn at(&self, idx: usize) -> &str {
        if self.data.len() == 1 {
            &self.data[0]
        } else {
            &self.data[idx]
        }
    }
}

impl BoundaryList {
    fn from_value(value: &Value, fn_name: &str) -> BuiltinResult<Self> {
        match value {
            Value::Num(value) => Ok(Self {
                data: vec![Boundary::Position(parse_boundary_float(*value, fn_name)?)],
                shape: vec![1, 1],
            }),
            Value::Int(value) => Ok(Self {
                data: vec![Boundary::Position(parse_boundary_integer(
                    value.clone(),
                    fn_name,
                )?)],
                shape: vec![1, 1],
            }),
            Value::Tensor(tensor) => {
                let mut data = Vec::with_capacity(tensor.len());
                for idx in 0..tensor.len() {
                    let value = tensor.numeric_value_at(idx).ok_or_else(|| {
                        transform_error(
                            fn_name,
                            format!("{fn_name}: numeric boundaries must be positive integers"),
                        )
                    })?;
                    let position = match value {
                        NumericScalar::F64(value) => parse_boundary_float(value, fn_name)?,
                        NumericScalar::F32(value) => {
                            parse_boundary_float(f64::from(value), fn_name)?
                        }
                        integer => parse_boundary_integer(
                            integer
                                .into_int_value()
                                .expect("non-floating numeric scalar is integer"),
                            fn_name,
                        )?,
                    };
                    data.push(Boundary::Position(position));
                }
                Ok(Self {
                    data,
                    shape: tensor.shape().to_vec(),
                })
            }
            Value::String(text) => Ok(Self {
                data: vec![Boundary::Text(text.clone())],
                shape: vec![1, 1],
            }),
            Value::StringArray(array) => Ok(Self {
                data: array.data.iter().cloned().map(Boundary::Text).collect(),
                shape: array.shape.clone(),
            }),
            Value::CharArray(array) => {
                let data = if array.rows == 0 {
                    vec![Boundary::Text(String::new())]
                } else {
                    (0..array.rows)
                        .map(|row| {
                            Boundary::Text(char_row_to_string_slice(&array.data, array.cols, row))
                        })
                        .collect()
                };
                Ok(Self {
                    data,
                    shape: if array.rows <= 1 {
                        vec![1, 1]
                    } else {
                        vec![array.rows, 1]
                    },
                })
            }
            Value::Cell(cell) => {
                let mut data = Vec::with_capacity(cell.data.len());
                for value in &cell.data {
                    data.push(Boundary::Text(scalar_text(value, fn_name)?));
                }
                Ok(Self {
                    data,
                    shape: cell.shape.clone(),
                })
            }
            Value::Object(_) => Ok(Self {
                data: vec![Boundary::Pattern(pattern_regex(value, fn_name)?)],
                shape: vec![1, 1],
            }),
            other => Err(transform_error(
                fn_name,
                format!("{fn_name}: expected a text, pattern, or numeric boundary, got {other:?}"),
            )),
        }
    }

    fn at(&self, idx: usize) -> &Boundary {
        if self.data.len() == 1 {
            &self.data[0]
        } else {
            &self.data[idx]
        }
    }
}

fn parse_boundary_integer(value: IntValue, fn_name: &str) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|position| *position > 0)
        .ok_or_else(|| {
            transform_error(
                fn_name,
                format!("{fn_name}: numeric boundaries must be positive integer scalars"),
            )
        })
}

fn parse_boundary_float(value: f64, fn_name: &str) -> BuiltinResult<usize> {
    if value > usize::MAX.saturating_sub(1) as f64 {
        return Err(transform_error(
            fn_name,
            format!("{fn_name}: numeric boundaries must be positive integer scalars"),
        ));
    }
    let parsed = value as usize;
    if parsed == 0 || parsed as f64 != value || parsed == usize::MAX {
        return Err(transform_error(
            fn_name,
            format!("{fn_name}: numeric boundaries must be positive integer scalars"),
        ));
    }
    Ok(parsed)
}

pub(super) fn locate_boundary(text: &str, boundary: &Boundary) -> BuiltinResult<(usize, usize)> {
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

/// Validate the boundary's container before recursive gather. Only a top-level
/// real numeric resident handle is a position control; a resident value inside
/// a cell or pattern object cannot become a valid boundary after download.
fn valid_raw_boundary_form(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) => true,
        Value::GpuTensor(handle) => {
            !runmat_accelerate_api::handle_is_logical(handle)
                && handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => true,
        Value::Cell(cell) => cell.data.iter().all(|item| match item {
            Value::String(_) => true,
            Value::StringArray(array) => array.data.len() == 1,
            Value::CharArray(array) => array.rows <= 1,
            _ => false,
        }),
        Value::Object(object) => {
            object.is_class(runmat_types::standard::PATTERN)
                && matches!(object.properties.get("Regex"), Some(Value::String(_)))
                && !crate::dispatcher::value_contains_gpu(value)
        }
        _ => false,
    }
}

fn is_multirow_char(value: &Value) -> bool {
    matches!(value, Value::CharArray(array) if array.rows > 1)
}

fn contains_string_or_nested_cell(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell.data.iter().any(|value| {
            matches!(
                value,
                Value::String(_) | Value::StringArray(_) | Value::Cell(_)
            ) || contains_string_or_nested_cell(value)
        }),
        _ => false,
    }
}

fn valid_extract_text_input(value: &Value, nested: bool) -> bool {
    match value {
        Value::String(_) => true,
        Value::StringArray(array) => !nested || array.data.len() == 1,
        Value::CharArray(array) => !nested || array.rows <= 1,
        Value::Cell(cell) => cell
            .data
            .iter()
            .all(|value| valid_extract_text_input(value, true)),
        _ => false,
    }
}

fn valid_insert_replacement(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => true,
        Value::Cell(cell) => cell.data.iter().all(|item| match item {
            Value::String(_) => true,
            Value::StringArray(array) => array.data.len() == 1,
            Value::CharArray(array) => array.rows <= 1,
            _ => false,
        }),
        _ => false,
    }
}

fn boundary_text_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::String(_) => Some(vec![1, 1]),
        Value::StringArray(array) => Some(array.shape.clone()),
        Value::CharArray(array) if array.rows <= 1 => Some(vec![1, 1]),
        Value::CharArray(array) => Some(vec![array.rows, 1]),
        Value::Cell(cell) => Some(cell.shape.clone()),
        _ => None,
    }
}

fn shape_is_scalar_or_same(shape: &[usize], text_shape: &[usize]) -> bool {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        == Some(1)
        || shape == text_shape
}

fn map_text_with_boundaries(
    value: Value,
    boundaries: &BoundaryList,
    fn_name: &str,
    map: impl Fn(&str, &Boundary) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            if is_missing_string(&text) {
                Ok(Value::String(text))
            } else {
                Ok(Value::String(map(&text, boundaries.at(0))?))
            }
        }
        Value::StringArray(array) => StringArray::new(
            array
                .data
                .into_iter()
                .enumerate()
                .map(|(idx, text)| {
                    if is_missing_string(&text) {
                        Ok(text)
                    } else {
                        map(&text, boundaries.at(idx))
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            array.shape,
        )
        .map(Value::StringArray)
        .map_err(|e| transform_error(fn_name, e)),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| {
                    map(
                        &char_row_to_string_slice(&array.data, array.cols, row),
                        boundaries.at(row),
                    )
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            char_rows(rows, fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .enumerate()
                .map(|(idx, value)| map_text_cell_element(value, boundaries.at(idx), fn_name, map))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape).map_err(|e| transform_error(fn_name, e))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

fn insert_one(
    text: &str,
    boundary: &Boundary,
    replacement: &str,
    after: bool,
    fn_name: &str,
) -> BuiltinResult<String> {
    if let Boundary::Position(_) = boundary {
        let (start, end) = locate_boundary(text, boundary)?;
        let idx = if after { end } else { start };
        return Ok(format!("{}{}{}", &text[..idx], replacement, &text[idx..]));
    }

    let spans: Vec<(usize, usize)> = match boundary {
        Boundary::Text(needle) => text
            .match_indices(needle)
            .map(|(start, matched)| (start, start + matched.len()))
            .collect(),
        Boundary::Pattern(pattern) => Regex::new(pattern)
            .map_err(|error| transform_error(fn_name, error.to_string()))?
            .find_iter(text)
            .map(|matched| (matched.start(), matched.end()))
            .collect(),
        Boundary::Position(_) => unreachable!(),
    };
    if spans.is_empty() {
        return Err(transform_error(fn_name, "boundary not found"));
    }
    let mut result = String::with_capacity(text.len() + replacement.len() * spans.len());
    let mut cursor = 0;
    for (start, end) in spans {
        result.push_str(&text[cursor..start]);
        if after {
            result.push_str(&text[start..end]);
            result.push_str(replacement);
        } else {
            result.push_str(replacement);
            result.push_str(&text[start..end]);
        }
        cursor = end;
    }
    result.push_str(&text[cursor..]);
    Ok(result)
}

fn map_text_with_insertions(
    value: Value,
    boundaries: &BoundaryList,
    replacements: &TextList,
    fn_name: &str,
    after: bool,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            if is_missing_string(&text) {
                Ok(Value::String(text))
            } else {
                insert_one(&text, boundaries.at(0), replacements.at(0), after, fn_name)
                    .map(Value::String)
            }
        }
        Value::StringArray(array) => StringArray::new(
            array
                .data
                .into_iter()
                .enumerate()
                .map(|(idx, text)| {
                    if is_missing_string(&text) {
                        Ok(text)
                    } else {
                        insert_one(
                            &text,
                            boundaries.at(idx),
                            replacements.at(idx),
                            after,
                            fn_name,
                        )
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            array.shape,
        )
        .map(Value::StringArray)
        .map_err(|error| transform_error(fn_name, error)),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| {
                    insert_one(
                        &char_row_to_string_slice(&array.data, array.cols, row),
                        boundaries.at(row),
                        replacements.at(row),
                        after,
                        fn_name,
                    )
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            char_rows(rows, fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .enumerate()
                .map(|(idx, value)| {
                    map_text_cell_insertion(
                        value,
                        boundaries.at(idx),
                        replacements.at(idx),
                        fn_name,
                        after,
                    )
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape)
                .map_err(|error| transform_error(fn_name, error))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

fn map_text_with_replacements(
    value: Value,
    starts: &BoundaryList,
    stops: &BoundaryList,
    replacements: &TextList,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            if is_missing_string(&text) {
                Ok(Value::String(text))
            } else {
                replace_between_one(&text, starts.at(0), stops.at(0), replacements.at(0))
                    .map(Value::String)
            }
        }
        Value::StringArray(array) => StringArray::new(
            array
                .data
                .into_iter()
                .enumerate()
                .map(|(index, text)| {
                    if is_missing_string(&text) {
                        Ok(text)
                    } else {
                        replace_between_one(
                            &text,
                            starts.at(index),
                            stops.at(index),
                            replacements.at(index),
                        )
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            array.shape,
        )
        .map(Value::StringArray)
        .map_err(|error| transform_error("replaceBetween", error)),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| {
                    replace_between_one(
                        &char_row_to_string_slice(&array.data, array.cols, row),
                        starts.at(row),
                        stops.at(row),
                        replacements.at(row),
                    )
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            char_rows(rows, "replaceBetween")
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    map_text_cell_replacement(
                        value,
                        starts.at(index),
                        stops.at(index),
                        replacements.at(index),
                    )
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape)
                .map_err(|error| transform_error("replaceBetween", error))
        }
        other => Err(transform_error(
            "replaceBetween",
            format!("replaceBetween: expected text input, got {other:?}"),
        )),
    }
}

fn map_text_cell_replacement(
    value: Value,
    start: &Boundary,
    stop: &Boundary,
    replacement: &str,
) -> BuiltinResult<Value> {
    match value {
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&array.data, array.cols, 0)
            };
            replace_between_one(&text, start, stop, replacement)
                .map(|text| Value::CharArray(CharArray::new_row(&text)))
        }
        Value::String(text) => {
            replace_between_one(&text, start, stop, replacement).map(Value::String)
        }
        Value::StringArray(array) if array.data.len() == 1 => {
            replace_between_one(&array.data[0], start, stop, replacement).map(Value::String)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| map_text_cell_replacement(value, start, stop, replacement))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape)
                .map_err(|error| transform_error("replaceBetween", error))
        }
        other => Err(transform_error(
            "replaceBetween",
            format!("replaceBetween: expected character vectors in cell input, got {other:?}"),
        )),
    }
}

fn replace_between_one(
    text: &str,
    start: &Boundary,
    stop: &Boundary,
    replacement: &str,
) -> BuiltinResult<String> {
    let (replace_start, replace_end) = replacement_span_between(text, start, stop)?;
    Ok(format!(
        "{}{}{}",
        &text[..replace_start],
        replacement,
        &text[replace_end..]
    ))
}

fn map_text_cell_insertion(
    value: Value,
    boundary: &Boundary,
    replacement: &str,
    fn_name: &str,
    after: bool,
) -> BuiltinResult<Value> {
    match value {
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&array.data, array.cols, 0)
            };
            insert_one(&text, boundary, replacement, after, fn_name)
                .map(|text| Value::CharArray(CharArray::new_row(&text)))
        }
        Value::String(text) => {
            insert_one(&text, boundary, replacement, after, fn_name).map(Value::String)
        }
        Value::StringArray(array) if array.data.len() == 1 => {
            insert_one(&array.data[0], boundary, replacement, after, fn_name).map(Value::String)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| map_text_cell_insertion(value, boundary, replacement, fn_name, after))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape)
                .map_err(|error| transform_error(fn_name, error))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected character vectors in cell input, got {other:?}"),
        )),
    }
}

fn map_text_cell_element(
    value: Value,
    boundary: &Boundary,
    fn_name: &str,
    map: impl Fn(&str, &Boundary) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => map(&text, boundary).map(Value::String),
        Value::StringArray(array) if array.data.len() == 1 => {
            map(&array.data[0], boundary).map(Value::String)
        }
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&array.data, array.cols, 0)
            };
            map(&text, boundary).map(|text| Value::CharArray(CharArray::new_row(&text)))
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| map_text_cell_element(value, boundary, fn_name, map))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape).map_err(|e| transform_error(fn_name, e))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected character vectors in cell input, got {other:?}"),
        )),
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
mod tests;
