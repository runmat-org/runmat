//! MATLAB text transformation compatibility helpers.
use runmat_builtins::REPLACE_BETWEEN_EXPLICIT_GPU_EXTENSION;

use regex::Regex;

#[cfg(test)]
use super::{
    extractafter::extract_after_builtin, extractbefore::extract_before_builtin,
    insertafter::insert_after_builtin, insertbefore::insert_before_builtin,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, ERASE_PUNCTUATION_BROAD_CELL_EXTENSION,
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
use runmat_value::{
    CellArray, CharArray, ExactArrayStorage, IntValue, NumericScalar, StringArray, StringElement,
    Value,
};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::common::{
    char_row_to_string_slice, contains_numeric_or_resident_text_input, scalar_text_input,
    text_input_admission, TextInput,
};
use crate::builtins::strings::core::compat::{
    broadcast_flat_index, broadcast_shape, logical_value, pattern_regex, scalar_text, text_items,
};
use crate::builtins::strings::text_analytics::documents::{
    erase_punctuation_tokenized_document, erase_urls_tokenized_document,
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};

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
async fn append_builtin(rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Ok(Value::string(String::new()));
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
        let mut missing = false;
        for list in &lists {
            let source = broadcast_flat_index(idx, &shape, &list.shape);
            if let Some(part) = &list.items[source] {
                text.push_str(part);
            } else {
                missing = true;
                break;
            }
        }
        out.push(if missing {
            StringElement::Missing
        } else {
            StringElement::text(text)
        });
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
            let mut items = Vec::with_capacity(cell.data().len());
            for value in cell.data() {
                match value {
                    Value::CharArray(array) if array.rows <= 1 || array.cols <= 1 => {
                        let text = runmat_value::decode_utf16_code_units(array.data())
                            .map_err(|e| transform_error("append", e.to_string()))?;
                        items.push(Some(runmat_value::Utf8Text::new(
                            text,
                        )));
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
                    shape: cell.shape().to_vec(),
                },
                AppendOutputKind::Cell,
            ))
        }
        Value::CharArray(array) if array.rows <= 1 || array.cols <= 1 => {
            let text = runmat_value::decode_utf16_code_units(array.data())
                .map_err(|e| transform_error("append", e.to_string()))?;
            Ok((
                text_items(Value::string(text), "append")?,
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
    values: Vec<StringElement>,
    shape: Vec<usize>,
    kind: AppendOutputKind,
) -> BuiltinResult<Value> {
    match kind {
        AppendOutputKind::String => string_array_or_scalar(values, shape, "append"),
        AppendOutputKind::Cell => {
            let values = values
                .into_iter()
                .map(|text| match text.as_str() {
                    Some(text) => char_rows(vec![text.to_owned()], "append"),
                    None => Ok(Value::String(StringElement::Missing)),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, shape).map_err(|error| transform_error("append", error))
        }
        AppendOutputKind::Char => {
            let text = values.into_iter().next().unwrap_or_default();
            match text.as_str() {
                Some(text) => char_rows(vec![text.to_owned()], "append"),
                None => Ok(Value::String(StringElement::Missing)),
            }
        }
    }
}

#[runtime_builtin(
    name = "reverse",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn reverse_builtin(text: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::REVERSE_DESCRIPTOR,
        &runmat_builtins::REVERSE_ERROR_TOO_MANY_OUTPUTS,
        "reverse",
    )?;
    if !is_host_text_tree(&text)
        || crate::builtins::strings::common::has_unsupported_character_pages(&text)
    {
        return Err(descriptor_transform_error(
            "reverse",
            &runmat_builtins::REVERSE_ERROR_INVALID_INPUT,
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
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::compat",
    context = true
)]
async fn strjust_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    text: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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
        scalar_text_input(value, runtime.as_ref(), |runtime, cause| {
            text_input_admission("strjust", runtime, cause)
        })?
        .ok_or_else(|| transform_error("strjust", "strjust: expected text alignment"))?
    } else {
        TextInput::Borrowed("right")
    };
    map_text_preserve(text, "strjust", |s| justify(s, side.as_str()))
}

#[runtime_builtin(
    name = "splitlines",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn splitlines_builtin(text: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SPLITLINES_DESCRIPTOR,
        &runmat_builtins::SPLITLINES_ERROR_TOO_MANY_OUTPUTS,
        "splitlines",
    )?;
    if contains_numeric_or_resident_text_input(&text) {
        return Err(descriptor_transform_error(
            "splitlines",
            &runmat_builtins::SPLITLINES_ERROR_INVALID_INPUT,
        ));
    }
    if crate::builtins::strings::common::has_unsupported_character_pages(&text) {
        return Err(descriptor_transform_error(
            "splitlines",
            &runmat_builtins::SPLITLINES_ERROR_INVALID_INPUT,
        ));
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("splitlines"))?;
    match text {
        Value::String(text) => split_element_lines(&text),
        Value::StringArray(array) if array.data().len() == 1 => {
            split_element_lines(&array.data()[0])
        }
        Value::CharArray(array) if array.rows <= 1 => {
            let text = char_row_to_string_slice(array.data(), array.cols, 0)
                .map_err(|e| transform_error("splitlines", e.to_string()))?;
            strings_from_lines(&text)
        }
        other => {
            let list = text_items(other, "splitlines")?;
            let values = list
                .items
                .into_iter()
                .map(|text| match text {
                    Some(text) => strings_from_lines(&text),
                    None => Ok(Value::String(StringElement::Missing)),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, list.shape).map_err(|e| transform_error("splitlines", e))
        }
    }
}

#[runtime_builtin(
    name = "replaceBetween",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn replace_between_builtin(
    text: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::REPLACE_BETWEEN_DESCRIPTOR,
        &runmat_builtins::REPLACE_BETWEEN_ERROR_TOO_MANY_OUTPUTS,
        "replaceBetween",
    )?;
    if rest.len() != 3 {
        return Err(descriptor_transform_error(
            "replaceBetween",
            &runmat_builtins::REPLACE_BETWEEN_ERROR_ARGUMENT_COUNT,
        ));
    }
    if crate::builtins::strings::common::has_unsupported_character_pages(&text)
        || rest
            .iter()
            .any(crate::builtins::strings::common::has_unsupported_character_pages)
    {
        return Err(descriptor_transform_error(
            "replaceBetween",
            &runmat_builtins::REPLACE_BETWEEN_ERROR_INVALID_INPUT,
        ));
    }
    if contains_numeric_or_resident_text_input(&text)
        || contains_numeric_or_resident_text_input(&rest[2])
    {
        return Err(descriptor_transform_error(
            "replaceBetween",
            &runmat_builtins::REPLACE_BETWEEN_ERROR_INVALID_INPUT,
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
async fn erase_punctuation_builtin(
    text: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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
        Value::Cell(cell) => cell.data().iter().all(is_host_text_tree),
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
        Value::Cell(cell) if !cell.data().iter().all(is_cellstr_element) => {
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
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn matches_builtin(text: Value, pattern: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::MATCHES_DESCRIPTOR,
        &runmat_builtins::MATCHES_ERROR_TOO_MANY_OUTPUTS,
        "matches",
    )?;
    if crate::dispatcher::value_contains_gpu(&text)
        || crate::dispatcher::value_contains_gpu(&pattern)
        || [&text, &pattern]
            .into_iter()
            .any(crate::builtins::strings::common::has_unsupported_character_pages)
    {
        return Err(descriptor_transform_error(
            "matches",
            &runmat_builtins::MATCHES_ERROR_INVALID_INPUT,
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
    let (out, shape) = match text {
        // TextList's cell traversal serves physical cell outputs elsewhere.
        // Logical storage instead requires MATLAB-visible column-major order.
        Value::Cell(cell) => {
            let out = cell
                .iter_column_major()
                .map(|value| {
                    scalar_text(value, "matches").map(|text| u8::from(regex.is_match(&text)))
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            (out, cell.shape().to_vec())
        }
        other => {
            let list = text_items(other, "matches")?;
            let out = list
                .items
                .iter()
                .map(|text| u8::from(text.as_ref().is_some_and(|text| regex.is_match(text))))
                .collect::<Vec<_>>();
            (out, list.shape)
        }
    };
    logical_value(out, shape, "matches")
}

pub(super) async fn boundary_transform(
    text: Value,
    rest: crate::call::BuiltinArguments,
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
    rest: crate::call::BuiltinArguments,
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
                data: vec![required_transform_text(text, fn_name)?.to_owned()],
                shape: vec![1, 1],
            }),
            Value::StringArray(array) => Ok(Self {
                data: array
                    .data()
                    .iter()
                    .map(|text| required_transform_text(text, fn_name).map(str::to_owned))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                shape: array.shape().to_vec(),
            }),
            Value::CharArray(array) => {
                let data = if array.rows == 0 {
                    vec![String::new()]
                } else {
                    (0..array.rows)
                        .map(|row| {
                            char_row_to_string_slice(array.data(), array.cols, row)
                                .map_err(|e| transform_error(fn_name, e.to_string()))
                        })
                        .collect::<BuiltinResult<Vec<_>>>()?
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
                    .data()
                    .iter()
                    .map(|value| scalar_text(value, fn_name))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                shape: cell.shape().to_vec(),
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
                data: vec![Boundary::Text(
                    required_transform_text(text, fn_name)?.to_owned(),
                )],
                shape: vec![1, 1],
            }),
            Value::StringArray(array) => Ok(Self {
                data: array
                    .data()
                    .iter()
                    .map(|text| {
                        required_transform_text(text, fn_name)
                            .map(|text| Boundary::Text(text.to_owned()))
                    })
                    .collect::<BuiltinResult<Vec<_>>>()?,
                shape: array.shape().to_vec(),
            }),
            Value::CharArray(array) => {
                let data = if array.rows == 0 {
                    vec![Boundary::Text(String::new())]
                } else {
                    (0..array.rows)
                        .map(|row| {
                            char_row_to_string_slice(array.data(), array.cols, row)
                                .map(Boundary::Text)
                                .map_err(|e| transform_error(fn_name, e.to_string()))
                        })
                        .collect::<BuiltinResult<Vec<_>>>()?
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
                let mut data = Vec::with_capacity(cell.data().len());
                for value in cell.data() {
                    data.push(Boundary::Text(scalar_text(value, fn_name)?));
                }
                Ok(Self {
                    data,
                    shape: cell.shape().to_vec(),
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
        Value::Cell(cell) => cell.data().iter().all(|item| match item {
            Value::String(_) => true,
            Value::StringArray(array) => array.data().len() == 1,
            Value::CharArray(array) => array.rows <= 1,
            _ => false,
        }),
        Value::Object(object) => {
            object.is_class(runmat_types::standard::PATTERN)
                && matches!(object.properties().get("Regex"), Some(Value::String(_)))
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
        Value::Cell(cell) => cell.data().iter().any(|value| {
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
        Value::StringArray(array) => !nested || array.data().len() == 1,
        Value::CharArray(array) => !nested || array.rows <= 1,
        Value::Cell(cell) => cell
            .data()
            .iter()
            .all(|value| valid_extract_text_input(value, true)),
        _ => false,
    }
}

fn valid_insert_replacement(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => true,
        Value::Cell(cell) => cell.data().iter().all(|item| match item {
            Value::String(_) => true,
            Value::StringArray(array) => array.data().len() == 1,
            Value::CharArray(array) => array.rows <= 1,
            _ => false,
        }),
        _ => false,
    }
}

fn boundary_text_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::String(_) => Some(vec![1, 1]),
        Value::StringArray(array) => Some(array.shape().to_vec()),
        Value::CharArray(array) if array.rows <= 1 => Some(vec![1, 1]),
        Value::CharArray(array) => Some(vec![array.rows, 1]),
        Value::Cell(cell) => Some(cell.shape().to_vec()),
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

fn map_string_array(
    array: StringArray,
    fn_name: &str,
    mut map: impl FnMut(usize, &str) -> BuiltinResult<String>,
) -> BuiltinResult<Value> {
    let (data, shape, _, _) = array.into_parts();
    let input: Vec<StringElement> = match data.try_into_owned() {
        Ok(values) => values.into_iter().collect(),
        Err(shared) => shared.iter().cloned().collect(),
    };
    let mut output = Vec::with_capacity(input.len());
    for (index, text) in input.into_iter().enumerate() {
        output.push(match text.as_str() {
            Some(source) => StringElement::text(map(index, source)?),
            None => StringElement::Missing,
        });
    }
    StringArray::from_shared_storage(ExactArrayStorage::shared_unbounded(output), shape)
        .map(Value::StringArray)
        .map_err(|error| transform_error(fn_name, error))
}

fn map_text_with_boundaries(
    value: Value,
    boundaries: &BoundaryList,
    fn_name: &str,
    map: impl Fn(&str, &Boundary) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            if text.is_missing() {
                Ok(Value::String(text))
            } else {
                Ok(Value::string(map(
                    text.as_str().expect("nonmissing subject"),
                    boundaries.at(0),
                )?))
            }
        }
        Value::StringArray(array) => map_string_array(array, fn_name, |index, text| {
            map(text, boundaries.at(index))
        }),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| {
                    let text = char_row_to_string_slice(array.data(), array.cols, row)
                        .map_err(|e| transform_error(fn_name, e.to_string()))?;
                    map(&text, boundaries.at(row))
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            char_rows(rows, fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data()
                .iter()
                .enumerate()
                .map(|(idx, value)| map_text_cell_element(value, boundaries.at(idx), fn_name, map))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape().to_vec())
                .map_err(|e| transform_error(fn_name, e))
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
            if text.is_missing() {
                Ok(Value::String(text))
            } else {
                insert_one(
                    text.as_str().expect("nonmissing subject"),
                    boundaries.at(0),
                    replacements.at(0),
                    after,
                    fn_name,
                )
                .map(Value::string)
            }
        }
        Value::StringArray(array) => map_string_array(array, fn_name, |index, text| {
            insert_one(
                text,
                boundaries.at(index),
                replacements.at(index),
                after,
                fn_name,
            )
        }),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| {
                    let text = char_row_to_string_slice(array.data(), array.cols, row)
                        .map_err(|e| transform_error(fn_name, e.to_string()))?;
                    insert_one(
                        &text,
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
                .data()
                .iter()
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
            make_cell_with_shape(values, cell.shape().to_vec())
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
            if text.is_missing() {
                Ok(Value::String(text))
            } else {
                replace_between_one(
                    text.as_str().expect("nonmissing subject"),
                    starts.at(0),
                    stops.at(0),
                    replacements.at(0),
                )
                .map(Value::string)
            }
        }
        Value::StringArray(array) => map_string_array(array, "replaceBetween", |index, text| {
            replace_between_one(
                text,
                starts.at(index),
                stops.at(index),
                replacements.at(index),
            )
        }),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| {
                    let text = char_row_to_string_slice(array.data(), array.cols, row)
                        .map_err(|e| transform_error("replaceBetween", e.to_string()))?;
                    replace_between_one(&text, starts.at(row), stops.at(row), replacements.at(row))
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            char_rows(rows, "replaceBetween")
        }
        Value::Cell(cell) => {
            let values = cell
                .data()
                .iter()
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
            make_cell_with_shape(values, cell.shape().to_vec())
                .map_err(|error| transform_error("replaceBetween", error))
        }
        other => Err(transform_error(
            "replaceBetween",
            format!("replaceBetween: expected text input, got {other:?}"),
        )),
    }
}

fn map_text_cell_replacement(
    value: &Value,
    start: &Boundary,
    stop: &Boundary,
    replacement: &str,
) -> BuiltinResult<Value> {
    match value {
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(array.data(), array.cols, 0)
                    .map_err(|e| transform_error("replaceBetween", e.to_string()))?
            };
            replace_between_one(&text, start, stop, replacement)
                .map(|text| Value::CharArray(CharArray::new_row(&text)))
        }
        Value::String(text) => map_element_value(text, |text| {
            replace_between_one(text, start, stop, replacement)
        }),
        Value::StringArray(array) if array.data().len() == 1 => {
            map_element_value(&array.data()[0], |text| {
                replace_between_one(text, start, stop, replacement)
            })
        }
        Value::Cell(cell) => {
            let values = cell
                .data()
                .iter()
                .map(|value| map_text_cell_replacement(value, start, stop, replacement))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape().to_vec())
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
    value: &Value,
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
                char_row_to_string_slice(array.data(), array.cols, 0)
                    .map_err(|e| transform_error(fn_name, e.to_string()))?
            };
            insert_one(&text, boundary, replacement, after, fn_name)
                .map(|text| Value::CharArray(CharArray::new_row(&text)))
        }
        Value::String(text) => map_element_value(text, |text| {
            insert_one(text, boundary, replacement, after, fn_name)
        }),
        Value::StringArray(array) if array.data().len() == 1 => {
            map_element_value(&array.data()[0], |text| {
                insert_one(text, boundary, replacement, after, fn_name)
            })
        }
        Value::Cell(cell) => {
            let values = cell
                .data()
                .iter()
                .map(|value| map_text_cell_insertion(value, boundary, replacement, fn_name, after))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape().to_vec())
                .map_err(|error| transform_error(fn_name, error))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected character vectors in cell input, got {other:?}"),
        )),
    }
}

fn map_text_cell_element(
    value: &Value,
    boundary: &Boundary,
    fn_name: &str,
    map: impl Fn(&str, &Boundary) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => map_element_value(text, |text| map(text, boundary)),
        Value::StringArray(array) if array.data().len() == 1 => {
            map_element_value(&array.data()[0], |text| map(text, boundary))
        }
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(array.data(), array.cols, 0)
                    .map_err(|e| transform_error(fn_name, e.to_string()))?
            };
            map(&text, boundary).map(|text| Value::CharArray(CharArray::new_row(&text)))
        }
        Value::Cell(cell) => {
            let values = cell
                .data()
                .iter()
                .map(|value| map_text_cell_element(value, boundary, fn_name, map))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape().to_vec())
                .map_err(|e| transform_error(fn_name, e))
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
            if text.is_missing() {
                Ok(Value::String(text))
            } else {
                Ok(Value::string(map(text
                    .as_str()
                    .expect("nonmissing subject"))))
            }
        }
        Value::StringArray(array) => map_string_array(array, fn_name, |_, text| Ok(map(text))),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| {
                    char_row_to_string_slice(array.data(), array.cols, row)
                        .map(|text| map(&text))
                        .map_err(|e| transform_error(fn_name, e.to_string()))
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            char_rows(rows, fn_name)
        }
        Value::Cell(cell) => {
            let (mut owner, shape, _rows, _cols) = cell.into_parts();
            if !owner.is_unique() {
                owner
                    .make_unique_unbounded()
                    .map_err(|error| transform_error(fn_name, error.to_string()))?;
            }
            let mut values = owner.try_into_owned().map_err(|_| {
                transform_error(fn_name, "shared admitted Cell requires a prepared output")
            })?;
            for value in values.as_mut_slice() {
                let source = std::mem::replace(value, Value::Num(0.0));
                *value = map_text_preserve(source, fn_name, map)?;
            }
            let owner = values.try_into_storage().map_err(|_| {
                transform_error(fn_name, "Cell output owner could not be republished")
            })?;
            CellArray::from_shared_storage(owner, shape)
                .map(Value::Cell)
                .map_err(|error| transform_error(fn_name, error))
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
        .map(|row| row.encode_utf16().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(row_count * cols);
    for row in rows {
        let mut units = row
            .encode_utf16()
            .map(runmat_value::Utf16CodeUnit::from_raw)
            .collect::<Vec<_>>();
        units.resize(cols, runmat_value::Utf16CodeUnit::from_raw(b' ' as u16));
        data.extend(units);
    }
    CharArray::from_code_units(data, row_count, cols)
        .map(Value::CharArray)
        .map_err(|e| transform_error(fn_name, e))
}

fn string_array_or_scalar(
    values: Vec<StringElement>,
    shape: Vec<usize>,
    fn_name: &str,
) -> BuiltinResult<Value> {
    if values.len() == 1 {
        Ok(Value::string(values.into_iter().next().unwrap()))
    } else {
        StringArray::from_elements_unbounded(values, shape)
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
    if side.eq_ignore_ascii_case("left") {
        format!("{trimmed}{}", " ".repeat(pad))
    } else if side.eq_ignore_ascii_case("center") || side.eq_ignore_ascii_case("centre") {
        let left = pad / 2;
        let right = pad - left;
        format!("{}{}{}", " ".repeat(left), trimmed, " ".repeat(right))
    } else {
        format!("{}{}", " ".repeat(pad), trimmed)
    }
}

#[cfg(test)]
mod tests;

fn required_transform_text<'a>(text: &'a StringElement, fn_name: &str) -> BuiltinResult<&'a str> {
    text.as_str().ok_or_else(|| {
        transform_error(
            fn_name,
            format!("{fn_name}: text argument must not be missing"),
        )
    })
}

fn map_element_value(
    text: &StringElement,
    map: impl FnOnce(&str) -> BuiltinResult<String>,
) -> BuiltinResult<Value> {
    match text.as_str() {
        Some(text) => map(text).map(Value::string),
        None => Ok(Value::String(StringElement::Missing)),
    }
}

fn split_element_lines(text: &StringElement) -> BuiltinResult<Value> {
    match text.as_str() {
        Some(text) => strings_from_lines(text),
        None => Ok(Value::String(StringElement::Missing)),
    }
}
