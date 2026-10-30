//! MATLAB text compatibility helpers that do not warrant larger domain modules yet.

use encoding_rs::Encoding;
use runmat_builtins::{
    MAT2STR_DESCRIPTOR, MAT2STR_ERROR_TOO_MANY_OUTPUTS, MAT2STR_INTEGER_PRECISION_EXTENSION,
    MAT2STR_TEXT_INPUT_EXTENSION, NATIVE2UNICODE_DESCRIPTOR, NATIVE2UNICODE_ERROR_TOO_MANY_OUTPUTS,
    SSCANF_DESCRIPTOR, SSCANF_ERROR_TOO_MANY_OUTPUTS, STR2NUM_DESCRIPTOR,
    STR2NUM_ERROR_TOO_MANY_OUTPUTS, STRTOK_DESCRIPTOR, STRTOK_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    encode_utf16_code_units, CharArray, IntValue, LogicalArray, NumericScalar, StringArray, Tensor,
    Utf16CodeUnit, Utf8Text, Value,
};

use crate::builtins::common::broadcast as matlab_broadcast;
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::common::{
    char_row_to_string_slice, contains_numeric_or_resident_text_input, scalar_text_input,
    text_input_admission, TextInput,
};
use crate::builtins::strings::core::numeric_length::parse_nonnegative_usize;
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};

pub(crate) fn compat_error(name: &str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn decoded_char_row(array: &CharArray, row: usize, fn_name: &str) -> BuiltinResult<String> {
    char_row_to_string_slice(array.data(), array.cols, row)
        .map_err(|error| compat_error(fn_name, format!("{fn_name}: {error}")))
}

fn map_flow(name: &'static str) -> impl Fn(crate::RuntimeError) -> crate::RuntimeError {
    move |err| map_control_flow_with_builtin(err, name)
}

#[runtime_builtin(
    name = "strtok",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn strtok_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    text: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &STRTOK_DESCRIPTOR,
        &STRTOK_ERROR_TOO_MANY_OUTPUTS,
        "strtok",
    )?;
    if rest.len() > 1 {
        return Err(compat_error(
            "strtok",
            "strtok: expected text and optional delimiters",
        ));
    }
    if contains_numeric_or_resident_text_input(&text)
        || rest
            .first()
            .is_some_and(contains_numeric_or_resident_text_input)
    {
        return Err(compat_error(
            "strtok",
            "strtok: expected host text and a text delimiter",
        ));
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("strtok"))?;
    let delimiter_owner = if let Some(value) = rest.first() {
        Some(
            gather_if_needed_async(value)
                .await
                .map_err(map_flow("strtok"))?,
        )
    } else {
        None
    };
    let delimiters = if let Some(value) = delimiter_owner.as_ref() {
        scalar_text_input(value, runtime.as_ref(), |runtime, cause| {
            text_input_admission("strtok", runtime, cause)
        })?
        .ok_or_else(|| compat_error("strtok", "strtok: expected text delimiters"))?
    } else {
        TextInput::Borrowed(" \t\n\r")
    };
    let (tokens, remainders) =
        map_text_pair_preserve(&text, "strtok", |s| strtok_pair(s, delimiters.as_str()))?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![tokens])),
        Some(n) => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![tokens, remainders],
        )),
        None => Ok(tokens),
    }
}

#[runtime_builtin(
    name = "str2num",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn str2num_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    text: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_bounded_outputs(
        &STR2NUM_DESCRIPTOR,
        &STR2NUM_ERROR_TOO_MANY_OUTPUTS,
        "str2num",
    )?;
    let method = str2num_evaluation_method(runtime.as_ref(), rest.view())?;
    if contains_numeric_or_resident_text_input(&text) {
        return Err(compat_error(
            "str2num",
            "str2num: expected a string scalar or character array",
        ));
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("str2num"))?;
    let text = scalar_text_input(&text, runtime.as_ref(), |runtime, cause| {
        text_input_admission("str2num", runtime, cause)
    })?
    .ok_or_else(|| compat_error("str2num", "str2num: expected a text scalar"))?;
    let runtime = runtime.ok_or_else(|| {
        crate::context::RuntimeCapabilityError::new(
            crate::context::RuntimeCapability::SourceEvaluation,
            "evaluate str2num numeric expression",
        )
        .into_runtime_error()
    })?;
    let request = crate::context::SessionSourceEvaluation::prepare_numeric_text(
        &runtime,
        text.as_str(),
        method,
    )?;
    let result = request.evaluate(runtime).await?;
    result.try_map_owned(
        |sequence| {
            let value = sequence.as_slice().first().cloned();
            let ok = value.is_some();
            Ok(str2num_outputs(
                value.unwrap_or_else(|| Value::Tensor(Tensor::zeros(vec![0, 0]))),
                ok,
            ))
        },
        |error| error.into_runtime_error(),
    )
}

fn str2num_evaluation_method(
    runtime: Option<&crate::context::RuntimeContext>,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<crate::context::NumericTextEvaluation> {
    use crate::context::NumericTextEvaluation;
    match (rest.len(), rest.get(0), rest.get(1)) {
        (0, _, _) => Ok(NumericTextEvaluation::All),
        (2, Some(name), Some(value)) => {
            let name = scalar_text_input(name, runtime, |runtime, cause| {
                text_input_admission("str2num", runtime, cause)
            })?
            .ok_or_else(|| compat_error("str2num", "str2num: expected a text scalar"))?;
            if !name.as_str().eq_ignore_ascii_case("Evaluation") {
                return Err(compat_error(
                    "str2num",
                    "str2num: expected one Evaluation name-value option",
                ));
            }
            let method = scalar_text_input(value, runtime, |runtime, cause| {
                text_input_admission("str2num", runtime, cause)
            })?
            .ok_or_else(|| compat_error("str2num", "str2num: expected a text scalar"))?;
            if method.as_str().eq_ignore_ascii_case("all") {
                Ok(NumericTextEvaluation::All)
            } else if method.as_str().eq_ignore_ascii_case("restricted") {
                Ok(NumericTextEvaluation::Restricted)
            } else {
                Err(compat_error(
                    "str2num",
                    "str2num: Evaluation must be 'all' or 'restricted'",
                ))
            }
        }
        _ => Err(compat_error(
            "str2num",
            "str2num: expected one Evaluation name-value option",
        )),
    }
}

fn str2num_outputs(value: Value, ok: bool) -> Value {
    match crate::output_count::current_output_count() {
        Some(0) => Value::output_list(Vec::new()),
        Some(1) => Value::output_list(vec![value]),
        Some(n) => crate::output_count::output_list_with_padding(n, vec![value, Value::Bool(ok)]),
        None => value,
    }
}

#[runtime_builtin(
    name = "mat2str",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn mat2str_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &MAT2STR_DESCRIPTOR,
        &MAT2STR_ERROR_TOO_MANY_OUTPUTS,
        "mat2str",
    )?;
    if rest.len() > 2 {
        return Err(compat_error(
            "mat2str",
            "mat2str: expected A, A,n, A,'class', or A,n,'class'",
        ));
    }
    reject_explicit_gpu_text_sink(&value, "mat2str", "input array")?;
    if matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MAT2STR_TEXT_INPUT_EXTENSION,
            "mat2str",
        )?;
    }

    let mut precision = None;
    let mut include_class = false;
    for (index, arg) in rest.iter().enumerate() {
        reject_explicit_gpu_text_sink(arg, "mat2str", "precision")?;
        if let Some(keyword) = scalar_text_input(arg, runtime.as_ref(), |runtime, cause| {
            text_input_admission("mat2str", runtime, cause)
        })? {
            if keyword.as_str().eq_ignore_ascii_case("class") && index + 1 == rest.len() {
                include_class = true;
                continue;
            }
            return Err(compat_error(
                "mat2str",
                "mat2str: the only supported text option is final 'class'",
            ));
        }
        if precision.is_some() || include_class {
            return Err(compat_error(
                "mat2str",
                "mat2str: precision must precede the optional 'class' flag",
            ));
        }
        if is_typed_integer_value(arg) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &MAT2STR_INTEGER_PRECISION_EXTENSION,
                "mat2str",
            )?;
        }
        let arg = gather_if_needed_async(arg)
            .await
            .map_err(map_flow("mat2str"))?;
        let parsed = parse_nonnegative_usize(&arg, "mat2str")?;
        if parsed == 0 {
            return Err(compat_error(
                "mat2str",
                "mat2str: precision must be a positive integer scalar",
            ));
        }
        precision = Some(parsed);
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_flow("mat2str"))?;
    Ok(Value::CharArray(CharArray::new_row(&mat2str_value(
        &value,
        precision,
        include_class,
    )?)))
}

#[runtime_builtin(
    name = "native2unicode",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn native2unicode_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    bytes: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &NATIVE2UNICODE_DESCRIPTOR,
        &NATIVE2UNICODE_ERROR_TOO_MANY_OUTPUTS,
        "native2unicode",
    )?;
    if rest.len() > 1 {
        return Err(compat_error(
            "native2unicode",
            "native2unicode: expected bytes and optional encoding",
        ));
    }
    reject_explicit_gpu_text_sink(&bytes, "native2unicode", "byte vector")?;
    let bytes = gather_if_needed_async(&bytes)
        .await
        .map_err(map_flow("native2unicode"))?;
    if matches!(&bytes, Value::String(_) | Value::CharArray(_))
        || matches!(&bytes, Value::StringArray(array) if array.data().len() == 1)
    {
        return Ok(bytes);
    }
    let encoding_owner = if let Some(value) = rest.first() {
        reject_explicit_gpu_text_sink(value, "native2unicode", "encoding")?;
        Some(
            gather_if_needed_async(value)
                .await
                .map_err(map_flow("native2unicode"))?,
        )
    } else {
        None
    };
    let encoding = if let Some(value) = encoding_owner.as_ref() {
        scalar_text_input(value, runtime.as_ref(), |runtime, cause| {
            text_input_admission("native2unicode", runtime, cause)
        })?
        .ok_or_else(|| compat_error("native2unicode", "native2unicode: expected text encoding"))?
    } else {
        TextInput::Borrowed("UTF-8")
    };
    let shape = byte_vector_shape(&bytes, "native2unicode")?;
    let bytes = bytes_from_value(&bytes, "native2unicode")?;
    decode_bytes(&bytes, encoding.as_str(), shape)
}

fn reject_explicit_gpu_text_sink(value: &Value, name: &str, role: &str) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        return Err(compat_error(
            name,
            format!("{name}: explicit gpuArray {role} is not supported"),
        ));
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

#[runtime_builtin(
    name = "sscanf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn sscanf_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    text: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &SSCANF_DESCRIPTOR,
        &SSCANF_ERROR_TOO_MANY_OUTPUTS,
        "sscanf",
    )?;
    if rest.len() > 2 {
        return Err(compat_error(
            "sscanf",
            "sscanf: expected text, format, and optional size",
        ));
    }
    if contains_numeric_or_resident_text_input(&text)
        || rest
            .first()
            .is_some_and(contains_numeric_or_resident_text_input)
    {
        return Err(compat_error(
            "sscanf",
            "sscanf: expected host text and a host text format",
        ));
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("sscanf"))?;
    let text = scalar_text_input(&text, runtime.as_ref(), |runtime, cause| {
        text_input_admission("sscanf", runtime, cause)
    })?
    .ok_or_else(|| compat_error("sscanf", "sscanf: expected a text scalar"))?;
    let format_owner = if let Some(format) = rest.first() {
        Some(
            gather_if_needed_async(format)
                .await
                .map_err(map_flow("sscanf"))?,
        )
    } else {
        None
    };
    let format = if let Some(format) = format_owner.as_ref() {
        scalar_text_input(format, runtime.as_ref(), |runtime, cause| {
            text_input_admission("sscanf", runtime, cause)
        })?
        .ok_or_else(|| compat_error("sscanf", "sscanf: expected a text scalar"))?
    } else {
        TextInput::Borrowed("%f")
    };
    let size = if let Some(size) = rest.get(1) {
        let size = gather_if_needed_async(size)
            .await
            .map_err(map_flow("sscanf"))?;
        Some(scan_size_from_value(&size)?)
    } else {
        None
    };
    let scan = sscanf_scan(text.as_str(), format.as_str(), size)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![scan.value])),
        Some(n) => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![
                scan.value,
                Value::Num(scan.count as f64),
                Value::CharArray(CharArray::new_row(&scan.errmsg)),
                Value::Num(scan.next_index as f64),
            ],
        )),
        None => Ok(scan.value),
    }
}

pub(crate) fn scalar_text(value: &Value, fn_name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => text
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| compat_error(fn_name, format!("{fn_name}: text must not be missing"))),
        Value::StringArray(array) if array.data().len() == 1 => array.data()[0]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| compat_error(fn_name, format!("{fn_name}: text must not be missing"))),
        Value::CharArray(array) if array.rows == 0 => Ok(String::new()),
        Value::CharArray(array) if array.rows == 1 => decoded_char_row(array, 0, fn_name),
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected a text scalar, got {other:?}"),
        )),
    }
}

pub(crate) fn pattern_regex(value: &Value, fn_name: &str) -> BuiltinResult<String> {
    match value {
        Value::Object(object) if object.is_class(runmat_types::standard::PATTERN) => {
            match object.properties().get("Regex") {
                Some(Value::String(regex)) => regex.as_str().map(str::to_owned).ok_or_else(|| {
                    compat_error(fn_name, format!("{fn_name}: pattern must not be missing"))
                }),
                _ => Err(compat_error(
                    fn_name,
                    format!("{fn_name}: invalid pattern object"),
                )),
            }
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Ok(regex::escape(&scalar_text(value, fn_name)?))
        }
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected text or pattern, got {other:?}"),
        )),
    }
}

pub(crate) fn pattern_object_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    regex: &str,
) -> BuiltinResult<Value> {
    let fields = [(
        "Regex",
        Value::String(crate::call::argument_text::copy_str(regex, runtime)?.into()),
    )];
    crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        runmat_types::standard::PATTERN.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?
    .with_owned(Ok)
}

#[cfg(test)]
pub(crate) fn pattern_object(regex: &str) -> Value {
    pattern_object_in_context(None, regex).expect("ordinary test pattern")
}

pub(crate) fn text_items(value: Value, fn_name: &str) -> BuiltinResult<TextList> {
    TextList::from_value(value, fn_name)
}

pub(crate) fn logical_value(
    data: Vec<u8>,
    shape: Vec<usize>,
    fn_name: &str,
) -> BuiltinResult<Value> {
    if data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| compat_error(fn_name, format!("{fn_name}: {e}")))
    }
}

pub(crate) struct TextList {
    pub(crate) items: Vec<Option<Utf8Text>>,
    pub(crate) shape: Vec<usize>,
}

impl TextList {
    pub(crate) fn from_value(value: Value, fn_name: &str) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Ok(Self {
                items: vec![missing_to_none(text)],
                shape: vec![1, 1],
            }),
            Value::StringArray(array) => Ok(Self {
                items: array.data().iter().cloned().map(missing_to_none).collect(),
                shape: array.shape().to_vec(),
            }),
            Value::CharArray(array) => {
                let mut items = Vec::with_capacity(array.rows.max(1));
                if array.rows == 0 {
                    return Ok(Self {
                        items,
                        shape: vec![0, 1],
                    });
                }
                for row in 0..array.rows {
                    items.push(Some(Utf8Text::new(decoded_char_row(&array, row, fn_name)?)));
                }
                Ok(Self {
                    items,
                    shape: vec![array.rows, 1],
                })
            }
            Value::Cell(cell) => {
                let mut items = Vec::with_capacity(cell.data().len());
                for value in cell.data() {
                    items.push(match value {
                        Value::String(text) => text.clone().into_text(),
                        Value::StringArray(array) if array.data().len() == 1 => {
                            array.data()[0].clone().into_text()
                        }
                        _ => Some(Utf8Text::new(scalar_text(value, fn_name)?)),
                    });
                }
                Ok(Self {
                    items,
                    shape: cell.shape().to_vec(),
                })
            }
            other => Err(compat_error(
                fn_name,
                format!("{fn_name}: expected text input, got {other:?}"),
            )),
        }
    }
}

pub(crate) fn broadcast_shape(
    a: &[usize],
    b: &[usize],
    fn_name: &str,
) -> BuiltinResult<Vec<usize>> {
    matlab_broadcast::broadcast_shapes(fn_name, a, b).map_err(|err| compat_error(fn_name, err))
}

pub(crate) fn broadcast_flat_index(
    linear: usize,
    shape: &[usize],
    source_shape: &[usize],
) -> usize {
    if source_shape.iter().product::<usize>() <= 1 {
        return 0;
    }
    let extended = matlab_broadcast::align_shape(source_shape, shape.len());
    let strides = matlab_broadcast::compute_strides(&extended);
    matlab_broadcast::broadcast_index(linear, shape, &extended, &strides)
}

fn missing_to_none(text: runmat_value::StringElement) -> Option<Utf8Text> {
    text.into_text()
}

fn map_text_pair_preserve(
    value: &Value,
    fn_name: &str,
    map: impl Fn(&str) -> (String, String) + Copy,
) -> BuiltinResult<(Value, Value)> {
    match value {
        Value::String(text) => {
            let Some(text) = text.as_str() else {
                return Ok((
                    Value::String(runmat_value::StringElement::Missing),
                    Value::String(runmat_value::StringElement::Missing),
                ));
            };
            let (first, second) = map(text);
            Ok((Value::string(first), Value::string(second)))
        }
        Value::StringArray(array) => {
            let shape = array.shape().to_vec();
            let mut first = Vec::with_capacity(array.data().len());
            let mut second = Vec::with_capacity(array.data().len());
            for text in array.data() {
                match text.as_str() {
                    Some(text) => {
                        let (a, b) = map(text);
                        first.push(runmat_value::StringElement::text(a));
                        second.push(runmat_value::StringElement::text(b));
                    }
                    None => {
                        first.push(runmat_value::StringElement::Missing);
                        second.push(runmat_value::StringElement::Missing);
                    }
                }
            }
            let first = StringArray::from_elements_unbounded(first, shape.clone())
                .map(Value::StringArray)
                .map_err(|e| compat_error(fn_name, e))?;
            let second = StringArray::from_elements_unbounded(second, shape)
                .map(Value::StringArray)
                .map_err(|e| compat_error(fn_name, e))?;
            Ok((first, second))
        }
        Value::CharArray(array) => {
            let mut first = Vec::with_capacity(array.rows);
            let mut second = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let text = decoded_char_row(array, row, fn_name)?;
                let (a, b) = map(&text);
                first.push(a);
                second.push(b);
            }
            Ok((
                char_rows_from_strings(first, fn_name)?,
                char_rows_from_strings(second, fn_name)?,
            ))
        }
        Value::Cell(cell) => {
            let shape = cell.shape().to_vec();
            let mut first = Vec::with_capacity(cell.data().len());
            let mut second = Vec::with_capacity(cell.data().len());
            for value in cell.data() {
                let (a, b) = map_text_pair_preserve(value, fn_name, map)?;
                first.push(a);
                second.push(b);
            }
            Ok((
                make_cell_with_shape(first, shape.clone()).map_err(|e| compat_error(fn_name, e))?,
                make_cell_with_shape(second, shape).map_err(|e| compat_error(fn_name, e))?,
            ))
        }
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

fn strtok_pair(text: &str, delimiters: &str) -> (String, String) {
    let start = text
        .char_indices()
        .find_map(|(idx, ch)| (!delimiters.contains(ch)).then_some(idx))
        .unwrap_or(text.len());
    if start == text.len() {
        return (String::new(), String::new());
    }
    let token_end = text[start..]
        .char_indices()
        .find_map(|(idx, ch)| delimiters.contains(ch).then_some(start + idx))
        .unwrap_or(text.len());
    (
        text[start..token_end].to_string(),
        text[token_end..].to_string(),
    )
}

fn char_rows_from_strings(rows: Vec<String>, fn_name: &str) -> BuiltinResult<Value> {
    let row_count = rows.len();
    let encoded = rows
        .iter()
        .map(|row| encode_utf16_code_units(row))
        .collect::<Vec<_>>();
    let cols = encoded.iter().map(Vec::len).max().unwrap_or(0);
    let mut data = Vec::with_capacity(row_count * cols);
    for mut row in encoded {
        row.resize(cols, Utf16CodeUnit::from_raw(b' ' as u16));
        data.extend(row);
    }
    CharArray::from_code_units(data, row_count, cols)
        .map(Value::CharArray)
        .map_err(|e| compat_error(fn_name, e))
}

fn mat2str_value(
    value: &Value,
    precision: Option<usize>,
    include_class: bool,
) -> BuiltinResult<String> {
    let body = match value {
        Value::Num(n) => format_number(*n, precision),
        Value::Int(i) => i.decimal_string(),
        Value::Bool(b) => {
            if *b {
                "true".into()
            } else {
                "false".into()
            }
        }
        Value::String(text) => match text.as_str() {
            Some(text) => format!("\"{}\"", text.replace('"', "\"\"")),
            None => "missing".to_owned(),
        },
        Value::CharArray(array) if array.rows <= 1 => {
            format!(
                "'{}'",
                decoded_char_row(array, 0, "mat2str")?.replace('\'', "''")
            )
        }
        Value::Tensor(tensor) => tensor_to_matrix_string(tensor, precision),
        Value::LogicalArray(array) => {
            let rows = array.shape.first().copied().unwrap_or(array.data.len());
            let cols = array.shape.get(1).copied().unwrap_or(1);
            let data = array
                .data
                .iter()
                .map(|v| f64::from(*v != 0))
                .collect::<Vec<_>>();
            matrix_to_string(&data, rows, cols, precision)
        }
        _ => value.to_string(),
    };
    if !include_class {
        return Ok(body);
    }
    Ok(match mat2str_class_name(value) {
        Some(class) => format!("{class}({body})"),
        None => body,
    })
}

fn mat2str_class_name(value: &Value) -> Option<&'static str> {
    match value {
        Value::Num(_) => Some("double"),
        Value::Int(value) => Some(value.class_name()),
        Value::Bool(_) | Value::LogicalArray(_) => Some("logical"),
        Value::Tensor(tensor) => Some(tensor.numeric_dtype().class_name()),
        _ => None,
    }
}

fn matrix_to_string(data: &[f64], rows: usize, cols: usize, precision: Option<usize>) -> String {
    matrix_to_string_with(rows, cols, |index| format_number(data[index], precision))
}

fn tensor_to_matrix_string(tensor: &Tensor, precision: Option<usize>) -> String {
    matrix_to_string_with(tensor.rows(), tensor.cols(), |index| {
        let value = tensor
            .numeric_value_at(index)
            .expect("validated tensor storage index");
        match value {
            NumericScalar::F64(value) => format_number(value, precision),
            NumericScalar::F32(value) => format_number(f64::from(value), precision),
            integer => integer
                .into_int_value()
                .expect("non-floating numeric scalar is integer")
                .decimal_string(),
        }
    })
}

fn matrix_to_string_with(
    rows: usize,
    cols: usize,
    mut format_value: impl FnMut(usize) -> String,
) -> String {
    let mut out = String::from("[");
    for row in 0..rows {
        if row > 0 {
            out.push(';');
        }
        for col in 0..cols {
            if col > 0 {
                out.push(' ');
            }
            out.push_str(&format_value(row + col * rows));
        }
    }
    out.push(']');
    out
}

fn format_number(value: f64, precision: Option<usize>) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f64::INFINITY {
        return "Inf".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-Inf".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let digits = precision.unwrap_or(15).max(1);
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= digits as i32 {
        let decimals = digits.saturating_sub(1);
        return normalize_scientific(format!("{value:.decimals$e}"));
    }
    let decimals = (digits as i32 - 1 - exponent).max(0) as usize;
    let fixed = trim_decimal(format!("{value:.decimals$}"));
    if fixed
        .parse::<f64>()
        .ok()
        .is_some_and(|rounded| rounded != 0.0 && rounded.abs().log10().floor() >= digits as f64)
    {
        let decimals = digits.saturating_sub(1);
        normalize_scientific(format!("{value:.decimals$e}"))
    } else {
        fixed
    }
}

fn trim_decimal(mut value: String) -> String {
    if value.contains('.') {
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
    }
    value
}

fn normalize_scientific(value: String) -> String {
    let Some((mantissa, exponent)) = value.split_once('e') else {
        return value;
    };
    let mantissa = trim_decimal(mantissa.to_string());
    let exponent = exponent.parse::<i32>().unwrap_or(0);
    format!("{mantissa}e{exponent:+03}")
}

fn bytes_from_value(value: &Value, fn_name: &str) -> BuiltinResult<Vec<u8>> {
    match value {
        Value::Tensor(tensor) => (0..tensor.len())
            .map(|index| {
                let value = tensor.numeric_value_at(index).ok_or_else(|| {
                    compat_error(
                        fn_name,
                        format!("{fn_name}: numeric storage is inconsistent"),
                    )
                })?;
                match value {
                    NumericScalar::F64(value) => byte_from_f64(value, fn_name),
                    NumericScalar::F32(value) => byte_from_f64(f64::from(value), fn_name),
                    integer => byte_from_intvalue(
                        &integer
                            .into_int_value()
                            .expect("non-floating numeric scalar is integer"),
                        fn_name,
                    ),
                }
            })
            .collect(),
        Value::Int(i) => Ok(vec![byte_from_intvalue(i, fn_name)?]),
        Value::Num(n) => Ok(vec![byte_from_f64(*n, fn_name)?]),
        Value::CharArray(array) => Ok(decoded_char_row(array, 0, fn_name)?.into_bytes()),
        Value::String(text) => text
            .as_str()
            .map(|text| text.as_bytes().to_vec())
            .ok_or_else(|| compat_error(fn_name, format!("{fn_name}: text must not be missing"))),
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected bytes or text, got {other:?}"),
        )),
    }
}

fn byte_from_intvalue(value: &IntValue, fn_name: &str) -> BuiltinResult<u8> {
    value
        .try_to_u64()
        .filter(|value| *value <= u8::MAX as u64)
        .map(|value| value as u8)
        .ok_or_else(|| {
            compat_error(
                fn_name,
                format!("{fn_name}: byte values must be in the range 0 through 255"),
            )
        })
}

fn byte_from_f64(value: f64, fn_name: &str) -> BuiltinResult<u8> {
    if !value.is_finite() {
        return Err(compat_error(
            fn_name,
            format!("{fn_name}: byte values must be finite"),
        ));
    }
    if !(0.0..=255.0).contains(&value) {
        return Err(compat_error(
            fn_name,
            format!("{fn_name}: byte values must be in the range 0 through 255"),
        ));
    }
    Ok(value.round() as u8)
}

fn decode_bytes(bytes: &[u8], encoding: &str, shape: Vec<usize>) -> BuiltinResult<Value> {
    let encoding = Encoding::for_label(encoding.as_bytes()).ok_or_else(|| {
        compat_error(
            "native2unicode",
            format!("native2unicode: unsupported encoding '{encoding}'"),
        )
    })?;
    let (text, _, _) = encoding.decode(bytes);
    let chars = encode_utf16_code_units(text.as_ref());
    let output = if shape.iter().product::<usize>() == chars.len() {
        CharArray::from_column_major(chars, shape)
    } else {
        Ok(CharArray::new_row(text.as_ref()))
    }
    .map_err(|error| compat_error("native2unicode", error))?;
    Ok(Value::CharArray(output))
}

fn byte_vector_shape(value: &Value, fn_name: &str) -> BuiltinResult<Vec<usize>> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape().to_vec(),
        Value::Int(_) | Value::Num(_) => vec![1, 1],
        other => {
            return Err(compat_error(
                fn_name,
                format!("{fn_name}: expected a numeric byte vector, got {other:?}"),
            ))
        }
    };
    if shape.iter().filter(|&&extent| extent > 1).count() > 1 {
        return Err(compat_error(
            fn_name,
            format!("{fn_name}: byte input must be a vector"),
        ));
    }
    Ok(match shape.as_slice() {
        [] => vec![1, 1],
        [length] => vec![1, *length],
        _ => shape,
    })
}

mod scan;
use self::scan::{scan_size_from_value, sscanf_scan};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::strings::core::{
        blanks::blanks_builtin, convert_chars_to_strings::convert_chars_to_strings_builtin,
        convert_contained_strings_to_chars::convert_contained_strings_to_chars_builtin,
        convert_strings_to_chars::convert_strings_to_chars_builtin,
        digits_pattern::digits_pattern_builtin, is_string_scalar::is_string_scalar_builtin,
        isletter::isletter_builtin, isspace::isspace_builtin, isstrprop::isstrprop_builtin,
        letters_pattern::letters_pattern_builtin, newline::newline_builtin,
        pattern::pattern_builtin, regexp_pattern::regexp_pattern_builtin,
        strncmpi::strncmpi_builtin as strncmpi_with_metadata, text_boundary::text_boundary_builtin,
        wildcard_pattern::wildcard_pattern_builtin,
    };
    use runmat_builtins::{
        BLANKS_GPU_INPUT_EXTENSION, DIGITS_PATTERN_DESCRIPTOR, DIGITS_PATTERN_ERROR_ARGUMENT_COUNT,
        DIGITS_PATTERN_ERROR_INVALID_COUNT, DIGITS_PATTERN_ERROR_INVALID_RANGE,
        DIGITS_PATTERN_ERROR_TOO_MANY_OUTPUTS, DIGITS_PATTERN_INTEGER_CAPABILITIES,
        ISLETTER_ERROR_TOO_MANY_OUTPUTS, ISSPACE_ERROR_TOO_MANY_OUTPUTS,
        ISSTRPROP_ERROR_TOO_MANY_OUTPUTS, ISSTRPROP_RESIDENT_INPUT_EXTENSION,
        LETTERS_PATTERN_ERROR_TOO_MANY_OUTPUTS, PATTERN_ERROR_TOO_MANY_OUTPUTS,
        REGEXP_PATTERN_ERROR_TOO_MANY_OUTPUTS, STRNCMPI_ERROR_TOO_MANY_OUTPUTS,
        TEXT_BOUNDARY_ERROR_TOO_MANY_OUTPUTS, WILDCARD_PATTERN_ERROR_TOO_MANY_OUTPUTS,
    };
    use runmat_value::{CellArray, IntValue, IntegerStorage, NumericDType};

    fn strncmpi_builtin(
        a: Value,
        b: Value,
        n: Value,
    ) -> impl std::future::Future<Output = BuiltinResult<Value>> {
        strncmpi_with_metadata(None, None, a, b, n)
    }

    #[test]
    fn text_broadcast_helpers_append_trailing_singletons() {
        assert_eq!(
            broadcast_shape(&[2, 1], &[2, 1, 3], "test").unwrap(),
            vec![2, 1, 3]
        );
        assert!(broadcast_shape(&[2, 3], &[1, 2, 3], "test").is_err());
        assert_eq!(
            (0..6)
                .map(|linear| broadcast_flat_index(linear, &[2, 1, 3], &[2, 1]))
                .collect::<Vec<_>>(),
            vec![0, 1, 0, 1, 0, 1]
        );
    }

    #[test]
    fn mat2str_preserves_exact_uint64_scalar_text() {
        assert_eq!(
            mat2str_value(
                &Value::Int(runmat_value::IntValue::U64(u64::MAX)),
                None,
                false,
            )
            .unwrap(),
            "18446744073709551615"
        );
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .expect("typed integer matrix");
        assert_eq!(
            mat2str_value(&Value::Tensor(tensor), None, false).unwrap(),
            "[18446744073709551615 9007199254740993]"
        );

        let typed = Tensor::new_integer(IntegerStorage::U16(vec![256, 512]), vec![1, 2])
            .expect("typed integer matrix");
        assert_eq!(
            mat2str_value(&Value::Tensor(typed), None, true).unwrap(),
            "uint16([256 512])"
        );
    }

    #[test]
    fn mat2str_precision_is_significant_digits_and_positive() {
        assert_eq!(format_number(std::f64::consts::PI, Some(3)), "3.14");
        assert_eq!(format_number(12_345.0, Some(3)), "1.23e+04");
        assert_eq!(format_number(0.000_012_345, Some(3)), "1.23e-05");
        assert_eq!(format_number(999.9, Some(3)), "1e+03");
        let error = block(mat2str_builtin(
            None,
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
        ))
        .expect_err("zero precision must reject");
        assert!(error.message().contains("positive"));
    }

    #[test]
    fn mat2str_typed_precision_and_text_input_follow_extension_policy() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block(mat2str_builtin(
            None,
            Value::Int(IntValue::U16(12)),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U8(3),
            )]),
        ))
        .expect_err("typed precision is extension-gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Mat2strIntegerPrecisionExtension")
        );
        let error = block(mat2str_builtin(
            None,
            Value::String("x".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("text input is extension-gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Mat2strTextInputExtension")
        );
    }

    #[test]
    fn bounded_count_parsers_preserve_typed_integer_values() {
        assert_eq!(
            parse_nonnegative_usize(&Value::Int(IntValue::U64(7)), "digitsPattern").unwrap(),
            7
        );
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![9]), vec![1, 1])
            .expect("typed scalar tensor");
        assert_eq!(
            parse_nonnegative_usize(&Value::Tensor(tensor), "digitsPattern").unwrap(),
            9
        );
        let pattern = block(digits_pattern_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U8(3),
            )]),
        ))
        .expect("typed count");
        assert_eq!(pattern_regex(&pattern, "test").unwrap(), r"\d{3}");

        for value in [
            Value::Num(1.5),
            Value::Num(usize::MAX as f64 + 1.0),
            Value::Int(IntValue::I8(-1)),
        ] {
            assert!(parse_nonnegative_usize(&value, "digitsPattern").is_err());
        }
    }

    #[test]
    fn sscanf_size_parsing_preserves_typed_integer_dimensions() {
        assert_eq!(
            scan_size_from_value(&Value::Int(IntValue::U16(4))).unwrap(),
            vec![4, 1]
        );
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2])
            .expect("typed size vector");
        assert_eq!(
            scan_size_from_value(&Value::Tensor(tensor)).unwrap(),
            vec![2, 3]
        );
        let single = Tensor::from_f32(vec![5.0, 6.0], vec![1, 2]).expect("single size vector");
        assert_eq!(
            scan_size_from_value(&Value::Tensor(single)).unwrap(),
            vec![5, 6]
        );
        assert_eq!(
            scan_size_from_value(&Value::Num(f64::INFINITY)).unwrap(),
            vec![usize::MAX / 2, 1]
        );
        let open_columns = Tensor::new(vec![2.0, f64::INFINITY], vec![1, 2]).unwrap();
        assert_eq!(
            scan_size_from_value(&Value::Tensor(open_columns)).unwrap(),
            vec![2, usize::MAX / 2]
        );
        let open_rows = Tensor::new(vec![f64::INFINITY, 2.0], vec![1, 2]).unwrap();
        assert!(scan_size_from_value(&Value::Tensor(open_rows)).is_err());

        for value in [
            Value::Num(1.5),
            Value::Num(usize::MAX as f64 + 1.0),
            Value::Int(IntValue::I16(-1)),
            Value::Num(0.0),
            Value::Int(IntValue::U8(0)),
        ] {
            assert!(scan_size_from_value(&value).is_err());
        }
    }

    #[test]
    fn sscanf_size_stops_scanning_and_pads_only_the_last_column() {
        for (size, expected, expected_count, expected_next) in [
            (vec![2, 1], vec![2.0, 3.0], 2, 4),
            (vec![2, 2], vec![2.0, 3.0, 4.0, 5.0], 4, 8),
        ] {
            let scan = sscanf_scan("2 3 4 5 bad", "%f", Some(size.clone())).unwrap();
            assert_eq!(scan.count, expected_count);
            assert_eq!(scan.next_index, expected_next);
            assert!(scan.errmsg.is_empty(), "unread input must not diagnose");
            assert_eq!(
                scan.value,
                Value::Tensor(Tensor::new(expected, size).unwrap())
            );
        }

        let partial = sscanf_scan("2 3 4", "%f", Some(vec![2, 2])).unwrap();
        assert_eq!(partial.count, 3);
        assert_eq!(partial.next_index, 6);
        assert!(partial.errmsg.is_empty());
        assert_eq!(
            partial.value,
            Value::Tensor(Tensor::new(vec![2.0, 3.0, 4.0, 0.0], vec![2, 2]).unwrap())
        );
        let short = sscanf_scan("2", "%f", Some(vec![2, 2])).unwrap();
        assert_eq!(
            short.value,
            Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap())
        );
        let failed_before_limit = sscanf_scan("2 x", "%f", Some(vec![2, 2])).unwrap();
        assert_eq!(failed_before_limit.count, 1);
        assert_eq!(failed_before_limit.next_index, 3);
        assert!(!failed_before_limit.errmsg.is_empty());

        let typed = sscanf_scan("7 8 9", "%ld", Some(vec![2, 2])).unwrap();
        assert_eq!(typed.count, 3);
        let Value::Tensor(typed) = typed.value else {
            panic!("expected typed tensor");
        };
        assert_eq!(typed.shape(), &[2, 2]);
        assert_eq!(
            typed.integer_storage(),
            Some(&IntegerStorage::I64(vec![7, 8, 9, 0]))
        );

        for (input, format) in [("skip 42", "%*s %f"), ("abc42", "%*3c%f")] {
            let scan = sscanf_scan(input, format, Some(vec![1, 1])).unwrap();
            assert_eq!(scan.count, 1);
            assert_eq!(scan.next_index, input.chars().count() + 1);
            assert!(scan.errmsg.is_empty());
            assert_eq!(
                scan.value,
                Value::Tensor(Tensor::new(vec![42.0], vec![1, 1]).unwrap())
            );
        }
    }

    #[test]
    fn sscanf_character_fields_keep_character_class_and_field_limit() {
        let astral_units = sscanf_scan("🙂X", "%1c%1c", Some(vec![2, 1])).unwrap();
        assert_eq!(astral_units.count, 2);
        assert_eq!(astral_units.next_index, 3);
        let Value::CharArray(astral_chars) = astral_units.value else {
            panic!("expected character output");
        };
        assert_eq!(
            astral_chars
                .data()
                .iter()
                .map(|unit| unit.raw())
                .collect::<Vec<_>>(),
            vec![0xD83D, 0xDE42],
        );
        let width_one = sscanf_scan("🙂X", "%1s", Some(vec![1, 1])).unwrap();
        assert_eq!(width_one.count, 1);
        assert_eq!(width_one.next_index, 2);
        let Value::CharArray(width_one) = width_one.value else {
            panic!("expected character output");
        };
        assert_eq!(width_one.data()[0].raw(), 0xD83D);

        for (input, format, expected, next_index) in [
            ("hello world", "%s", "hello", 6),
            ("abcde", "%3c", "abc", 4),
            ("abc", "%5c", "abc", 4),
            ("skip done", "%*s %s", "done", 10),
        ] {
            let scan = sscanf_scan(input, format, Some(vec![1, 1])).unwrap();
            assert_eq!(scan.value, Value::CharArray(CharArray::new_row(expected)));
            assert_eq!(scan.count, expected.chars().count());
            assert_eq!(scan.next_index, next_index);
            assert!(scan.errmsg.is_empty());
        }

        for (text, format) in [("abcdef efgh", "%s"), ("abcdef", "%6c")] {
            let matrix = sscanf_scan(text, format, Some(vec![2, 2])).unwrap();
            assert_eq!(matrix.count, 4);
            assert_eq!(matrix.next_index, 5, "{format}");
            assert!(matrix.errmsg.is_empty());
            assert_eq!(
                matrix.value,
                Value::CharArray(
                    CharArray::from_column_major(
                        runmat_value::encode_utf16_code_units("abcd"),
                        vec![2, 2]
                    )
                    .unwrap()
                )
            );
        }

        let partial = sscanf_scan("abc", "%s", Some(vec![2, 2])).unwrap();
        assert_eq!(partial.count, 3);
        assert_eq!(
            partial.value,
            Value::CharArray(
                CharArray::from_column_major(
                    runmat_value::encode_utf16_code_units("abc\0"),
                    vec![2, 2]
                )
                .unwrap()
            )
        );

        let mixed = sscanf_scan("7 ABZ", "%*d %s", Some(vec![2, 1])).unwrap();
        assert_eq!(mixed.count, 2);
        assert_eq!(mixed.next_index, 5);
        assert_eq!(
            mixed.value,
            Value::Tensor(Tensor::new(vec![65.0, 66.0], vec![2, 1]).unwrap())
        );
        let mixed_direct = sscanf_scan("7 Z", "%d %c", None).unwrap();
        assert_eq!(
            mixed_direct.value,
            Value::Tensor(Tensor::new(vec![7.0, 90.0], vec![2, 1]).unwrap())
        );
    }

    #[test]
    fn sscanf_long_integer_formats_preserve_full_width_values() {
        let signed = sscanf_scan("-9223372036854775808 9223372036854775807", "%ld", None)
            .expect("signed long scan");
        let Value::Tensor(signed) = signed.value else {
            panic!("expected signed tensor");
        };
        assert_eq!(signed.numeric_dtype(), NumericDType::I64);
        assert_eq!(
            signed.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))
        );

        let unsigned = sscanf_scan("18446744073709551615 ff 177", "%lu %lx %lo", None)
            .expect("unsigned long scan");
        let Value::Tensor(unsigned) = unsigned.value else {
            panic!("expected unsigned tensor");
        };
        assert_eq!(unsigned.numeric_dtype(), NumericDType::U64);
        assert_eq!(
            unsigned.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 255, 127]))
        );
    }

    #[test]
    fn sscanf_mixed_long_integer_formats_follow_double_output_rule() {
        let result = sscanf_scan("42 7.5", "%ld %f", None).expect("mixed scan");
        let Value::Tensor(result) = result.value else {
            panic!("expected tensor");
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F64);
        assert_eq!(result.materialize_f64(), vec![42.0, 7.5]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn native2unicode_numeric_low_rank_shapes_remain_scalar_and_row() {
        let scalar = block(native2unicode_builtin(
            None,
            Value::Tensor(Tensor::new(vec![104.0], Vec::new()).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("rank-zero byte");
        assert_eq!(scalar, Value::CharArray(CharArray::new_row("h")));

        let row = block(native2unicode_builtin(
            None,
            Value::Tensor(Tensor::new(vec![104.0, 105.0], vec![2]).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("rank-one bytes");
        assert_eq!(row, Value::CharArray(CharArray::new_row("hi")));

        let scalar_text = Value::StringArray(
            StringArray::new(vec!["hé".into()], vec![1, 1]).expect("string scalar"),
        );
        assert_eq!(
            block(native2unicode_builtin(
                None,
                scalar_text.clone(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .expect("documented string-scalar pass-through"),
            scalar_text
        );
        let array_text = Value::StringArray(
            StringArray::new(vec!["a".into(), "b".into()], vec![1, 2])
                .expect("nonscalar string array"),
        );
        let error = block(native2unicode_builtin(
            None,
            array_text,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("nonscalar string array must not pass through");
        assert!(error.message().contains("numeric byte vector"));
    }

    #[test]
    fn native2unicode_reads_typed_integer_byte_storage_exactly() {
        let bytes = Tensor::new_integer(IntegerStorage::U8(vec![104, 105]), vec![1, 2]).unwrap();
        assert_eq!(
            block(native2unicode_builtin(
                None,
                Value::Tensor(bytes),
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("hi"))
        );
        let single = Tensor::from_f32(vec![111.0, 107.0], vec![1, 2]).expect("single bytes");
        assert_eq!(
            block(native2unicode_builtin(
                None,
                Value::Tensor(single),
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("ok"))
        );

        for value in [IntValue::I16(-1), IntValue::U16(256)] {
            let error = block(native2unicode_builtin(
                None,
                Value::Int(value),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("out-of-range byte");
            assert!(error.message().contains("0 through 255"));
        }

        let column = Tensor::new_integer(IntegerStorage::U8(vec![104, 105]), vec![2, 1])
            .expect("column bytes");
        let Value::CharArray(decoded) = block(native2unicode_builtin(
            None,
            Value::Tensor(column),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap() else {
            panic!("expected character vector");
        };
        assert_eq!(decoded.shape, vec![2, 1]);
    }

    fn block<T>(value: impl std::future::Future<Output = BuiltinResult<T>>) -> BuiltinResult<T> {
        futures::executor::block_on(value)
    }

    #[test]
    fn basic_text_core_helpers_work() {
        assert_eq!(newline_builtin().unwrap(), Value::String("\n".into()));
        assert_eq!(
            block(blanks_builtin(Value::Num(3.0))).unwrap(),
            Value::CharArray(CharArray::new_row("   "))
        );
        assert_eq!(
            is_string_scalar_builtin(Value::String("x".into())).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn blanks_accepts_every_integer_class_and_clamps_negative_lengths() {
        for value in [
            IntValue::I8(3),
            IntValue::I16(3),
            IntValue::I32(3),
            IntValue::I64(3),
            IntValue::U8(3),
            IntValue::U16(3),
            IntValue::U32(3),
            IntValue::U64(3),
        ] {
            assert_eq!(
                block(blanks_builtin(Value::Int(value))).expect("integer length"),
                Value::CharArray(CharArray::new_row("   "))
            );
        }
        for value in [
            Value::Num(-3.0),
            Value::Int(IntValue::I8(-3)),
            Value::Int(IntValue::I16(-3)),
            Value::Int(IntValue::I32(-3)),
            Value::Int(IntValue::I64(-3)),
        ] {
            assert_eq!(
                block(blanks_builtin(value)).expect("negative length"),
                Value::CharArray(CharArray::new_row(""))
            );
        }
        let single = Tensor::from_f32(vec![3.0], vec![1, 1]).expect("single scalar");
        assert_eq!(
            block(blanks_builtin(Value::Tensor(single))).expect("single length"),
            Value::CharArray(CharArray::new_row("   "))
        );
        assert!(block(blanks_builtin(Value::Num(1.5))).is_err());
        assert!(block(blanks_builtin(Value::Int(IntValue::U64(u64::MAX)))).is_err());
    }

    #[test]
    fn blanks_gpu_input_follows_compatibility_mode() {
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&[3]),
                    shape: &[1, 1],
                })
                .expect("upload integer length");
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block(blanks_builtin(Value::GpuTensor(handle.clone())))
                    .expect_err("strict mode rejects resident length");
                assert_eq!(
                    error.identifier(),
                    BLANKS_GPU_INPUT_EXTENSION.error_identifier
                );
            }
            {
                let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
                assert_eq!(
                    block(blanks_builtin(Value::GpuTensor(handle))).expect("RunMat extension"),
                    Value::CharArray(CharArray::new_row("   "))
                );
            }
        });
    }

    #[test]
    fn strncmpi_and_classifiers_work() {
        assert_eq!(
            block(strncmpi_builtin(
                Value::String("RunMat".into()),
                Value::String("runway".into()),
                Value::Num(3.0),
            ))
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            block(isletter_builtin(Value::CharArray(CharArray::new_row("a1")))).unwrap(),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap())
        );
        assert_eq!(
            block(isletter_builtin(Value::Int(runmat_value::IntValue::U64(
                u64::MAX,
            ))))
            .unwrap(),
            Value::Bool(false)
        );
        let integer = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::from(b'A'), u64::from(b'1'), u64::MAX]),
            vec![1, 3],
        )
        .unwrap();
        assert_eq!(
            block(isstrprop_builtin(
                None,
                Value::Tensor(integer),
                Value::String("alpha".into()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap())
        );
        assert_eq!(
            block(isspace_builtin(Value::Int(IntValue::U16(32)))).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn migrated_classifiers_reject_excess_outputs_before_execution() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let strncmpi_error = block(strncmpi_builtin(
            Value::String("ab".into()),
            Value::String("AB".into()),
            Value::Num(2.0),
        ))
        .expect_err("strncmpi has one output");
        assert_eq!(
            strncmpi_error.identifier(),
            STRNCMPI_ERROR_TOO_MANY_OUTPUTS.identifier
        );
        let isletter_error = block(isletter_builtin(Value::String("a".into())))
            .expect_err("isletter has one output");
        assert_eq!(
            isletter_error.identifier(),
            ISLETTER_ERROR_TOO_MANY_OUTPUTS.identifier
        );
        let isspace_error =
            block(isspace_builtin(Value::String(" ".into()))).expect_err("isspace has one output");
        assert_eq!(
            isspace_error.identifier(),
            ISSPACE_ERROR_TOO_MANY_OUTPUTS.identifier
        );
        let isstrprop_error = block(isstrprop_builtin(
            None,
            Value::String("a".into()),
            Value::String("alpha".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("isstrprop has one output");
        assert_eq!(
            isstrprop_error.identifier(),
            ISSTRPROP_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }

    #[test]
    fn strncmpi_short_prefix_and_missing_behavior_remain_explicit() {
        assert_eq!(
            block(strncmpi_builtin(
                Value::String("ab".into()),
                Value::String("AB".into()),
                Value::Num(3.0),
            ))
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            block(strncmpi_builtin(
                Value::String(runmat_value::StringElement::Missing),
                Value::String("x".into()),
                Value::Num(0.0),
            ))
            .unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn strncmpi_gathers_automatic_resident_integer_count() {
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            let count = [3_u64];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&count),
                    shape: &[1, 1],
                })
                .expect("upload exact count")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("automatic input");
            let result = block(strncmpi_builtin(
                Value::String("RunMat".into()),
                Value::String("runway".into()),
                Value::GpuTensor(handle.clone()),
            ))
            .expect("gather N");
            assert_eq!(result, Value::Bool(true));
            handle
                .allocation_lease()
                .validate_live()
                .expect("input remains live");
        });
    }

    #[test]
    fn isstrprop_only_gates_explicit_resident_integer_input() {
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&[u16::from(b'A'), u16::from(b'1')]),
                    shape: &[1, 2],
                })
                .unwrap();
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                assert_eq!(
                    block(isstrprop_builtin(
                        None,
                        Value::GpuTensor(handle.clone()),
                        Value::String("alpha".into()),
                        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                    ))
                    .unwrap(),
                    Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap())
                );
            }
            let handle = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("admit provenance view");
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block(isstrprop_builtin(
                    None,
                    Value::GpuTensor(handle.clone()),
                    Value::String("alpha".into()),
                    crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                ))
                .unwrap_err();
                assert_eq!(
                    error.identifier(),
                    ISSTRPROP_RESIDENT_INPUT_EXTENSION.error_identifier
                );
            }
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert_eq!(
                block(isstrprop_builtin(
                    None,
                    Value::GpuTensor(handle),
                    Value::String("alpha".into()),
                    crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                ))
                .unwrap(),
                Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap())
            );
        });
    }

    #[test]
    fn isstrprop_force_cell_output_wraps_numeric_result() {
        let result = block(isstrprop_builtin(
            None,
            Value::Int(IntValue::U16(u16::from(b'A'))),
            Value::String("alpha".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("ForceCellOutput".into()),
                Value::Bool(true),
            ]),
        ))
        .expect("documented ForceCellOutput form");
        assert!(matches!(
            result,
            Value::Cell(cell)
                if cell.shape == vec![1, 1]
                    && cell.data() == vec![Value::Bool(true)]
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn isstrprop_numeric_low_rank_classification_scalarizes_only_one_element() {
        let scalar = block(isstrprop_builtin(
            None,
            Value::Tensor(Tensor::new(vec![65.0], Vec::new()).unwrap()),
            Value::String("alpha".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("rank-zero character code");
        assert_eq!(scalar, Value::Bool(true));

        let vector = block(isstrprop_builtin(
            None,
            Value::Tensor(Tensor::new(vec![65.0, 49.0], vec![2]).unwrap()),
            Value::String("alpha".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("rank-one character codes");
        assert_eq!(
            vector,
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![2]).unwrap())
        );
    }

    #[test]
    fn isstrprop_numeric_codes_use_unicode_categories_and_reject_surrogates() {
        let codes = Tensor::new_integer(
            IntegerStorage::U16(vec![0x0661, 0x2014, 0xd800]),
            vec![1, 3],
        )
        .expect("Unicode code units");
        let digit = block(isstrprop_builtin(
            None,
            Value::Tensor(codes.clone()),
            Value::String("digit".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("Unicode decimal classification");
        assert_eq!(
            digit,
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap())
        );
        let punctuation = block(isstrprop_builtin(
            None,
            Value::Tensor(codes),
            Value::String("punct".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("Unicode punctuation classification");
        assert_eq!(
            punctuation,
            Value::LogicalArray(LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap())
        );
    }

    #[test]
    fn isspace_accepts_string_scalars_and_rejects_string_arrays() {
        let scalar = StringArray::new(vec!["a b".into()], vec![1, 1]).expect("string scalar");
        assert_eq!(
            block(isspace_builtin(Value::StringArray(scalar))).expect("string scalar"),
            Value::LogicalArray(LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap())
        );

        let array = StringArray::new(vec![" ".into(), "x".into()], vec![1, 2])
            .expect("nonscalar string array");
        let error = block(isspace_builtin(Value::StringArray(array)))
            .expect_err("nonscalar string arrays reject");
        assert!(error.message().contains("string scalar"));
    }

    #[test]
    fn conversions_and_numeric_parsing_work() {
        assert_eq!(
            block(convert_strings_to_chars_builtin(
                Value::String("abc".into()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("abc"))
        );
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        assert!(matches!(
            block(convert_strings_to_chars_builtin(
                Value::String("a".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String("b".into())]),
            ))
            .unwrap(),
            Value::OutputList(outputs) if outputs.len() == 2
        ));
        drop(_guard);
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        assert!(matches!(
            block(convert_chars_to_strings_builtin(
                Value::CharArray(CharArray::new_row("a")),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::CharArray(CharArray::new_row("b"))]),
            ))
            .unwrap(),
            Value::OutputList(outputs) if outputs.as_slice() == [Value::String("a".into()), Value::String("b".into())]
        ));
        assert!(matches!(
            block(convert_contained_strings_to_chars_builtin(
                Value::String("a".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String("b".into())]),
            ))
            .unwrap(),
            Value::OutputList(outputs) if outputs.as_slice() == [Value::CharArray(CharArray::new_row("a")), Value::CharArray(CharArray::new_row("b"))]
        ));
        drop(_guard);
        assert_eq!(
            block(convert_chars_to_strings_builtin(
                Value::CharArray(CharArray::new_row("abc")),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::String("abc".into())
        );
        let missing = block(str2num_builtin(
            None,
            Value::String("1 2; 3 4".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("expression evaluation needs an executor-owned source service");
        assert_eq!(
            missing.identifier(),
            Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
        );
        let bad_method = block(str2num_builtin(
            None,
            Value::String("1 2".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Evaluation".into()),
                Value::String("other".into()),
            ]),
        ))
        .expect_err("invalid Evaluation method must precede text evaluation");
        assert!(bad_method.message().contains("Evaluation must be"));
        assert_eq!(
            block(mat2str_builtin(
                None,
                Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("[1 2;3 4]"))
        );
    }

    #[test]
    fn conversion_output_selection_preserves_existing_padding_and_zero_output_behavior() {
        {
            let _outputs =
                crate::output_count::push_output_count(Some(3)).expect("output-count scope");
            let result = block(convert_strings_to_chars_builtin(
                Value::String("a".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "b".into(),
                )]),
            ))
            .unwrap();
            assert_eq!(
                result,
                Value::output_list(vec![
                    Value::CharArray(CharArray::new_row("a")),
                    Value::CharArray(CharArray::new_row("b")),
                    Value::Num(0.0),
                ])
            );
        }
        {
            let _outputs =
                crate::output_count::push_output_count(Some(0)).expect("output-count scope");
            let result = block(convert_chars_to_strings_builtin(
                Value::CharArray(CharArray::new_row("a")),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::CharArray(
                    CharArray::new_row("b"),
                )]),
            ))
            .unwrap();
            assert_eq!(result, Value::output_list(Vec::new()));
        }
    }

    #[test]
    fn integer_conversion_passthrough_is_exact_for_every_class() {
        for int in [
            IntValue::I8(i8::MIN),
            IntValue::I16(i16::MIN),
            IntValue::I32(i32::MIN),
            IntValue::I64(i64::MIN),
            IntValue::U8(u8::MAX),
            IntValue::U16(u16::MAX),
            IntValue::U32(u32::MAX),
            IntValue::U64(u64::MAX),
        ] {
            let value = Value::Int(int);
            assert_eq!(
                block(convert_strings_to_chars_builtin(
                    value.clone(),
                    crate::builtins::common::test_support::builtin_arguments(Vec::new())
                ))
                .unwrap(),
                value
            );
            assert_eq!(
                block(convert_chars_to_strings_builtin(
                    value.clone(),
                    crate::builtins::common::test_support::builtin_arguments(Vec::new())
                ))
                .unwrap(),
                value
            );
            assert_eq!(
                block(convert_contained_strings_to_chars_builtin(
                    value.clone(),
                    crate::builtins::common::test_support::builtin_arguments(Vec::new())
                ))
                .unwrap(),
                value
            );
        }
    }

    #[test]
    fn integer_conversion_passthrough_preserves_provider_ownership_without_gather() {
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[u64::MAX]),
                    shape: &[1, 1],
                })
                .expect("upload integer");
            for output in [
                block(convert_strings_to_chars_builtin(
                    Value::GpuTensor(handle.clone()),
                    crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                ))
                .unwrap(),
                block(convert_chars_to_strings_builtin(
                    Value::GpuTensor(handle.clone()),
                    crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                ))
                .unwrap(),
                block(convert_contained_strings_to_chars_builtin(
                    Value::GpuTensor(handle.clone()),
                    crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                ))
                .unwrap(),
            ] {
                let Value::GpuTensor(returned) = output else {
                    panic!("expected unchanged resident integer handle");
                };
                assert_eq!(returned.buffer_id(), handle.buffer_id());
                assert_eq!(returned.device_id(), handle.device_id());
                assert_eq!(returned.shape(), handle.shape());
                assert_eq!(returned.view_identity(), handle.view_identity());
                assert!(returned.shares_allocation_with(&handle));
                handle.allocation_lease().validate_live().unwrap();
            }
        });
    }

    #[test]
    fn conversion_scope_and_container_shapes_match_their_contracts() {
        let matrix = CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        assert_eq!(
            block(convert_chars_to_strings_builtin(
                Value::CharArray(matrix),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::String("acbd".into())
        );

        let cellstr = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("a")),
                Value::CharArray(CharArray::new_row("b")),
            ],
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            block(convert_chars_to_strings_builtin(
                Value::Cell(cellstr),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap())
        );

        let mixed = Value::Cell(
            CellArray::new(
                vec![
                    Value::String("keep".into()),
                    Value::Int(IntValue::U64(u64::MAX)),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        assert_eq!(
            block(convert_strings_to_chars_builtin(
                mixed.clone(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .unwrap(),
            mixed
        );

        assert_eq!(
            block(convert_contained_strings_to_chars_builtin(
                Value::String("top-level".into()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("top-level"))
        );
        for text in [
            runmat_value::StringElement::text(""),
            runmat_value::StringElement::Missing,
        ] {
            assert_eq!(
                block(convert_strings_to_chars_builtin(
                    Value::String(text.into()),
                    crate::builtins::common::test_support::builtin_arguments(Vec::new())
                ))
                .unwrap(),
                Value::CharArray(CharArray::from_unicode_scalars(Vec::new(), 0, 0).unwrap())
            );
        }
        let converted = block(convert_contained_strings_to_chars_builtin(
            mixed,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        let Value::Cell(cell) = converted else {
            panic!("expected cell");
        };
        assert_eq!(
            cell.data(),
            vec![
                Value::CharArray(CharArray::new_row("keep")),
                Value::Int(IntValue::U64(u64::MAX))
            ]
        );

        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("admit unregistered resident fixture")
                .handle,
        );
        let nested = Value::Cell(CellArray::new(vec![resident.clone()], 1, 1).unwrap());
        let Value::Cell(preserved) = block(convert_contained_strings_to_chars_builtin(
            nested,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap() else {
            panic!("expected cell");
        };
        assert_eq!(preserved.data(), vec![resident]);
    }

    #[test]
    fn convert_contained_strings_maps_typed_structure_arrays() {
        let mut first = runmat_value::UnboundedStructValueBuilder::new();
        first.insert("label", Value::String("first".into()));
        let mut second = runmat_value::UnboundedStructValueBuilder::new();
        second.insert("label", Value::String("second".into()));
        let second = second.finish();
        let first = first.finish();
        let array = runmat_value::StructArray::new(vec![first, second], vec![2, 1]).unwrap();

        let converted = block(convert_contained_strings_to_chars_builtin(
            Value::StructArray(array),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        let Value::StructArray(converted) = converted else {
            panic!("expected structure array");
        };
        assert_eq!(converted.shape(), [2, 1]);
        assert!(matches!(
            converted.get_linear(0).unwrap().fields.get("label"),
            Some(Value::CharArray(value))
                if value.try_row_string().unwrap().as_deref() == Some("first")
        ));
        assert!(matches!(
            converted.get_linear(1).unwrap().fields.get("label"),
            Some(Value::CharArray(value))
                if value.try_row_string().unwrap().as_deref() == Some("second")
        ));
    }

    #[test]
    fn tokenizing_encoding_and_scanning_work() {
        assert_eq!(
            block(strtok_builtin(
                None,
                Value::String("  alpha,beta".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    " ,".into()
                )]),
            ))
            .unwrap(),
            Value::String("alpha".into())
        );
        assert_eq!(
            block(native2unicode_builtin(
                None,
                Value::Tensor(
                    Tensor::new_with_dtype(vec![104.0, 105.0], vec![1, 2], NumericDType::U8)
                        .unwrap()
                ),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("hi"))
        );
        assert_eq!(
            block(sscanf_builtin(
                None,
                Value::String("1 2 x".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "%f".into()
                )]),
            ))
            .unwrap(),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap())
        );
    }

    #[test]
    fn scanning_next_index_includes_consumed_prefix_on_partial_match() {
        for (text, format, expected_count, expected_next) in [
            ("1 2 x", "%f", 2, 5),
            ("1 x", "%f %f", 1, 3),
            ("é 7 x", "é %d %d", 1, 5),
        ] {
            let scan = sscanf_scan(text, format, None).unwrap();
            assert_eq!(scan.count, expected_count, "{text}");
            assert_eq!(scan.next_index, expected_next, "{text}");
            assert!(!scan.errmsg.is_empty(), "{text}");
        }
        let complete = sscanf_scan("1 2 ", "%f", None).unwrap();
        assert_eq!(complete.next_index, 5);
        assert!(complete.errmsg.is_empty());
    }

    #[test]
    fn migrated_text_output_bounds_precede_input_materialization() {
        {
            let _outputs =
                crate::output_count::push_output_count(Some(3)).expect("output-count scope");
            let strtok = block(strtok_builtin(
                None,
                Value::Num(7.0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("strtok must reject excess outputs first");
            assert_eq!(
                strtok.identifier(),
                STRTOK_ERROR_TOO_MANY_OUTPUTS.identifier
            );
            let str2num = block(str2num_builtin(
                None,
                Value::Num(7.0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("str2num must reject excess outputs first");
            assert_eq!(
                str2num.identifier(),
                STR2NUM_ERROR_TOO_MANY_OUTPUTS.identifier
            );
        }
        {
            let _outputs =
                crate::output_count::push_output_count(Some(5)).expect("output-count scope");
            let sscanf = block(sscanf_builtin(
                None,
                Value::Num(7.0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("sscanf must reject excess outputs first");
            assert_eq!(
                sscanf.identifier(),
                SSCANF_ERROR_TOO_MANY_OUTPUTS.identifier
            );
        }
        {
            let _outputs =
                crate::output_count::push_output_count(Some(2)).expect("output-count scope");
            let mat2str = block(mat2str_builtin(
                None,
                Value::Num(7.0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("mat2str must reject excess outputs first");
            assert_eq!(
                mat2str.identifier(),
                MAT2STR_ERROR_TOO_MANY_OUTPUTS.identifier
            );
        }
    }

    #[test]
    fn pattern_constructors_store_regex() {
        let Value::Object(pattern) = pattern_object("x") else {
            panic!("pattern constructor must return an object");
        };
        assert!(pattern.is_class(runmat_types::standard::PATTERN));
        assert_eq!(
            pattern_regex(
                &block(digits_pattern_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(Vec::new())
                ))
                .unwrap(),
                "test"
            )
            .unwrap(),
            r"\d+"
        );
        let value = block(digits_pattern_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
        ))
        .unwrap();
        assert_eq!(pattern_regex(&value, "test").unwrap(), "\\d{2}");
        let bounded = block(digits_pattern_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::U8(2)),
                Value::Int(IntValue::U16(4)),
            ]),
        ))
        .unwrap();
        assert_eq!(pattern_regex(&bounded, "test").unwrap(), r"\d{2,4}");
        let unbounded = block(digits_pattern_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::I8(3)),
                Value::Num(f64::INFINITY),
            ]),
        ))
        .unwrap();
        assert_eq!(pattern_regex(&unbounded, "test").unwrap(), r"\d{3,}");
        assert_eq!(
            pattern_regex(
                &block(letters_pattern_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(Vec::new())
                ))
                .unwrap(),
                "test"
            )
            .unwrap(),
            r"\p{Alphabetic}+"
        );
        assert_eq!(
            pattern_regex(
                &block(wildcard_pattern_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(Vec::new())
                ))
                .unwrap(),
                "test"
            )
            .unwrap(),
            ".*?"
        );
        assert_eq!(
            pattern_regex(
                &block(text_boundary_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(Vec::new())
                ))
                .unwrap(),
                "test"
            )
            .unwrap(),
            r"(?:^|$)"
        );
        assert_eq!(
            pattern_regex(
                &block(text_boundary_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                        "start".into()
                    )])
                ))
                .unwrap(),
                "test"
            )
            .unwrap(),
            r"^"
        );
        assert_eq!(
            pattern_regex(
                &block(text_boundary_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                        "end".into()
                    )])
                ))
                .unwrap(),
                "test"
            )
            .unwrap(),
            r"$"
        );
    }

    #[test]
    fn pattern_constructors_reject_extra_outputs_before_reading_inputs() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        for (result, identifier) in [
            (
                block(pattern_builtin(None, Value::Num(7.0))),
                PATTERN_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
            (
                block(regexp_pattern_builtin(None, Value::Num(7.0))),
                REGEXP_PATTERN_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
            (
                block(digits_pattern_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                        -1.0,
                    )]),
                )),
                DIGITS_PATTERN_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
            (
                block(letters_pattern_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                        -1.0,
                    )]),
                )),
                LETTERS_PATTERN_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
            (
                block(wildcard_pattern_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                        -1.0,
                    )]),
                )),
                WILDCARD_PATTERN_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
            (
                block(text_boundary_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(7.0)]),
                )),
                TEXT_BOUNDARY_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
        ] {
            assert_eq!(result.expect_err("excess outputs").identifier(), identifier);
        }
    }

    #[test]
    fn digits_pattern_validates_range_controls_and_argument_count() {
        let zero = block(digits_pattern_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U64(0),
            )]),
        ))
        .unwrap();
        assert_eq!(pattern_regex(&zero, "test").unwrap(), r"\d{0}");
        for args in [
            vec![Value::Num(-1.0)],
            vec![Value::Num(1.5)],
            vec![Value::Num(1.0), Value::Num(f64::NEG_INFINITY)],
            vec![Value::Bool(true)],
        ] {
            let error = block(digits_pattern_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                DIGITS_PATTERN_ERROR_INVALID_COUNT.identifier
            );
        }
        let range_error = block(digits_pattern_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(4.0),
                Value::Num(3.0),
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            range_error.identifier(),
            DIGITS_PATTERN_ERROR_INVALID_RANGE.identifier
        );
        let arity_error = block(digits_pattern_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            arity_error.identifier(),
            DIGITS_PATTERN_ERROR_ARGUMENT_COUNT.identifier
        );
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("admit unregistered resident fixture")
                .handle,
        );
        let resident_error = block(digits_pattern_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![resident]),
        ))
        .unwrap_err();
        assert_eq!(
            resident_error.identifier(),
            DIGITS_PATTERN_ERROR_INVALID_COUNT.identifier
        );
    }

    #[test]
    fn digits_pattern_range_reads_all_integer_classes_exactly() {
        for (minimum, maximum) in [
            (IntValue::I8(2), IntValue::I8(4)),
            (IntValue::I16(2), IntValue::I16(4)),
            (IntValue::I32(2), IntValue::I32(4)),
            (IntValue::I64(2), IntValue::I64(4)),
            (IntValue::U8(2), IntValue::U8(4)),
            (IntValue::U16(2), IntValue::U16(4)),
            (IntValue::U32(2), IntValue::U32(4)),
            (IntValue::U64(2), IntValue::U64(4)),
        ] {
            let pattern = block(digits_pattern_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Int(minimum),
                    Value::Int(maximum),
                ]),
            ))
            .unwrap();
            assert_eq!(pattern_regex(&pattern, "test").unwrap(), r"\d{2,4}");
        }
    }

    #[test]
    fn digits_pattern_descriptor_declares_all_public_forms() {
        assert_eq!(DIGITS_PATTERN_DESCRIPTOR.signatures.len(), 3);
        assert_eq!(DIGITS_PATTERN_DESCRIPTOR.errors.len(), 4);
        assert_eq!(DIGITS_PATTERN_INTEGER_CAPABILITIES.len(), 2);
    }

    #[test]
    fn text_boundary_rejects_invalid_type() {
        let err = block(text_boundary_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "middle".into(),
            )]),
        ))
        .expect_err("expected invalid boundary type");
        assert!(err.to_string().contains("unsupported boundary type"));
    }

    #[test]
    fn text_boundary_pattern_works_with_replace() {
        let pattern = block(text_boundary_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "start".into(),
            )]),
        ))
        .unwrap();
        let result = futures::executor::block_on(crate::call_builtin_async(
            "replace",
            &[
                Value::String("abc".into()),
                pattern,
                Value::String(">".into()),
            ],
        ))
        .expect("replace");
        assert_eq!(result.value(), &Value::String(">abc".into()));
    }
}
