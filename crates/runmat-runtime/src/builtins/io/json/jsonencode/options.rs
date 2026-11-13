use super::*;

#[derive(Debug)]
pub(super) struct PreparedOptionLayout {
    entries: Vec<PreparedOption>,
}

#[derive(Debug)]
struct PreparedOption {
    name: JsonEncodeOptionName,
    location: OptionLocation,
}

#[derive(Debug)]
enum OptionLocation {
    StructField(String),
    PairValue(usize),
}

impl PreparedOption {
    fn value<'a>(&self, rest: crate::call::FunctionArgumentView<'a>) -> BuiltinResult<&'a Value> {
        match &self.location {
            OptionLocation::StructField(key) => match (rest.len(), rest.first()) {
                (1, Some(Value::Struct(options))) => options
                    .fields
                    .get(key)
                    .ok_or_else(|| jsonencode_error(&JSONENCODE_ERROR_INTERNAL)),
                _ => Err(jsonencode_error(&JSONENCODE_ERROR_INTERNAL)),
            },
            OptionLocation::PairValue(index) => rest
                .get(*index)
                .ok_or_else(|| jsonencode_error(&JSONENCODE_ERROR_INTERNAL)),
        }
    }
}

pub(super) fn preflight_jsonencode_extensions(
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    layout: &PreparedOptionLayout,
) -> BuiltinResult<()> {
    if value_contains_complex(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &JSONENCODE_COMPLEX_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if value_contains_sparse(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &JSONENCODE_SPARSE_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    preflight_option_extensions(layout, rest)?;
    if value_contains_explicit_gpu(value) || rest.iter().any(value_contains_explicit_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &JSONENCODE_RESIDENT_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

pub(super) fn prepare_option_layout(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<PreparedOptionLayout> {
    if rest.is_empty() {
        return Ok(PreparedOptionLayout {
            entries: Vec::new(),
        });
    }
    if rest.len() == 1 {
        return match &rest[0] {
            Value::Struct(options) => {
                let mut entries = Vec::with_capacity(options.fields.len());
                for name in options.fields.keys() {
                    entries.push(PreparedOption {
                        name: decode_option_name(name)?,
                        location: OptionLocation::StructField(name.to_owned()),
                    });
                }
                Ok(PreparedOptionLayout { entries })
            }
            _ => Err(jsonencode_error(&JSONENCODE_ERROR_OPTIONS_CONFIG)),
        };
    }
    if !rest.len().is_multiple_of(2) {
        return Err(jsonencode_error(&JSONENCODE_ERROR_NAME_VALUE_PAIRS));
    }
    let mut entries = Vec::with_capacity(rest.len() / 2);
    for (pair_index, pair) in rest.chunks_exact(2).enumerate() {
        let name = option_name(&pair[0])?;
        entries.push(PreparedOption {
            name: decode_option_name(&name)?,
            location: OptionLocation::PairValue(pair_index * 2 + 1),
        });
    }
    Ok(PreparedOptionLayout { entries })
}

fn decode_option_name(name: &str) -> BuiltinResult<JsonEncodeOptionName> {
    JsonEncodeOptionName::from_public_name(name).ok_or_else(|| {
        jsonencode_error_with(
            &JSONENCODE_ERROR_UNKNOWN_OPTION,
            format!("{} ('{}')", JSONENCODE_ERROR_UNKNOWN_OPTION.message, name),
        )
    })
}

pub(super) fn preflight_option_extensions(
    layout: &PreparedOptionLayout,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for entry in &layout.entries {
        let option_value = entry.value(rest)?;
        if !option_value_is_scalar(option_value) {
            return Err(jsonencode_error(&JSONENCODE_ERROR_OPTION_VALUE));
        }
        if is_typed_integer_option(option_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &JSONENCODE_TYPED_INTEGER_OPTION_EXTENSION,
                BUILTIN_NAME,
            )?;
        } else if is_floating_numeric_option(option_value)
            && !(entry.name == JsonEncodeOptionName::ConvertInfAndNaN
                && is_documented_numeric_bool(option_value))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &JSONENCODE_NUMERIC_OPTION_EXTENSION,
                BUILTIN_NAME,
            )?;
        } else if is_text_option(option_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &JSONENCODE_TEXT_OPTION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    Ok(())
}

fn option_value_is_scalar(value: &Value) -> bool {
    match value {
        Value::Bool(_) | Value::Int(_) | Value::Num(_) | Value::String(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::LogicalArray(logical) => logical.data.len() == 1,
        Value::CharArray(chars) => chars.rows == 1,
        Value::StringArray(strings) => strings.data().len() == 1,
        Value::GpuTensor(handle) => handle.shape().iter().all(|&extent| extent == 1),
        _ => false,
    }
}

fn is_documented_numeric_bool(value: &Value) -> bool {
    let number = match value {
        Value::Num(number) => Some(*number),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    };
    matches!(number, Some(0.0 | 1.0))
}

fn is_typed_integer_option(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn is_floating_numeric_option(value: &Value) -> bool {
    match value {
        Value::Num(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_none(),
        Value::GpuTensor(_) => false,
        _ => false,
    }
}

fn is_text_option(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn value_contains_complex(value: &Value) -> bool {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => true,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
        }
        Value::Cell(cell) => cell.data().iter().any(value_contains_complex),
        Value::Struct(struct_value) => struct_value.fields.values().any(value_contains_complex),
        Value::StructArray(array) => array.any_value(value_contains_complex),
        Value::Object(object) => object.properties().values().any(value_contains_complex),
        _ => false,
    }
}

fn value_contains_sparse(value: &Value) -> bool {
    match value {
        Value::SparseTensor(_) => true,
        Value::Cell(cell) => cell.data().iter().any(value_contains_sparse),
        Value::Struct(value) => value.fields.values().any(value_contains_sparse),
        Value::StructArray(array) => array.any_value(value_contains_sparse),
        Value::Object(value) => value.properties().values().any(value_contains_sparse),
        _ => false,
    }
}

fn value_contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data().iter().any(value_contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(value_contains_explicit_gpu),
        Value::StructArray(array) => array.any_value(value_contains_explicit_gpu),
        Value::Object(value) => value.properties().values().any(value_contains_explicit_gpu),
        _ => false,
    }
}

pub(super) fn parse_options(
    layout: &PreparedOptionLayout,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<JsonEncodeOptions> {
    let mut options = JsonEncodeOptions::default();
    for entry in &layout.entries {
        apply_option(entry.name, entry.value(args)?, &mut options)?;
    }
    Ok(options)
}

fn option_name(value: &Value) -> BuiltinResult<std::borrow::Cow<'_, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => {
            Ok(std::borrow::Cow::Borrowed(s.as_str()))
        }
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::common::text::decode_utf16_with_error(ca.data(), || {
                jsonencode_error(&JSONENCODE_ERROR_OPTION_NAME)
            })
            .map(std::borrow::Cow::Owned)
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            Ok(std::borrow::Cow::Borrowed(
                sa.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str(),
            ))
        }
        _ => Err(jsonencode_error(&JSONENCODE_ERROR_OPTION_NAME)),
    }
}

fn apply_option(
    name: JsonEncodeOptionName,
    value: &Value,
    options: &mut JsonEncodeOptions,
) -> BuiltinResult<()> {
    match name {
        JsonEncodeOptionName::PrettyPrint => {
            options.pretty_print = coerce_bool(value)?;
            Ok(())
        }
        JsonEncodeOptionName::ConvertInfAndNaN => {
            options.convert_inf_and_nan = coerce_bool(value)?;
            Ok(())
        }
    }
}

fn coerce_bool(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => bool_from_f64(*n),
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                    return Ok(!int.is_zero());
                }
                bool_from_f64(tensor::tensor_value_f64(t, 0))
            } else {
                Err(jsonencode_error(&JSONENCODE_ERROR_OPTION_VALUE))
            }
        }
        Value::LogicalArray(la) => match la.data.len() {
            1 => Ok(la.data[0] != 0),
            _ => Err(jsonencode_error(&JSONENCODE_ERROR_OPTION_VALUE)),
        },
        Value::CharArray(ca) if ca.rows == 1 => {
            let text = crate::builtins::common::text::decode_utf16_with_error(ca.data(), || {
                jsonencode_error(&JSONENCODE_ERROR_OPTION_VALUE)
            })?;
            parse_bool_string(&text)
        }
        Value::String(runmat_value::StringElement::Text(s)) => parse_bool_string(s),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            parse_bool_string(
                sa.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string"),
            )
        }
        _ => Err(jsonencode_error(&JSONENCODE_ERROR_OPTION_VALUE)),
    }
}

fn bool_from_f64(value: f64) -> BuiltinResult<bool> {
    if value.is_finite() {
        Ok(value != 0.0)
    } else {
        Err(jsonencode_error(&JSONENCODE_ERROR_OPTION_VALUE))
    }
}

fn parse_bool_string(text: &str) -> BuiltinResult<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        _ => Err(jsonencode_error(&JSONENCODE_ERROR_OPTION_VALUE)),
    }
}
