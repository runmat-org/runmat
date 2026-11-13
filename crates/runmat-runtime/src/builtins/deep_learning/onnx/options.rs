use super::*;
use std::collections::BTreeMap;

/// Keep the legacy lowercase/last-value-wins public parsing order. Unknown
/// names stay here until validation so the integer compatibility gate and
/// first-error ordering remain unchanged; no semantic consumer switches on
/// the retained text.
pub(super) struct ParsedExportOption {
    pub(super) name: String,
    pub(super) key: Option<ExportOnnxOptionKey>,
    pub(super) value: Value,
}

pub(super) fn parse_option_inputs(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Vec<ParsedExportOption>> {
    parse_name_values(args, EXPORT_ONNX).map(|values| {
        values
            .into_iter()
            .map(|(name, value)| ParsedExportOption {
                key: ExportOnnxOptionKey::from_name(&name),
                name,
                value,
            })
            .collect()
    })
}

pub(super) fn unsupported_option(name: &str) -> crate::RuntimeError {
    deep_learning_error(
        EXPORT_ONNX,
        format!("exportONNXNetwork: unsupported option '{name}'"),
    )
}

pub(super) fn parse_export_options(
    map: BTreeMap<ExportOnnxOptionKey, Value>,
) -> BuiltinResult<ExportOptions> {
    let mut options = ExportOptions::default();
    for (key, value) in map {
        match key {
            ExportOnnxOptionKey::OpsetVersion => {
                let version = integer_scalar(&value, key.canonical_name())?;
                if !(MIN_SUPPORTED_OPSET_VERSION..=MAX_SUPPORTED_OPSET_VERSION).contains(&version) {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        format!(
                            "exportONNXNetwork: OpsetVersion must be between {MIN_SUPPORTED_OPSET_VERSION} and {MAX_SUPPORTED_OPSET_VERSION}"
                        ),
                    ));
                }
                options.opset_version = version;
            }
            ExportOnnxOptionKey::BatchSize => {
                options.batch_size = Some(integer_scalar(&value, key.canonical_name())?);
            }
            ExportOnnxOptionKey::InputNames => {
                options.input_name = Some(single_name(&value, key.canonical_name())?);
            }
            ExportOnnxOptionKey::OutputNames => {
                options.output_name = Some(single_name(&value, key.canonical_name())?);
            }
            ExportOnnxOptionKey::Verbose => {
                let _ = logical_scalar(&value, EXPORT_ONNX, key.canonical_name())?;
            }
        }
    }
    Ok(options)
}

pub(super) fn validate_option(key: ExportOnnxOptionKey, value: &Value) -> BuiltinResult<()> {
    if let Value::GpuTensor(handle) = value {
        match key.value_role() {
            ExportOnnxOptionValueRole::SingleName => {
                // Names are host text. Fall through to the ordinary text
                // decoder, which reports the existing class error before
                // any resident value is downloaded.
            }
            role @ (ExportOnnxOptionValueRole::PositiveInteger
            | ExportOnnxOptionValueRole::LogicalFlag) => {
                let scalar = handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                    == Some(1);
                let numeric = runmat_accelerate_api::handle_precision(handle).is_some()
                    || runmat_accelerate_api::handle_integer_type(handle).is_some();
                let logical = runmat_accelerate_api::handle_is_logical(handle);
                let admitted_class = if role == ExportOnnxOptionValueRole::PositiveInteger {
                    numeric && !logical
                } else {
                    logical || numeric
                };
                if scalar
                    && runmat_accelerate_api::handle_storage(handle)
                        == runmat_accelerate_api::GpuTensorStorage::Real
                    && admitted_class
                {
                    return Ok(());
                }
                return Err(deep_learning_error(
                    EXPORT_ONNX,
                    format!(
                        "exportONNXNetwork: {} must have supported real scalar storage",
                        key.canonical_name().to_ascii_lowercase()
                    ),
                ));
            }
        }
    }
    match key {
        ExportOnnxOptionKey::OpsetVersion => {
            let version = integer_scalar(value, key.canonical_name())?;
            if !(MIN_SUPPORTED_OPSET_VERSION..=MAX_SUPPORTED_OPSET_VERSION).contains(&version) {
                return Err(deep_learning_error(EXPORT_ONNX, format!("exportONNXNetwork: OpsetVersion must be between {MIN_SUPPORTED_OPSET_VERSION} and {MAX_SUPPORTED_OPSET_VERSION}")));
            }
        }
        ExportOnnxOptionKey::BatchSize => {
            integer_scalar(value, key.canonical_name())?;
        }
        ExportOnnxOptionKey::InputNames => {
            single_name(value, key.canonical_name())?;
        }
        ExportOnnxOptionKey::OutputNames => {
            single_name(value, key.canonical_name())?;
        }
        ExportOnnxOptionKey::Verbose => {
            logical_scalar(value, EXPORT_ONNX, key.canonical_name())?;
        }
    }
    Ok(())
}

pub(in crate::builtins::deep_learning) fn integer_scalar(
    value: &Value,
    label: &str,
) -> BuiltinResult<i64> {
    positive_i64(value, EXPORT_ONNX, label)
}

fn single_name(value: &Value, label: &str) -> BuiltinResult<String> {
    let values = match value {
        Value::String(_) | Value::CharArray(_) => vec![scalar_text(value, EXPORT_ONNX)?],
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| scalar_text(&Value::String(text.clone()), EXPORT_ONNX))
            .collect::<BuiltinResult<Vec<_>>>()?,
        Value::Cell(cell) => {
            let mut names = Vec::with_capacity(cell.data().len());
            for item in cell.data() {
                names.push(scalar_text(item, EXPORT_ONNX)?);
            }
            names
        }
        other => {
            return Err(deep_learning_error(
                EXPORT_ONNX,
                format!("exportONNXNetwork: {label} must be text, string array, or cell text, got {other:?}"),
            ));
        }
    };
    match values.as_slice() {
        [name] if !name.is_empty() => Ok(name.clone()),
        [_] => Err(deep_learning_error(
            EXPORT_ONNX,
            format!("exportONNXNetwork: {label} must not be empty"),
        )),
        _ => Err(deep_learning_error(
            EXPORT_ONNX,
            format!("exportONNXNetwork: this exporter supports exactly one {label} value"),
        )),
    }
}
