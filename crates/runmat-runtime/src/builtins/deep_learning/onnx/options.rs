use super::*;

pub(super) fn parse_export_options(
    map: std::collections::BTreeMap<String, Value>,
) -> BuiltinResult<ExportOptions> {
    let mut options = ExportOptions::default();
    for (key, value) in map {
        match key.as_str() {
            "opsetversion" => {
                let version = integer_scalar(&value, "OpsetVersion")?;
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
            "batchsize" => {
                options.batch_size = Some(integer_scalar(&value, "BatchSize")?);
            }
            "inputnames" => {
                options.input_name = Some(single_name(&value, "InputNames")?);
            }
            "outputnames" => {
                options.output_name = Some(single_name(&value, "OutputNames")?);
            }
            "verbose" => {
                let _ = logical_scalar(&value, EXPORT_ONNX, "Verbose")?;
            }
            other => {
                return Err(deep_learning_error(
                    EXPORT_ONNX,
                    format!("exportONNXNetwork: unsupported option '{other}'"),
                ));
            }
        }
    }
    Ok(options)
}

pub(super) fn validate_option(key: &str, value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_))
        && matches!(key, "opsetversion" | "batchsize" | "verbose")
    {
        return Ok(());
    }
    match key {
        "opsetversion" => {
            let version = integer_scalar(value, "OpsetVersion")?;
            if !(MIN_SUPPORTED_OPSET_VERSION..=MAX_SUPPORTED_OPSET_VERSION).contains(&version) {
                return Err(deep_learning_error(EXPORT_ONNX, format!("exportONNXNetwork: OpsetVersion must be between {MIN_SUPPORTED_OPSET_VERSION} and {MAX_SUPPORTED_OPSET_VERSION}")));
            }
        }
        "batchsize" => {
            integer_scalar(value, "BatchSize")?;
        }
        "inputnames" => {
            single_name(value, "InputNames")?;
        }
        "outputnames" => {
            single_name(value, "OutputNames")?;
        }
        "verbose" => {
            logical_scalar(value, EXPORT_ONNX, "Verbose")?;
        }
        other => {
            return Err(deep_learning_error(
                EXPORT_ONNX,
                format!("exportONNXNetwork: unsupported option '{other}'"),
            ))
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
        Value::StringArray(array) => array.data.clone(),
        Value::Cell(cell) => {
            let mut names = Vec::with_capacity(cell.data.len());
            for item in &cell.data {
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
