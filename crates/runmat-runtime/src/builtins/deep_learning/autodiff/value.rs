use super::*;

pub(in crate::builtins::deep_learning) fn dlarray_format_and_labels(
    value: &Value,
) -> (Value, Value) {
    let Value::Object(object) = value else {
        return (Value::string(String::new()), Value::string(String::new()));
    };
    (
        object
            .properties()
            .get("Format")
            .cloned()
            .unwrap_or_else(|| Value::string(String::new())),
        object
            .properties()
            .get("Labels")
            .cloned()
            .unwrap_or_else(|| Value::string(String::new())),
    )
}

pub(in crate::builtins::deep_learning) fn dlarray_data(
    value: &Value,
    function: &'static str,
) -> BuiltinResult<Value> {
    match value {
        Value::Object(object) if object.class_name().is(DLARRAY_CLASS) => {
            object.properties().get("Data").cloned().ok_or_else(|| {
                deep_learning_error(function, format!("{function}: dlarray is missing Data"))
            })
        }
        other => Ok(other.clone()),
    }
}

pub(in crate::builtins::deep_learning) fn host_tensor_from_value(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Tensor> {
    match value {
        Value::Object(object) if object.class_name().is(DLARRAY_CLASS) => {
            let data = object.properties().get("Data").ok_or_else(|| {
                deep_learning_error(function, format!("{function}: dlarray is missing Data"))
            })?;
            host_tensor_from_value(data, function, label)
        }
        Value::GpuTensor(_) => Err(unsupported_error(
            function,
            format!("{function}: automatic differentiation for GPU-backed dlarray values requires provider-resident tape kernels"),
        )),
        other => tensor::value_into_tensor_for(function, other.clone())
            .map_err(|err| deep_learning_error(function, format!("{function}: {label} {err}"))),
    }
}

#[derive(Clone)]
pub(super) struct Payload {
    pub(super) tensor: Tensor,
    pub(super) node: Option<usize>,
    pub(super) format: Value,
    pub(super) labels: Value,
    pub(super) dtype: NumericDType,
    pub(super) is_dlarray: bool,
}

impl Payload {
    pub(super) fn parse(value: Value, function: &'static str) -> BuiltinResult<Self> {
        let is_dlarray =
            matches!(&value, Value::Object(object) if object.class_name().is(DLARRAY_CLASS));
        let data = dlarray_data(&value, function)?;
        if crate::builtins::common::validation::value_has_native_integer_class(&data) {
            return Err(deep_learning_error(
                function,
                format!("{function}: native integer dlarray operands require explicit double or single conversion"),
            ));
        }
        let node = dlarray_node_id(&value);
        let (format, labels) = dlarray_format_and_labels(&value);
        let tensor = host_tensor_from_value(&data, function, "operand")?;
        let dtype = tensor.numeric_dtype();
        if !matches!(dtype, NumericDType::F64 | NumericDType::F32) {
            return Err(deep_learning_error(
                function,
                format!("{function}: dlarray arithmetic requires double or single data"),
            ));
        }
        Ok(Self {
            tensor,
            node,
            format,
            labels,
            dtype,
            is_dlarray,
        })
    }
}

pub(super) fn floating_binary_dtype(lhs: NumericDType, rhs: NumericDType) -> NumericDType {
    match (lhs, rhs) {
        (NumericDType::F32, _) | (_, NumericDType::F32) => NumericDType::F32,
        (NumericDType::F64, NumericDType::F64) => NumericDType::F64,
        _ => unreachable!("Payload admission limits arithmetic to floating classes"),
    }
}

pub(super) fn result_metadata_owner<'a>(lhs: &'a Payload, rhs: &'a Payload) -> &'a Payload {
    if lhs.is_dlarray || !rhs.is_dlarray {
        lhs
    } else {
        rhs
    }
}

pub(super) fn floating_tensor_values(
    tensor: &Tensor,
    function: &'static str,
) -> BuiltinResult<Vec<f64>> {
    match tensor
        .clone()
        .into_numeric_storage()
        .map_err(|err| deep_learning_error(function, err))?
    {
        NumericStorage::F64(values) => Ok(values),
        NumericStorage::F32(values) => Ok(values.into_iter().map(f64::from).collect::<Vec<_>>()),
        storage => Err(deep_learning_error(
            function,
            format!(
                "{function}: automatic differentiation requires double or single tensors, got {}",
                storage.class_name()
            ),
        )),
    }
}

pub(super) fn floating_tensor_from_f64(
    data: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    match dtype {
        NumericDType::F64 | NumericDType::F32 => Tensor::new_with_dtype(data, shape, dtype)
            .map_err(|err| deep_learning_error(function, err)),
        dtype => Err(deep_learning_error(
            function,
            format!(
                "{function}: automatic differentiation cannot construct {} gradients",
                dtype.class_name()
            ),
        )),
    }
}

pub(super) fn node_dtype(
    nodes: &[Node],
    id: Option<usize>,
    fallback: NumericDType,
) -> NumericDType {
    id.and_then(|id| nodes.get(id))
        .and_then(|node| node.value.numeric_dtype())
        .unwrap_or(fallback)
}

pub(super) fn wrap_result(
    payload: &Payload,
    tensor: Tensor,
    node: Option<usize>,
) -> BuiltinResult<Value> {
    let Value::Object(mut out) = super::super::object_with_identity(
        DLARRAY_CLASS,
        [
            ("Data", Value::Tensor(tensor)),
            ("Format", payload.format.clone()),
            ("Labels", payload.labels.clone()),
        ],
    )?
    else {
        unreachable!("dlarray object constructor")
    };
    if let Some(id) = node {
        attach_node_to_object(&mut out, id)?;
    }
    Ok(Value::Object(out))
}
