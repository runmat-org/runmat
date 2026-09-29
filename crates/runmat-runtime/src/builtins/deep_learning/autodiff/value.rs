use super::*;

pub(super) fn dlarray_format_and_labels(value: &Value) -> (Value, Value) {
    let Value::Object(object) = value else {
        return (Value::String(String::new()), Value::String(String::new()));
    };
    (
        object
            .properties
            .get("Format")
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
        object
            .properties
            .get("Labels")
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    )
}

pub(super) fn dlarray_data(value: &Value, function: &'static str) -> BuiltinResult<Value> {
    match value {
        Value::Object(object) if object.class_name == DLARRAY_CLASS => {
            object.properties.get("Data").cloned().ok_or_else(|| {
                deep_learning_error(function, format!("{function}: dlarray is missing Data"))
            })
        }
        other => Ok(other.clone()),
    }
}

pub(super) fn host_tensor_from_value(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Tensor> {
    match value {
        Value::Object(object) if object.class_name == DLARRAY_CLASS => {
            let data = object.properties.get("Data").ok_or_else(|| {
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
            matches!(&value, Value::Object(object) if object.class_name == DLARRAY_CLASS);
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

pub(super) fn floating_scalar_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        value => panic!(
            "automatic differentiation gradient unexpectedly used {} storage",
            value.class_name()
        ),
    }
}

pub(super) fn node_dtype(
    nodes: &[Node],
    id: Option<usize>,
    fallback: NumericDType,
) -> NumericDType {
    id.and_then(|id| nodes.get(id))
        .map(|node| node.value.numeric_dtype())
        .unwrap_or(fallback)
}

pub(super) fn wrap_result(payload: &Payload, tensor: Tensor, node: Option<usize>) -> Value {
    let mut out = ObjectInstance::new(DLARRAY_CLASS.to_string());
    out.properties
        .insert("Data".to_string(), Value::Tensor(tensor));
    out.properties
        .insert("Format".to_string(), payload.format.clone());
    out.properties
        .insert("Labels".to_string(), payload.labels.clone());
    if let Some(id) = node {
        out.properties.insert(
            AD_NODE_PROPERTY.to_string(),
            Value::Int(IntValue::U64(
                u64::try_from(id).expect("node index fits u64"),
            )),
        );
        let tape_id = active_tape()
            .expect("recorded output requires active tape")
            .borrow()
            .id;
        out.properties.insert(
            AD_TAPE_PROPERTY.to_string(),
            Value::Int(IntValue::U64(tape_id)),
        );
    }
    Value::Object(out)
}
