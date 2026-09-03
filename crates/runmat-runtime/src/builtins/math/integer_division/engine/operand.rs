use super::*;

pub(super) struct IdivideBuffer {
    pub(super) data: Vec<i128>,
    pub(super) shape: Vec<usize>,
    pub(super) class: Option<IntegerClass>,
}

impl IdivideBuffer {
    pub(super) async fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(value) => Ok(Self {
                data: vec![scalar_double_integer(value)?],
                shape: vec![1, 1],
                class: None,
            }),
            Value::Int(value) => Ok(Self {
                data: vec![value.to_i128()],
                shape: vec![1, 1],
                class: Some(value.integer_class()),
            }),
            Value::Tensor(tensor) => Self::from_tensor(tensor),
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|detail| error(&ERROR_INVALID_INPUT, detail))?;
                Self::from_tensor(tensor)
            }
            other => Err(error(
                &ERROR_INVALID_INPUT,
                format!("unsupported integer input {other:?}"),
            )),
        }
    }

    fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        let shape = tensor.shape.clone();
        if let Some(storage) = tensor.integer_storage() {
            return Ok(Self {
                data: storage
                    .exact_values()
                    .iter()
                    .map(|value| value.to_i128())
                    .collect(),
                shape,
                class: Some(storage.integer_class()),
            });
        }
        match tensor.numeric_dtype() {
            NumericDType::F32 | NumericDType::F64 => Err(error(
                &ERROR_INVALID_INPUT,
                "dense inputs must use an integer class",
            )),
            _ => Err(error(
                &ERROR_INVALID_INPUT,
                "integer tensor is missing authoritative native storage",
            )),
        }
    }
}

pub(super) fn output_class(
    left: &IdivideBuffer,
    right: &IdivideBuffer,
) -> BuiltinResult<IntegerClass> {
    match (left.class, right.class) {
        (Some(lhs), Some(rhs)) if lhs == rhs => Ok(lhs),
        (Some(class), None) | (None, Some(class)) => {
            if matches!(class, IntegerClass::Int64 | IntegerClass::UInt64) {
                return Err(error(
                    &ERROR_INVALID_INPUT,
                    "scalar double operands are not supported with int64 or uint64",
                ));
            }
            Ok(class)
        }
        (Some(_), Some(_)) => Err(error(
            &ERROR_INVALID_INPUT,
            "integer inputs must have matching classes unless one input is a scalar double",
        )),
        (None, None) => Err(error(
            &ERROR_INVALID_INPUT,
            "at least one input must be an integer class",
        )),
    }
}

fn scalar_double_integer(value: f64) -> BuiltinResult<i128> {
    if value.is_finite() && value.fract() == 0.0 {
        Ok(value as i128)
    } else {
        Err(error(
            &ERROR_INVALID_INPUT,
            "scalar double operands must be finite integer-valued values",
        ))
    }
}
