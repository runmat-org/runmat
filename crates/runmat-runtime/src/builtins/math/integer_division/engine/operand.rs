use super::*;

pub(super) struct IdivideBuffer {
    pub(super) data: Vec<i128>,
    pub(super) shape: Vec<usize>,
    pub(super) class: Option<IntegerClass>,
}

pub(super) enum AdmittedOperand {
    Host(IdivideBuffer),
    Resident {
        handle: runmat_accelerate_api::GpuTensorHandle,
        class: IntegerClass,
    },
}

impl AdmittedOperand {
    pub(super) fn new(value: Value) -> BuiltinResult<Self> {
        if let Value::GpuTensor(handle) = value {
            let element = gpu_helpers::expected_handle_numeric_element_type(&handle)
                .map_err(|detail| error(&ERROR_INVALID_INPUT, detail))?;
            let class = runmat_accelerate_api::handle_integer_class(&handle).ok_or_else(|| {
                error(
                    &ERROR_INVALID_INPUT,
                    "resident inputs must use an integer class",
                )
            })?;
            if !gpu_helpers::numeric_descriptor_matches(
                &handle,
                element,
                runmat_accelerate_api::GpuTensorStorage::Real,
            ) {
                return Err(error(
                    &ERROR_INVALID_INPUT,
                    "resident inputs require explicit real storage",
                ));
            }
            Ok(Self::Resident { handle, class })
        } else {
            IdivideBuffer::from_host(value).map(Self::Host)
        }
    }

    pub(super) fn class(&self) -> Option<IntegerClass> {
        match self {
            Self::Host(buffer) => buffer.class,
            Self::Resident { class, .. } => Some(*class),
        }
    }

    pub(super) fn shape(&self) -> &[usize] {
        match self {
            Self::Host(buffer) => &buffer.shape,
            Self::Resident { handle, .. } => handle.shape(),
        }
    }

    pub(super) async fn materialize(self) -> BuiltinResult<IdivideBuffer> {
        match self {
            Self::Host(buffer) => Ok(buffer),
            Self::Resident { handle, .. } => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|detail| error(&ERROR_INVALID_INPUT, detail))?;
                IdivideBuffer::from_tensor(tensor)
            }
        }
    }
}

impl IdivideBuffer {
    fn from_host(value: Value) -> BuiltinResult<Self> {
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
            other => Err(error(
                &ERROR_INVALID_INPUT,
                format!("unsupported integer input {other:?}"),
            )),
        }
    }

    fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        let shape = tensor.shape().to_vec();
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
    left: &AdmittedOperand,
    right: &AdmittedOperand,
) -> BuiltinResult<IntegerClass> {
    match (left.class(), right.class()) {
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
