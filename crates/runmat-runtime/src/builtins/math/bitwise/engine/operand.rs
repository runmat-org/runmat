use super::*;

pub(super) struct BitBuffer {
    pub(super) data: Vec<u64>,
    pub(super) shape: Vec<usize>,
    /// `None` represents MATLAB double/logical bit operands, which use the
    /// documented unsigned-64 interpretation and return a double result.
    pub(super) compute_class: Option<IntegerClass>,
    pub(super) output_class: Option<IntegerClass>,
    pub(super) is_scalar: bool,
}

pub(super) struct ShiftBuffer {
    pub(super) data: Vec<i128>,
    pub(super) shape: Vec<usize>,
}

pub(super) struct BitValueBuffer {
    pub(super) data: Vec<bool>,
    pub(super) shape: Vec<usize>,
}

pub(super) fn scalar_or_exact_size_plan(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Result<BroadcastPlan, String> {
    let scalar = |shape: &[usize]| {
        shape
            .iter()
            .try_fold(1usize, |length, dimension| length.checked_mul(*dimension))
            == Some(1)
    };
    let same_size = || {
        let rank = lhs_shape.len().max(rhs_shape.len()).max(2);
        let lhs = crate::builtins::common::broadcast::align_shape(lhs_shape, rank);
        let rhs = crate::builtins::common::broadcast::align_shape(rhs_shape, rank);
        lhs == rhs
    };
    if !scalar(lhs_shape) && !scalar(rhs_shape) && !same_size() {
        return Err("inputs must be scalar or have exactly the same size".to_string());
    }
    let rank = lhs_shape.len().max(rhs_shape.len());
    let lhs_canonical = crate::builtins::common::broadcast::align_shape(lhs_shape, rank);
    let rhs_canonical = crate::builtins::common::broadcast::align_shape(rhs_shape, rank);
    BroadcastPlan::new(&lhs_canonical, &rhs_canonical)
}

pub(super) async fn bit_buffer_from(
    name: &'static str,
    value: Value,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<BitBuffer> {
    match value {
        Value::Num(value) => Ok(BitBuffer {
            data: vec![double_to_bits(name, value, assumed)?],
            shape: vec![1, 1],
            compute_class: assumed,
            output_class: None,
            is_scalar: true,
        }),
        Value::Bool(value) => Ok(BitBuffer {
            data: vec![double_to_bits(name, f64::from(value), assumed)?],
            shape: vec![1, 1],
            compute_class: assumed,
            output_class: None,
            is_scalar: true,
        }),
        Value::Int(value) => Ok(BitBuffer {
            data: vec![int_to_bits(&value)],
            shape: vec![1, 1],
            compute_class: require_assumed_class(name, value.integer_class(), assumed)?,
            output_class: Some(value.integer_class()),
            is_scalar: true,
        }),
        Value::Tensor(tensor) => tensor_to_bit_buffer(name, tensor, assumed),
        Value::LogicalArray(array) => Ok(BitBuffer {
            data: array.data.into_iter().map(|v| u64::from(v != 0)).collect(),
            shape: array.shape,
            compute_class: assumed,
            output_class: None,
            is_scalar: false,
        }),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))?;
            tensor_to_bit_buffer(name, tensor, assumed)
        }
        other => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("{name}: unsupported input {other:?}"),
        )),
    }
}

pub(super) async fn shift_buffer_from(value: Value) -> BuiltinResult<ShiftBuffer> {
    match value {
        Value::Bool(value) => Ok(ShiftBuffer {
            data: vec![i128::from(value)],
            shape: vec![1, 1],
        }),
        Value::Num(value) => Ok(ShiftBuffer {
            data: vec![double_to_shift(value)?],
            shape: vec![1, 1],
        }),
        Value::Int(value) => Ok(ShiftBuffer {
            data: vec![value.to_i128()],
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => tensor_to_shift_buffer(tensor),
        Value::LogicalArray(array) => Ok(ShiftBuffer {
            data: array.data.into_iter().map(|v| i128::from(v != 0)).collect(),
            shape: array.shape,
        }),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| error_with_detail(BITSHIFT_NAME, &ERROR_INVALID_INPUT, err))?;
            tensor_to_shift_buffer(tensor)
        }
        other => Err(error_with_detail(
            BITSHIFT_NAME,
            &ERROR_INVALID_INPUT,
            format!("bitshift: unsupported shift input {other:?}"),
        )),
    }
}

pub(super) async fn bit_value_buffer_from(value: Value) -> BuiltinResult<BitValueBuffer> {
    match value {
        Value::Bool(value) => Ok(BitValueBuffer {
            data: vec![value],
            shape: vec![1, 1],
        }),
        Value::Num(value) => Ok(BitValueBuffer {
            data: vec![finite_nonzero_bit_value(value)?],
            shape: vec![1, 1],
        }),
        Value::Int(value) => Ok(BitValueBuffer {
            data: vec![value.to_i128() != 0],
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => tensor_to_bit_value_buffer(tensor),
        Value::LogicalArray(array) => Ok(BitValueBuffer {
            data: array.data.into_iter().map(|value| value != 0).collect(),
            shape: array.shape,
        }),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_INVALID_INPUT, err))?;
            tensor_to_bit_value_buffer(tensor)
        }
        other => Err(error_with_detail(
            BITSET_NAME,
            &ERROR_INVALID_INPUT,
            format!("bitset: unsupported bit value {other:?}"),
        )),
    }
}

pub(super) fn tensor_to_bit_value_buffer(tensor: Tensor) -> BuiltinResult<BitValueBuffer> {
    let shape = tensor.shape.clone();
    let data = match tensor.integer_storage() {
        Some(storage) => storage
            .exact_values()
            .iter()
            .map(|value| value.to_i128() != 0)
            .collect(),
        None => tensor::tensor_into_values_f64(tensor)
            .into_iter()
            .map(finite_nonzero_bit_value)
            .collect::<BuiltinResult<Vec<_>>>()?,
    };
    Ok(BitValueBuffer { data, shape })
}

pub(super) fn finite_nonzero_bit_value(value: f64) -> BuiltinResult<bool> {
    if value.is_finite() {
        Ok(value != 0.0)
    } else {
        Err(error_with_detail(
            BITSET_NAME,
            &ERROR_INVALID_INPUT,
            "bit values must be finite numeric or logical values",
        ))
    }
}

pub(super) fn tensor_to_shift_buffer(tensor: Tensor) -> BuiltinResult<ShiftBuffer> {
    let shape = tensor.shape.clone();
    let data = match tensor.integer_storage() {
        Some(storage) => storage
            .exact_values()
            .iter()
            .map(IntValue::to_i128)
            .collect(),
        None => tensor::tensor_into_values_f64(tensor)
            .into_iter()
            .map(double_to_shift)
            .collect::<BuiltinResult<Vec<_>>>()?,
    };
    Ok(ShiftBuffer { data, shape })
}

pub(super) fn tensor_to_bit_buffer(
    name: &'static str,
    tensor: Tensor,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<BitBuffer> {
    let is_scalar = tensor::element_count(&tensor.shape) == 1;
    let shape = tensor.shape.clone();
    let (data, native_class, output_class) = match tensor.integer_storage() {
        Some(storage) => (
            storage.exact_values().iter().map(int_to_bits).collect(),
            Some(storage.integer_class()),
            Some(storage.integer_class()),
        ),
        None => {
            if !matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ) {
                return Err(error_with_detail(
                    name,
                    &ERROR_INVALID_INPUT,
                    "integer tensor is missing authoritative native storage",
                ));
            }
            let data = tensor::tensor_into_values_f64(tensor)
                .into_iter()
                .map(|value| double_to_bits(name, value, assumed))
                .collect::<BuiltinResult<Vec<_>>>()?;
            (data, None, None)
        }
    };
    let compute_class = match native_class {
        Some(class) => require_assumed_class(name, class, assumed)?,
        None => assumed,
    };
    Ok(BitBuffer {
        data,
        shape,
        compute_class,
        output_class,
        is_scalar,
    })
}
