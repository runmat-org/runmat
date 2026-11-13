use super::*;
use runmat_value::LogicalArray;
pub(super) fn scalar(value: &Value, parsed: &ParsedArguments) -> Option<Value> {
    let reduced = ResolvedDims::resolve(&[1, 1], &parsed.selection).len() != 0;
    match value {
        Value::Num(number) => Some(Value::Num(
            if reduced && number.is_nan() && parsed.nan_mode == ReductionNaN::Omit {
                1.
            } else {
                *number
            },
        )),
        Value::Int(integer) if parsed.output == OutputTemplate::Native => {
            Some(Value::Int(integer.clone()))
        }
        Value::Int(integer) => Some(Value::Num(integer.to_f64())),
        Value::Bool(value) if parsed.output == OutputTemplate::Native => Some(Value::Bool(*value)),
        Value::Bool(value) => Some(Value::Num(if *value { 1. } else { 0. })),
        _ => None,
    }
}
pub(super) fn execute(value: Value, parsed: &ParsedArguments) -> BuiltinResult<Value> {
    if parsed.output == OutputTemplate::Native {
        match &value {
            Value::Int(value) => return Ok(Value::Int(value.clone())),
            Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
                let axes = ResolvedDims::resolve(tensor.shape(), &parsed.selection);
                let dims = axes.iter().collect::<Vec<_>>();
                return super::super::integer_native::product(
                    tensor.integer_storage().expect("integer input"),
                    tensor.shape(),
                    &dims,
                )
                .map_err(prod_internal_error);
            }
            _ => {}
        }
    }
    let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        && parsed.output == OutputTemplate::Native;
    let input = tensor::value_into_tensor_for(NAME, value)?;
    let axes = ResolvedDims::resolve(input.shape(), &parsed.selection);
    let output = product(&input, &axes, parsed.nan_mode, parsed.output)?;
    if logical {
        if output.len() == 1 {
            return Ok(Value::Bool(tensor::tensor_value_f64(&output, 0) != 0.));
        }
        let shape = output.shape().to_vec();
        let data = (0..output.len())
            .map(|index| u8::from(tensor::tensor_value_f64(&output, index) != 0.))
            .collect();
        return LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(prod_internal_error);
    }
    Ok(tensor::tensor_into_value(output))
}
pub(super) fn product(
    tensor: &Tensor,
    axes: &ResolvedDims,
    nan_mode: ReductionNaN,
    output: OutputTemplate,
) -> BuiltinResult<Tensor> {
    if axes.len() == 0 {
        let dtype =
            if tensor.numeric_dtype() == NumericDType::F32 && output != OutputTemplate::Double {
                NumericDType::F32
            } else {
                NumericDType::F64
            };
        return Ok(tensor::coerce_tensor_dtype(tensor.clone(), dtype));
    }
    let shape = if tensor.shape().is_empty() {
        vec![1, 1]
    } else {
        tensor.shape().to_vec()
    };
    let mut out_shape = shape.clone();
    for axis in axes.iter() {
        out_shape[axis] = 1;
    }
    let count = tensor::element_count(&out_shape);
    let mut coordinates = vec![0usize; shape.len()];
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in &out_shape {
        strides.push(stride);
        stride = stride.saturating_mul(extent);
    }
    fn reduce<T: num_traits::Float>(
        values: impl Iterator<Item = T>,
        shape: &[usize],
        axes: &ResolvedDims,
        count: usize,
        coordinates: &mut [usize],
        strides: &[usize],
        nan_mode: ReductionNaN,
    ) -> Vec<T> {
        let mut products = vec![T::one(); count];
        let mut saw_nan = vec![false; count];
        for (linear, value) in values.enumerate() {
            let mut remainder = linear;
            for (axis, &extent) in shape.iter().enumerate() {
                coordinates[axis] = if extent == 0 { 0 } else { remainder % extent };
                if extent != 0 {
                    remainder /= extent;
                }
            }
            let destination = coordinates
                .iter()
                .enumerate()
                .filter(|(axis, _)| !axes.contains(*axis))
                .map(|(axis, coordinate)| coordinate * strides[axis])
                .sum::<usize>();
            if value.is_nan() {
                if nan_mode == ReductionNaN::Include {
                    saw_nan[destination] = true;
                }
            } else {
                products[destination] = products[destination] * value;
            }
        }
        for (value, nan) in products.iter_mut().zip(saw_nan) {
            if nan {
                *value = T::nan();
            }
        }
        products
    }
    if tensor.numeric_dtype() == NumericDType::F32 && output != OutputTemplate::Double {
        let values = tensor
            .as_f32_slice()
            .expect("native single storage")
            .iter()
            .copied();
        Tensor::from_f32(
            reduce(
                values,
                &shape,
                axes,
                count,
                &mut coordinates,
                &strides,
                nan_mode,
            ),
            out_shape,
        )
        .map_err(prod_internal_error)
    } else {
        let values = (0..tensor.len()).map(|i| tensor::tensor_value_f64(tensor, i));
        Tensor::new(
            reduce(
                values,
                &shape,
                axes,
                count,
                &mut coordinates,
                &strides,
                nan_mode,
            ),
            out_shape,
        )
        .map_err(prod_internal_error)
    }
}
