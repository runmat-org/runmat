use super::geometry::*;
use super::selection::{update_best_complex, update_best_real, BestComplex, BestReal};
use super::*;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor;
use runmat_value::{ComplexTensor, NumericDType};

pub(super) fn reduction_max_host(
    value: Value,
    args: &ReductionArgs,
) -> BuiltinResult<MaxEvaluation> {
    if let Value::Int(value) = &value {
        let storage = crate::builtins::math::reduction::integer_native::storage_from_scalar(value);
        return reduce_integer_max(&storage, vec![1, 1], args);
    }
    if let Some((storage, shape)) = native_integer_input(&value) {
        return reduce_integer_max(storage, shape, args);
    }
    match materialize_for_max("max", value)? {
        InputData::Real(tensor) => reduce_real_tensor(tensor, args),
        InputData::Complex(tensor) => reduce_complex_tensor(tensor, args),
    }
}

fn native_integer_input(value: &Value) -> Option<(&runmat_value::IntegerStorage, Vec<usize>)> {
    match value {
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| (storage, tensor.shape().to_vec())),
        _ => None,
    }
}

fn reduce_integer_max(
    storage: &runmat_value::IntegerStorage,
    shape: Vec<usize>,
    args: &ReductionArgs,
) -> BuiltinResult<MaxEvaluation> {
    if storage.is_empty() {
        // Integer extrema preserve the empty input shape for every reduction
        // selector. A non-empty reduced shape cannot represent an empty typed
        // payload, and MATLAB extrema of an empty array stay empty.
        let output_shape = shape;
        let values = crate::builtins::math::reduction::integer_native::empty_like(
            storage,
            output_shape.clone(),
        )
        .map_err(|error| max_internal_error(format!("max: {error}")))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|error| max_internal_error(format!("max: {error}")))?;
        return Ok(MaxEvaluation {
            values,
            indices: tensor::tensor_into_value(indices),
        });
    }

    let resolved = resolve_reduction_dims(&shape, &args.selection)?;
    let comparison = match args.comparison {
        ComparisonMethod::Auto | ComparisonMethod::Real => {
            crate::builtins::math::reduction::integer_native::ExtremaComparison::Natural
        }
        ComparisonMethod::Abs => {
            crate::builtins::math::reduction::integer_native::ExtremaComparison::Absolute
        }
    };
    let extrema = crate::builtins::math::reduction::integer_native::extrema(
        storage,
        &shape,
        resolved.output_shape,
        &resolved.reduced_dims,
        &resolved.dims_mask,
        &resolved.reduce_strides,
        resolved.reduce_all,
        args.linear_index,
        crate::builtins::math::reduction::integer_native::ExtremaDirection::Max,
        comparison,
    )
    .map_err(|error| max_internal_error(format!("max: {error}")))?;
    Ok(MaxEvaluation {
        values: extrema.values,
        indices: extrema.indices,
    })
}

pub(in crate::builtins::math::reduction::max) enum InputData {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn real_tensor_from_f64(
    values: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
) -> Result<Tensor, String> {
    match dtype {
        NumericDType::F64 => Tensor::new(values, shape),
        NumericDType::F32 => Tensor::from_f32(
            values.into_iter().map(|value| value as f32).collect(),
            shape,
        ),
        dtype => Err(format!(
            "max: unexpected {} storage in floating reduction",
            dtype.class_name()
        )),
    }
}

pub(in crate::builtins::math::reduction::max) fn materialize_for_max(
    name: &str,
    value: Value,
) -> BuiltinResult<InputData> {
    match value {
        Value::Tensor(t) => Ok(InputData::Real(t)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(max_invalid_input)?;
            Ok(InputData::Real(tensor))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| max_internal_error(format!("{name}: {e}")))?;
            Ok(InputData::Real(tensor))
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| max_internal_error(format!("{name}: {e}")))?;
            Ok(InputData::Real(tensor))
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| max_internal_error(format!("{name}: {e}")))?;
            Ok(InputData::Real(tensor))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| max_internal_error(format!("{name}: {e}")))?;
            Ok(InputData::Complex(tensor))
        }
        Value::ComplexTensor(ct) => Ok(InputData::Complex(ct)),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::SparseTensor(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::Cell(_) => Err(max_invalid_input(format!(
            "{name}: expected numeric or logical input, received non-numeric value"
        ))),
        Value::GpuTensor(_) => Err(max_internal_error(format!(
            "{name}: internal error – GPU tensors must be gathered before host execution"
        ))),
        Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::Listener(_) => Err(max_invalid_input(format!("{name}: unsupported input type"))),
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
        | Value::OutputList(_) => Err(max_invalid_input(format!("{name}: unsupported input type"))),
    }
}

fn reduce_real_tensor(tensor: Tensor, args: &ReductionArgs) -> BuiltinResult<MaxEvaluation> {
    let shape = tensor.shape().to_vec();
    let dtype = tensor.numeric_dtype();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| max_internal_error(format!("max: {e}")))?;
    let data = storage.materialize_f64();
    if data.is_empty() {
        let output_shape = resolve_output_shape(&shape, &args.selection, &[])?;
        let values = real_tensor_from_f64(Vec::new(), output_shape.clone(), dtype)
            .map_err(|e| max_internal_error(format!("max: {e}")))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|e| max_internal_error(format!("max: {e}")))?;
        return Ok(MaxEvaluation {
            values: tensor::tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let resolved = resolve_reduction_dims(&shape, &args.selection)?;
    let output_shape = resolved.output_shape.clone();
    let output_len = tensor::element_count(&output_shape);

    if output_len == 0 {
        let values = real_tensor_from_f64(Vec::new(), output_shape.clone(), dtype)
            .map_err(|e| max_internal_error(format!("max: {e}")))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|e| max_internal_error(format!("max: {e}")))?;
        return Ok(MaxEvaluation {
            values: tensor::tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let strides = compute_strides(&shape);
    let output_strides = compute_strides(&output_shape);
    let dims_mask = resolved.dims_mask.clone();
    let reduce_strides = resolved.reduce_strides.clone();

    let mut best = vec![BestReal::new(); output_len];
    let mut coords = vec![0usize; shape.len()];
    for &value in &data {
        let out_idx = map_output_index(&coords, &output_strides, &dims_mask);
        let reduce_idx = map_reduce_index(
            &coords,
            &resolved.reduced_dims,
            &reduce_strides,
            resolved.reduce_all,
        );
        let full_idx = map_linear_index(&coords, &strides);

        update_best_real(
            &mut best[out_idx],
            value,
            reduce_idx,
            full_idx,
            args.nan_mode,
            args.comparison,
        );
        increment_coords(&mut coords, &shape);
    }

    let mut values = vec![0.0f64; output_len];
    let mut indices = vec![0.0f64; output_len];

    for (i, entry) in best.iter().enumerate() {
        if entry.nan_fixed {
            values[i] = f64::NAN;
            indices[i] = if args.linear_index || resolved.reduce_all {
                (entry.full_index + 1) as f64
            } else if resolved.reduced_dims.is_empty() {
                1.0
            } else {
                (entry.reduce_index + 1) as f64
            };
            continue;
        }
        if !entry.has_value {
            values[i] = f64::NAN;
            indices[i] = f64::NAN;
            continue;
        }
        values[i] = entry.value;
        indices[i] = if args.linear_index || resolved.reduce_all {
            (entry.full_index + 1) as f64
        } else if resolved.reduced_dims.is_empty() {
            1.0
        } else {
            (entry.reduce_index + 1) as f64
        };
    }

    let value_tensor = real_tensor_from_f64(values, output_shape.clone(), dtype)
        .map_err(|e| max_internal_error(format!("max: {e}")))?;
    let index_tensor =
        Tensor::new(indices, output_shape).map_err(|e| max_internal_error(format!("max: {e}")))?;

    Ok(MaxEvaluation {
        values: tensor::tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn reduce_complex_tensor(
    tensor: ComplexTensor,
    args: &ReductionArgs,
) -> BuiltinResult<MaxEvaluation> {
    let shape = tensor.shape().to_vec();
    let dtype = tensor.numeric_dtype();
    let data = tensor.materialize_f64();
    if data.is_empty() {
        let output_shape = resolve_output_shape(&shape, &args.selection, &[])?;
        let values =
            ComplexTensor::from_f64_values_with_dtype(Vec::new(), output_shape.clone(), dtype)
                .map_err(|e| max_internal_error(format!("max: {e}")))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|e| max_internal_error(format!("max: {e}")))?;
        return Ok(MaxEvaluation {
            values: complex_tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let resolved = resolve_reduction_dims(&shape, &args.selection)?;
    let output_shape = resolved.output_shape.clone();
    let output_len = tensor::element_count(&output_shape);

    if output_len == 0 {
        let values =
            ComplexTensor::from_f64_values_with_dtype(Vec::new(), output_shape.clone(), dtype)
                .map_err(|e| max_internal_error(format!("max: {e}")))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|e| max_internal_error(format!("max: {e}")))?;
        return Ok(MaxEvaluation {
            values: complex_tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let strides = compute_strides(&shape);
    let output_strides = compute_strides(&output_shape);
    let dims_mask = resolved.dims_mask.clone();
    let reduce_strides = resolved.reduce_strides.clone();

    let mut best = vec![BestComplex::new(); output_len];
    let mut coords = vec![0usize; shape.len()];

    for &(re, im) in &data {
        let out_idx = map_output_index(&coords, &output_strides, &dims_mask);
        let reduce_idx = map_reduce_index(
            &coords,
            &resolved.reduced_dims,
            &reduce_strides,
            resolved.reduce_all,
        );
        let full_idx = map_linear_index(&coords, &strides);
        update_best_complex(
            &mut best[out_idx],
            (re, im),
            reduce_idx,
            full_idx,
            args.nan_mode,
            args.comparison,
        );
        increment_coords(&mut coords, &shape);
    }

    let mut values = vec![(0.0f64, 0.0f64); output_len];
    let mut indices = vec![0.0f64; output_len];

    for (i, entry) in best.iter().enumerate() {
        if entry.nan_fixed {
            values[i] = (f64::NAN, f64::NAN);
            indices[i] = if args.linear_index || resolved.reduce_all {
                (entry.full_index + 1) as f64
            } else if resolved.reduced_dims.is_empty() {
                1.0
            } else {
                (entry.reduce_index + 1) as f64
            };
            continue;
        }
        if !entry.has_value {
            values[i] = (f64::NAN, f64::NAN);
            indices[i] = f64::NAN;
            continue;
        }
        values[i] = entry.value;
        indices[i] = if args.linear_index || resolved.reduce_all {
            (entry.full_index + 1) as f64
        } else if resolved.reduced_dims.is_empty() {
            1.0
        } else {
            (entry.reduce_index + 1) as f64
        };
    }

    let value_tensor =
        ComplexTensor::from_f64_values_with_dtype(values, output_shape.clone(), dtype)
            .map_err(|e| max_internal_error(format!("max: {e}")))?;
    let index_tensor =
        Tensor::new(indices, output_shape).map_err(|e| max_internal_error(format!("max: {e}")))?;
    Ok(MaxEvaluation {
        values: complex_tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}
