use super::*;

pub(super) async fn scale_product(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if is_decomposition_value(&lhs) {
        let mut spec = object_to_spec(&lhs)?;
        ensure_nonfloating_object_enabled(&spec)?;
        let scalar = decomposition_scale_scalar_async(&rhs, "scale factor").await?;
        spec.scale = complex_mul(spec.scale, scalar);
        return Ok(Value::Object(spec_to_object(&spec)?));
    }
    if is_decomposition_value(&rhs) {
        let mut spec = object_to_spec(&rhs)?;
        ensure_nonfloating_object_enabled(&spec)?;
        let scalar = decomposition_scale_scalar_async(&lhs, "scale factor").await?;
        spec.scale = complex_mul(scalar, spec.scale);
        return Ok(Value::Object(spec_to_object(&spec)?));
    }
    Err(invalid(
        "decomposition: multiplication requires one decomposition object and one scalar",
    ))
}

pub(super) async fn divide_or_solve(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if is_decomposition_value(&lhs) {
        let mut spec = object_to_spec(&lhs)?;
        ensure_nonfloating_object_enabled(&spec)?;
        let scalar = decomposition_scale_scalar_async(&rhs, "scale divisor").await?;
        spec.scale = complex_div(spec.scale, scalar)?;
        return Ok(Value::Object(spec_to_object(&spec)?));
    }
    if is_decomposition_value(&rhs) {
        let spec = object_to_spec(&rhs)?;
        ensure_nonfloating_object_enabled(&spec)?;
        ensure_gpu_operand_enabled(&lhs)?;
        let matrix = effective_matrix(&spec)?;
        let lhs = if is_nonfloating_coefficient_matrix(&spec.matrix) {
            floating_solve_operand(&lhs).await?
        } else {
            lhs
        };
        return shared_solve::evaluate(SolveOrientation::Right, &lhs, &matrix).await;
    }
    Err(invalid(
        "decomposition: right division requires a decomposition object",
    ))
}

pub(super) async fn left_solve(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if !is_decomposition_value(&lhs) {
        return Err(invalid(
            "decomposition: left division requires a decomposition object on the left",
        ));
    }
    let spec = object_to_spec(&lhs)?;
    ensure_nonfloating_object_enabled(&spec)?;
    ensure_gpu_operand_enabled(&rhs)?;
    let matrix = effective_matrix(&spec)?;
    let rhs = if is_nonfloating_coefficient_matrix(&spec.matrix) {
        floating_solve_operand(&rhs).await?
    } else {
        rhs
    };
    shared_solve::evaluate(SolveOrientation::Left, &matrix, &rhs).await
}

pub(super) fn is_decomposition_value(value: &Value) -> bool {
    matches!(value, Value::Object(object) if object.is_class(CLASS_IDENTITY))
}

pub(super) fn effective_matrix(spec: &DecompositionSpec) -> BuiltinResult<Value> {
    let matrix = oriented_matrix(spec)?;
    scale_matrix(promote_coefficient_to_floating(matrix)?, spec.scale)
}

pub(super) fn ensure_nonfloating_object_enabled(spec: &DecompositionSpec) -> BuiltinResult<()> {
    if is_nonfloating_coefficient_matrix(&spec.matrix) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECOMPOSITION_NONFLOATING_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

pub(super) fn ensure_gpu_operand_enabled(value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECOMPOSITION_GPU_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

pub(super) fn ensure_scale_operand_extensions_enabled(value: &Value) -> BuiltinResult<()> {
    ensure_gpu_operand_enabled(value)?;
    if crate::builtins::common::validation::is_typed_complex_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECOMPOSITION_COMPLEX_INTEGER_SCALE_FACTOR_EXTENSION,
            NAME,
        )?;
    } else if is_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECOMPOSITION_INTEGER_SCALE_FACTOR_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

pub(super) async fn floating_solve_operand(value: &Value) -> BuiltinResult<Value> {
    if !is_nonfloating_coefficient_matrix(value) {
        return Ok(value.clone());
    }
    let host = crate::dispatcher::gather_if_needed_async(value)
        .await
        .map_err(|err| {
            build_runtime_error(err.message())
                .with_builtin(NAME)
                .with_source(err)
                .build()
        })?;
    promote_coefficient_to_floating(host)
}

pub(super) fn promote_coefficient_to_floating(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let shape = tensor.shape.clone();
            Ok(Value::Tensor(
                Tensor::new(tensor::tensor_into_values_f64(tensor), shape).map_err(internal)?,
            ))
        }
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
            let shape = tensor.shape.clone();
            Ok(Value::ComplexTensor(
                ComplexTensor::new(tensor.materialize_f64(), shape).map_err(internal)?,
            ))
        }
        Value::LogicalArray(array) => Ok(Value::Tensor(
            Tensor::new(
                array.data.iter().map(|&value| f64::from(value)).collect(),
                array.shape,
            )
            .map_err(internal)?,
        )),
        Value::Int(value) => Ok(Value::Num(value.to_f64())),
        Value::Bool(value) => Ok(Value::Num(if value { 1.0 } else { 0.0 })),
        other => Ok(other),
    }
}
