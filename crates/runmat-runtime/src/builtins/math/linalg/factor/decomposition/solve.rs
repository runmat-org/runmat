use super::*;
use crate::context::RuntimeContext;

pub(super) async fn scale_product(
    lhs: Value,
    rhs: Value,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    let (object, scalar) = if is_decomposition_value(&lhs) {
        (lhs, rhs)
    } else if is_decomposition_value(&rhs) {
        (rhs, lhs)
    } else {
        return Err(invalid(
            "decomposition: multiplication requires one decomposition object and one scalar",
        ));
    };
    let mut spec = object_to_spec(&object)?;
    ensure_nonfloating_object_enabled(&spec)?;
    let scalar = decomposition_scale_scalar_async(&scalar, "scale factor").await?;
    spec.scale = complex_mul(spec.scale, scalar);
    native::publish_value(&spec, runtime)
}
pub(super) async fn divide_or_solve(
    lhs: Value,
    rhs: Value,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    if is_decomposition_value(&lhs) {
        let mut spec = object_to_spec(&lhs)?;
        ensure_nonfloating_object_enabled(&spec)?;
        let scalar = decomposition_scale_scalar_async(&rhs, "scale divisor").await?;
        spec.scale = complex_div(spec.scale, scalar)?;
        return native::publish_value(&spec, runtime);
    }
    if !is_decomposition_value(&rhs) {
        return Err(invalid(
            "decomposition: right division requires a decomposition object",
        ));
    }
    let spec = object_to_spec(&rhs)?;
    ensure_nonfloating_object_enabled(&spec)?;
    retained_solve::evaluate(&spec, &lhs, true, runtime).await
}
pub(super) async fn left_solve(
    lhs: Value,
    rhs: Value,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    if !is_decomposition_value(&lhs) {
        return Err(invalid(
            "decomposition: left division requires a decomposition object on the left",
        ));
    }
    let spec = object_to_spec(&lhs)?;
    ensure_nonfloating_object_enabled(&spec)?;
    retained_solve::evaluate(&spec, &rhs, false, runtime).await
}
pub(super) fn is_decomposition_value(value: &Value) -> bool {
    matches!(value,Value::Object(object) if object.is_class(CLASS_IDENTITY))
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
