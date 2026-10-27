use runmat_runtime::RuntimeError;
use runmat_value::Value;

#[derive(Clone, Copy)]
pub enum AutoBinaryOp {
    Elementwise,
    MatMul,
}

#[derive(Clone, Copy)]
pub enum AutoUnaryOp {
    Transpose,
}

#[cfg(feature = "native-accel")]
pub async fn accel_promote_binary(
    runtime: &runmat_runtime::context::RuntimeContext,
    op: AutoBinaryOp,
    a: &Value,
    b: &Value,
) -> Result<(Value, Value), RuntimeError> {
    use runmat_accelerate::{promote_builtin_binary_in_context, BinaryOp};
    let mapped = match op {
        AutoBinaryOp::Elementwise => BinaryOp::Elementwise,
        AutoBinaryOp::MatMul => BinaryOp::MatMul,
    };
    promote_builtin_binary_in_context(runtime, mapped, a, b)
        .await
        .map_err(promotion_error)
}

#[cfg(not(feature = "native-accel"))]
pub async fn accel_promote_binary(
    _runtime: &runmat_runtime::context::RuntimeContext,
    _op: AutoBinaryOp,
    a: &Value,
    b: &Value,
) -> Result<(Value, Value), RuntimeError> {
    Ok((a.clone(), b.clone()))
}

#[cfg(feature = "native-accel")]
pub async fn accel_promote_unary(
    runtime: &runmat_runtime::context::RuntimeContext,
    op: AutoUnaryOp,
    value: &Value,
) -> Result<Value, RuntimeError> {
    use runmat_accelerate::{promote_builtin_unary_in_context, UnaryOp};
    let mapped = match op {
        AutoUnaryOp::Transpose => UnaryOp::Transpose,
    };
    promote_builtin_unary_in_context(runtime, mapped, value)
        .await
        .map_err(promotion_error)
}

#[cfg(not(feature = "native-accel"))]
pub async fn accel_promote_unary(
    _runtime: &runmat_runtime::context::RuntimeContext,
    _op: AutoUnaryOp,
    value: &Value,
) -> Result<Value, RuntimeError> {
    Ok(value.clone())
}

#[cfg(feature = "native-accel")]
fn promotion_error(error: runmat_accelerate_api::ProviderExecutionError) -> RuntimeError {
    runmat_runtime::build_runtime_error(error.to_string())
        .with_source(error)
        .build()
}
