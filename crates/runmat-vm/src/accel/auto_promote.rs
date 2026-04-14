use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

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
    op: AutoBinaryOp,
    a: &Value,
    b: &Value,
) -> Result<(Value, Value), RuntimeError> {
    use runmat_accelerate::{promote_binary, BinaryOp};
    let mapped = match op {
        AutoBinaryOp::Elementwise => BinaryOp::Elementwise,
        AutoBinaryOp::MatMul => BinaryOp::MatMul,
    };
    Ok(promote_binary(mapped, a, b)
        .await
        .map_err(|e| e.to_string())?)
}

#[cfg(not(feature = "native-accel"))]
pub async fn accel_promote_binary(
    _op: AutoBinaryOp,
    a: &Value,
    b: &Value,
) -> Result<(Value, Value), RuntimeError> {
    Ok((a.clone(), b.clone()))
}

#[cfg(feature = "native-accel")]
pub async fn accel_promote_unary(op: AutoUnaryOp, value: &Value) -> Result<Value, RuntimeError> {
    use runmat_accelerate::{promote_unary, UnaryOp};
    let mapped = match op {
        AutoUnaryOp::Transpose => UnaryOp::Transpose,
    };
    Ok(promote_unary(mapped, value)
        .await
        .map_err(|e| e.to_string())?)
}

#[cfg(not(feature = "native-accel"))]
pub async fn accel_promote_unary(
    _op: AutoUnaryOp,
    value: &Value,
) -> Result<Value, RuntimeError> {
    Ok(value.clone())
}
