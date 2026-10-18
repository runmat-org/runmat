//! Consume a selected host unary result in its admitted physical owner.
//! Numerical functions and domain decisions remain with the original builtin.
use crate::context::preparation::PreparedHostNumericalOutput;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::BuiltinPreparationOperation;
use runmat_value::{NumericDType, Value};

pub(crate) fn real(
    runtime: &RuntimeContext,
    output: PreparedHostNumericalOutput,
    operation: BuiltinPreparationOperation,
    value: &Value,
    mut f32_kernel: impl FnMut(f32) -> BuiltinResult<f32>,
    mut f64_kernel: impl FnMut(f64) -> BuiltinResult<f64>,
) -> BuiltinResult<Value> {
    if output.operation() != operation {
        return Err(invalid());
    }
    let result = match value {
        Value::Num(number) if output.dtype() == NumericDType::F64 => {
            output.map_real_f64(runtime, std::slice::from_ref(number), &mut f64_kernel)?
        }
        Value::Tensor(tensor) if output.matches(tensor.numeric_dtype(), tensor.shape()) => {
            match tensor.numeric_dtype() {
                NumericDType::F32 => output.map_real_f32(
                    runtime,
                    tensor.as_f32_slice().ok_or_else(invalid)?,
                    &mut f32_kernel,
                )?,
                NumericDType::F64 => output.map_real_f64(
                    runtime,
                    tensor.as_f64_slice().ok_or_else(invalid)?,
                    &mut f64_kernel,
                )?,
                _ => return Err(invalid()),
            }
        }
        _ => return Err(invalid()),
    };
    Ok(scalarize(result))
}

pub(crate) fn promoting(
    runtime: &RuntimeContext,
    output: PreparedHostNumericalOutput,
    operation: BuiltinPreparationOperation,
    value: &Value,
    publish_complex: bool,
    mut f32_kernel: impl FnMut(f32) -> BuiltinResult<(f32, f32)>,
    mut f64_kernel: impl FnMut(f64) -> BuiltinResult<(f64, f64)>,
) -> BuiltinResult<Value> {
    if output.operation() != operation {
        return Err(invalid());
    }
    let result = match value {
        Value::Num(number) if output.dtype() == NumericDType::F64 => output.map_promoting_f64(
            runtime,
            std::slice::from_ref(number),
            publish_complex,
            &mut f64_kernel,
        )?,
        Value::Tensor(tensor) if output.matches(tensor.numeric_dtype(), tensor.shape()) => {
            match tensor.numeric_dtype() {
                NumericDType::F32 => output.map_promoting_f32(
                    runtime,
                    tensor.as_f32_slice().ok_or_else(invalid)?,
                    publish_complex,
                    &mut f32_kernel,
                )?,
                NumericDType::F64 => output.map_promoting_f64(
                    runtime,
                    tensor.as_f64_slice().ok_or_else(invalid)?,
                    publish_complex,
                    &mut f64_kernel,
                )?,
                _ => return Err(invalid()),
            }
        }
        _ => return Err(invalid()),
    };
    Ok(scalarize(result))
}

fn scalarize(value: Value) -> Value {
    match value {
        Value::Tensor(tensor) => super::tensor::tensor_into_value(tensor),
        Value::ComplexTensor(tensor) => super::random_args::complex_tensor_into_value(tensor),
        value => value,
    }
}

fn invalid() -> RuntimeError {
    build_runtime_error("selected host unary output differs from its admitted operation or operand")
        .with_identifier("RunMat:Placement:InvalidDecision")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
