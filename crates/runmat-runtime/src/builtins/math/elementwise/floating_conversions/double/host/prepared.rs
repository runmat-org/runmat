use super::double_error_with_detail;
use crate::context::host_allocation::host_allocation_runtime_error;
use crate::context::preparation::PreparedHostNumericalOutput;
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_builtins::DOUBLE_ERROR_INTERNAL;
use runmat_types::NumericClass;
use runmat_value::{NumericDType, Tensor, Value};

pub(super) fn from_tensor(
    runtime: &RuntimeContext,
    tensor: Tensor,
    mut output: PreparedHostNumericalOutput,
) -> BuiltinResult<Value> {
    if !output.matches_floating_cast(NumericClass::Double, NumericDType::F64, tensor.shape()) {
        return Err(double_error_with_detail(
            &DOUBLE_ERROR_INTERNAL,
            "selected exact double output does not match the prepared source",
        ));
    }
    let copied = match (tensor.as_f64_slice(), tensor.as_f32_slice()) {
        (Some(source), _) => output.copy_f64_from_f64(runtime, source),
        (_, Some(source)) => output.copy_f64_from_f32(runtime, source),
        _ => {
            return Err(double_error_with_detail(
                &DOUBLE_ERROR_INTERNAL,
                "selected exact double source has an unsupported native class",
            ));
        }
    };
    copied.map_err(|error| {
        host_allocation_runtime_error(
            Some(runtime),
            DOUBLE_ERROR_INTERNAL
                .identifier
                .unwrap_or("RunMat:double:Internal"),
            "selected exact double output authority changed",
            error,
        )
    })?;
    output
        .into_tensor(runtime)
        .map(Value::Tensor)
        .map_err(|error| {
            host_allocation_runtime_error(
                Some(runtime),
                DOUBLE_ERROR_INTERNAL
                    .identifier
                    .unwrap_or("RunMat:double:Internal"),
                "selected exact double output could not be published",
                error,
            )
        })
}
