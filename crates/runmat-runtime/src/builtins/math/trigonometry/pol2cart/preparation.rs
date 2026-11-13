//! The parsed coordinate form owns data roles and full broadcast admission.
use super::*;
use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId, POL2CART_CATALOG_ENTRY};
use runmat_value::NumericDType;

fn floating_class(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Num(_) => Some(NumericDType::F64),
        Value::Tensor(tensor) => match tensor.numeric_dtype() {
            dtype @ (NumericDType::F32 | NumericDType::F64) => Some(dtype),
            _ => None,
        },
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none() =>
        {
            runmat_accelerate_api::handle_precision(handle).map(|precision| match precision {
                runmat_accelerate_api::ProviderPrecision::F32 => NumericDType::F32,
                runmat_accelerate_api::ProviderPrecision::F64 => NumericDType::F64,
            })
        }
        _ => None,
    }
}

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    inputs: Vec<Value>,
    shape: &[usize],
    requested_outputs: Option<usize>,
) -> BuiltinResult<RootedValue> {
    let _inputs =
        RootedValueLease::for_values(inputs.iter()).map_err(|cause| cause.into_runtime_error())?;
    let class = floating_class(&inputs[0]);
    if class.is_none() || inputs.iter().any(|value| floating_class(value) != class) {
        return super::execute(inputs, shape, requested_outputs).await;
    }
    let count = inputs.len();
    execute_invocation(
        runtime,
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: POL2CART_CATALOG_ENTRY.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId((count - 2) as u16),
            signature: count - 2,
            input_count: count,
            requested_outputs,
        },
        inputs,
        |inputs| async move {
            let prepared_shape = admission::admit(&inputs, requested_outputs)?;
            if prepared_shape != shape || inputs.iter().any(|value| floating_class(value) != class)
            {
                return Err(pol2cart_error(
                    &ERROR_INTERNAL,
                    "preparation changed admitted coordinate shape or class",
                ));
            }
            super::execute(inputs, shape, requested_outputs).await
        },
    )
    .await
}
