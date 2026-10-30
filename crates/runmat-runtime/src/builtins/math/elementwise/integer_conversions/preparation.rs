use crate::builtins::common::integer_conversion::{admits_preparation, cast_value, CastError};
use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId};
use runmat_types::IntegerClass;
use runmat_value::Value;

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    value: Value,
    target: IntegerClass,
    error: fn(CastError) -> RuntimeError,
) -> BuiltinResult<RootedValue> {
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    if !admits_preparation(&value).map_err(error)? {
        let converted = cast_value(value, target).await.map_err(error)?;
        return RootedValue::new(converted).map_err(|cause| cause.into_runtime_error());
    }
    if let Value::GpuTensor(handle) = &value {
        crate::builtins::common::gpu_helpers::require_resident_compute_owner(handle, || {
            error(CastError::Internal(
                "no acceleration provider owns the input handle".into(),
            ))
        })?;
    }
    execute_invocation(
        runtime,
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 1,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
        |mut operands| async move {
            // The shared carrier checks the count. Re-admit inside cast_value:
            // an injected service may intentionally transform its operand.
            let converted = cast_value(operands.pop().expect("one admitted operand"), target)
                .await
                .map_err(error)?;
            RootedValue::new(converted).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await
}
