//! Dense real floating construction, after shape/class admission.
use super::{host, provider};
use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId, COMPLEX_CATALOG_ENTRY};
use runmat_value::{NumericDType, Value};

fn dense_float(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if matches!(t.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
}

fn admission(binary: bool) -> BuiltinPreparationAdmission {
    BuiltinPreparationAdmission {
        binding: BuiltinBindingIdentity {
            builtin: COMPLEX_CATALOG_ENTRY.identity,
            variant: "default",
        },
        form: BuiltinPreparationFormId(u16::from(binary)),
        signature: usize::from(binary),
        input_count: if binary { 2 } else { 1 },
        requested_outputs: crate::output_count::current_output_count(),
    }
}

pub(super) async fn unary(
    runtime: Option<&RuntimeContext>,
    real: Value,
) -> BuiltinResult<RootedValue> {
    let _input = RootedValueLease::for_values(std::iter::once(&real))
        .map_err(|cause| cause.into_runtime_error())?;
    if !dense_float(&real) {
        let output = provider::unary(real).await?;
        return RootedValue::new(output).map_err(|cause| cause.into_runtime_error());
    }
    execute_invocation(
        runtime,
        admission(false),
        vec![real],
        |mut values| async move {
            let output = provider::unary(values.remove(0)).await?;
            RootedValue::new(output).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await
}

pub(super) async fn binary(
    runtime: Option<&RuntimeContext>,
    real: Value,
    imaginary: Value,
) -> BuiltinResult<RootedValue> {
    let _inputs = RootedValueLease::for_values([&real, &imaginary].into_iter())
        .map_err(|cause| cause.into_runtime_error())?;
    if !dense_float(&real) || !dense_float(&imaginary) {
        let output = provider::binary(real, imaginary).await?;
        return RootedValue::new(output).map_err(|cause| cause.into_runtime_error());
    }
    if let (Value::Tensor(real), Value::Tensor(imaginary)) = (&real, &imaginary) {
        host::compatible_shape(real, imaginary)?;
    }
    execute_invocation(
        runtime,
        admission(true),
        vec![real, imaginary],
        |mut values| async move {
            let imaginary = values.remove(1);
            let output = provider::binary(values.remove(0), imaginary).await?;
            RootedValue::new(output).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await
}
