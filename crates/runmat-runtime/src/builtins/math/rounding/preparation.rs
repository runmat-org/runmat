//! Automatic preparation after the owning rounding form has admitted its inputs.
use crate::context::preparation::{
    execute_current_invocation, execute_current_request, AdmittedBuiltinPreparation,
    BuiltinPreparationAdmission,
};
use crate::BuiltinResult;
use runmat_accelerate_api::{NumericalDomainEvidence, RoundDigitsMode};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinPreparationFormId};
use runmat_value::{NumericDType, Value};
use std::future::Future;

pub(super) async fn digits<F, Fut>(
    value: Value,
    digits: i32,
    mode: RoundDigitsMode,
    control_count: usize,
    body: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<Value>>,
{
    if !matches!(&value, Value::Tensor(tensor) if matches!(tensor.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
    {
        return body(value).await;
    }
    // The owner parsed the controls before reaching this boundary. They travel
    // as operation evidence, never as uploaded numerical operands.
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: runmat_builtins::ROUND_CATALOG_ENTRY.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(1),
            signature: control_count,
            input_count: control_count + 1,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
    )?
    .with_numerical_domain(|_| Ok(NumericalDomainEvidence::RoundDigits { digits, mode }))?;
    execute_current_request(request, |mut values| body(values.remove(0))).await
}

pub(super) async fn unary<F, Fut>(
    entry: &'static BuiltinCatalogEntry,
    value: Value,
    body: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<Value>>,
{
    // These are the real dense forms served by the unary provider hooks.
    // Integer identity, complex, tabular and resident forms keep their own body.
    if !matches!(&value, Value::Tensor(tensor) if matches!(tensor.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
    {
        return body(value).await;
    }
    execute_current_invocation(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 1,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
        |mut values| body(values.remove(0)),
    )
    .await
}
