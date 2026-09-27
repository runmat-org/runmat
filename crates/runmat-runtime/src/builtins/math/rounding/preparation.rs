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

pub(super) async fn remainder<F, Fut>(
    entry: &'static BuiltinCatalogEntry,
    kind: runmat_types::FloatingRemainderKind,
    left: Value,
    right: Value,
    shape_error: impl FnOnce(String) -> crate::RuntimeError,
    body: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<Value>>,
{
    let eligible = matches!((&left, &right), (Value::Tensor(a), Value::Tensor(b))
        if matches!(a.numeric_dtype(), NumericDType::F32 | NumericDType::F64)
            && a.numeric_dtype() == b.numeric_dtype());
    if !eligible {
        return body(left, right).await;
    }
    let (Value::Tensor(a), Value::Tensor(b)) = (&left, &right) else {
        unreachable!()
    };
    let shape = crate::builtins::common::broadcast::broadcast_shapes(
        entry.identity.name,
        &a.shape,
        &b.shape,
    );
    let shape = shape
        .and_then(|shape| {
            if !shape.contains(&0)
                && shape
                    .iter()
                    .try_fold(1_usize, |n, d| n.checked_mul(*d))
                    .is_none()
            {
                Err("broadcast element count overflow".into())
            } else {
                Ok(shape)
            }
        })
        .map_err(shape_error)?;
    let _ = shape;
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 2,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![left, right],
    )?
    .with_numerical_domain(|_| Ok(NumericalDomainEvidence::FloatingRemainder(kind)))?;
    execute_current_request(request, |mut values| {
        let right = values.remove(1);
        body(values.remove(0), right)
    })
    .await
}

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
