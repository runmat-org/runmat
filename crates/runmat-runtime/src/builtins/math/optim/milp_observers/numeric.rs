//! Admit only the numerical fields a selected observer actually reads.
use super::*;
use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};
pub(super) fn preflight(kind: MilpObserverKind, value: &Value) -> BuiltinResult<()> {
    if !matches!(
        value,
        Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_)
    ) || !matches!(
        crate::value_fact::value_fact(value).kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single | NumericClass::Double,
            domain: NumericDomain::Real
        })
    ) {
        return Err(invalid(
            kind,
            "needed observer values must be real single/double data",
        ));
    }
    if crate::builtins::common::residency::host_gather_intent(value)?
        == crate::builtins::common::residency::HostGatherIntent::ExplicitOrUnproven
    {
        return Err(invalid(kind,"needed observer data requires automatic residency; gather explicit or unproven device data first"));
    }
    Ok(())
}
pub(super) async fn gather(
    value: &Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_in_context_async(value, cooperation.runtime()).await
}
pub(super) fn scalar(kind: MilpObserverKind, value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Tensor(value) if value.len() == 1 && value.integer_storage().is_none() => {
            Ok(crate::builtins::common::tensor::tensor_value_f64(value, 0))
        }
        _ => Err(invalid(
            kind,
            "needed optimValues field must be a floating scalar",
        )),
    }
}
