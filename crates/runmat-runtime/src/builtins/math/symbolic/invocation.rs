//! Scalar symbolic admission uses the selected caller, never an ambient owner.
#[cfg(test)]
mod tests;
use crate::{
    context::{
        host_allocation::{HostAllocationError, HostAllocationPolicy, HostAllocationReceipt},
        RuntimeContext,
    },
    BuiltinResult,
};
use runmat_builtins::{SymbolicScalarOperand, SymbolicScalarOperation};
use runmat_value::Value;
pub(super) fn admit(
    operation: SymbolicScalarOperation,
    runtime: Option<&RuntimeContext>,
    allocates: bool,
) -> BuiltinResult<()> {
    let policy =
        HostAllocationReceipt::policy(runtime).map_err(|e| control(operation, runtime, e))?;
    if allocates && policy == HostAllocationPolicy::Managed {
        return Err(control(
            operation,
            runtime,
            HostAllocationError::MissingManagedOwner,
        ));
    }
    Ok(())
}
fn control(
    operation: SymbolicScalarOperation,
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> crate::RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    let mut error = crate::builtin::builtin_control_error(
        cancellation.as_ref(),
        operation.catalog_entry().identity.name,
        source,
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
pub(super) fn is_text(value: &Value) -> bool {
    super::support::is_text_scalar(value)
}
pub(super) fn accepts(value: &Value, role: SymbolicScalarOperand) -> bool {
    use SymbolicScalarOperand::*;
    if is_text(value) {
        return matches!(
            role,
            DecimalInput | Variable | VariableOrPoint | Direction | Digits
        );
    }
    match value {
        Value::Symbolic(_) => matches!(
            role,
            Algebra | DecimalInput | Variable | VariableOrPoint | Point
        ),
        Value::Num(_) | Value::Int(_) => !matches!(role, Variable | Direction),
        Value::Bool(_) => !matches!(role, Variable | Direction | Digits),
        Value::Tensor(t) => t.len() == 1 && !matches!(role, Variable | Direction),
        _ => false,
    }
}
pub(super) fn validate(
    operation: SymbolicScalarOperation,
    index: usize,
    count: usize,
    value: &Value,
) -> BuiltinResult<()> {
    if accepts(value, operation.operand(index, count)) {
        return Ok(());
    }
    Err(terminal(crate::runtime_descriptor_error(
        operation.catalog_entry().identity.name,
        operation.operand_error(index, count),
    )))
}
pub(super) fn terminal(mut error: crate::RuntimeError) -> crate::RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
