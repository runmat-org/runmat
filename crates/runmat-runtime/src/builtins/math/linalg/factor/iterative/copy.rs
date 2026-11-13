//! Bounded copies into independently claimed iterative language values.
//! Construction owners, not published Values or recurrence slices, cross
//! cooperation awaits. Dropping the future retires an unfinished output.
use super::value::IterativeValue;
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPolicy, HostAllocationReceipt},
    RuntimeContext,
};
use crate::RuntimeError;
use runmat_value::Value;

pub(in crate::builtins::math::linalg::factor) fn admission(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    error: HostAllocationError,
) -> RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    match crate::context::host_allocation::HostAllocationDisposition::from(error) {
        crate::context::host_allocation::HostAllocationDisposition::Ordinary(error) => {
            crate::build_runtime_error(format!("{builtin}: host admission failed: {error}"))
                .with_builtin(builtin)
                .with_identifier(format!("RunMat:{builtin}:HostAllocation"))
                .with_source(error)
                .build()
        }
        _ => crate::builtin::builtin_control_error(cancellation.as_ref(), builtin, error),
    }
}

pub(in crate::builtins::math::linalg::factor) fn unbounded_checkpoint(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
) -> Result<(), RuntimeError> {
    if HostAllocationReceipt::policy(runtime).map_err(|e| admission(runtime, builtin, e))?
        != HostAllocationPolicy::Unbounded
    {
        return Err(admission(
            runtime,
            builtin,
            HostAllocationError::StaleAuthority,
        ));
    }
    Ok(())
}

pub(in crate::builtins::math::linalg::factor) async fn admitted<T: IterativeValue>(
    receipt: &HostAllocationReceipt,
    runtime: &RuntimeContext,
    source: &[T],
    builtin: &'static str,
) -> Result<Value, RuntimeError> {
    let mut budget = runtime.cpu_cooperation();
    admitted_with_budget::<T>(receipt, runtime, source, builtin, &mut budget).await
}

pub(in crate::builtins::math::linalg::factor) async fn admitted_with_budget<T: IterativeValue>(
    receipt: &HostAllocationReceipt,
    runtime: &RuntimeContext,
    source: &[T],
    builtin: &'static str,
    budget: &mut crate::execution::CpuCooperationBudget,
) -> Result<Value, RuntimeError> {
    let fail = |e| admission(Some(runtime), builtin, e);
    let mut output = receipt
        .with_output_allocation(runtime, |permit| T::allocate_output(source.len(), permit))
        .map_err(fail)?;
    for (index, chunk) in source.chunks(1024).enumerate() {
        receipt.checkpoint(runtime).map_err(fail)?;
        budget
            .checkpoint(chunk.len() as u64)
            .await
            .map_err(|e| e.into_runtime_error_in_context(Some(runtime)))?;
        receipt.checkpoint(runtime).map_err(fail)?;
        T::copy_output(&mut output, index * 1024, chunk)
            .map_err(|e| fail(HostAllocationError::Numeric(e)))?;
    }
    receipt.checkpoint(runtime).map_err(fail)?;
    let value = T::finish_output(output).map_err(|e| fail(HostAllocationError::Numeric(e)))?;
    receipt.checkpoint(runtime).map_err(fail)?;
    Ok(value)
}

pub(in crate::builtins::math::linalg::factor) async fn unbounded<T: IterativeValue>(
    runtime: Option<&RuntimeContext>,
    source: &[T],
    builtin: &'static str,
) -> Result<Value, RuntimeError> {
    unbounded_checkpoint(runtime, builtin)?;
    let mut values = Vec::new();
    values.try_reserve_exact(source.len()).map_err(|_| {
        admission(
            runtime,
            builtin,
            HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            ),
        )
    })?;
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    for chunk in source.chunks(1024) {
        if let Some(budget) = budget.as_mut() {
            budget
                .checkpoint(chunk.len() as u64)
                .await
                .map_err(|e| e.into_runtime_error_in_context(runtime))?;
        }
        unbounded_checkpoint(runtime, builtin)?;
        values.extend_from_slice(chunk);
    }
    let value = T::unbounded_output(values).map_err(|message| {
        crate::build_runtime_error(message)
            .with_builtin(builtin)
            .build()
    })?;
    unbounded_checkpoint(runtime, builtin)?;
    Ok(value)
}
