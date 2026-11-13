//! Temporary dense-consumer slots keep physical backing before its receipt.
use crate::context::{
    host_allocation::{
        HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedElement,
        HostTypedWorkspace,
    },
    RuntimeContext,
};
use crate::{BuiltinResult, RuntimeError};

pub(in crate::builtins::math::linalg::factor) struct Buffer<T: HostTypedElement> {
    body: Body<T>,
    receipt: Option<HostAllocationReceipt>,
}
enum Body<T: HostTypedElement> {
    Managed(HostTypedWorkspace<T>),
    Ordinary(Vec<T>),
}
fn error(runtime: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    super::with_eig_context(
        crate::context::host_allocation::host_allocation_runtime_error(
            runtime,
            "RunMat:eig:Workspace",
            "dense eig consumer storage",
            source,
        ),
    )
}
impl<T: HostTypedElement> Buffer<T> {
    pub async fn new(count: usize, runtime: Option<&RuntimeContext>) -> BuiltinResult<Self> {
        let bytes = HostTypedWorkspace::<T>::quoted_bytes(count).map_err(|e| error(runtime, e))?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(bytes, 0).map_err(|e| error(runtime, e))?,
        )
        .map_err(|e| error(runtime, e))?;
        let body = if let Some(receipt) = &receipt {
            Body::Managed(
                receipt
                    .temporary_typed(runtime.expect("managed context"), count)
                    .map_err(|e| error(runtime, e))?,
            )
        } else {
            check(runtime, None)?;
            let mut values =
                runmat_value::exact_capacity_vec(count).map_err(|e| error(runtime, e.into()))?;
            let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
            for index in 0..count {
                values.push(T::default());
                if index.is_multiple_of(4096) {
                    if let Some(budget) = &mut budget {
                        budget
                            .checkpoint(4096)
                            .await
                            .map_err(|e| e.into_runtime_error_in_context(runtime))?;
                    }
                }
            }
            check(runtime, None)?;
            Body::Ordinary(values)
        };
        Ok(Self { body, receipt })
    }
    pub fn get(&self) -> &[T] {
        match &self.body {
            Body::Managed(v) => v.as_slice(),
            Body::Ordinary(v) => v,
        }
    }
    pub fn get_mut(&mut self) -> &mut [T] {
        match &mut self.body {
            Body::Managed(v) => v.as_mut_slice(),
            Body::Ordinary(v) => v,
        }
    }
    pub fn check(&self, runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
        check(runtime, self.receipt.as_ref())
    }
}
pub(in crate::builtins::math::linalg::factor) fn check(
    runtime: Option<&RuntimeContext>,
    receipt: Option<&HostAllocationReceipt>,
) -> BuiltinResult<()> {
    if runtime.is_some_and(RuntimeContext::is_cancelled) {
        return Err(crate::execution::ExecutionServiceError::Cancelled
            .into_runtime_error_in_context(runtime));
    }
    if let (Some(runtime), Some(receipt)) = (runtime, receipt) {
        receipt
            .checkpoint(runtime)
            .map_err(|e| error(Some(runtime), e))
    } else {
        crate::builtins::math::linalg::factor::iterative::copy::unbounded_checkpoint(runtime, "eig")
    }
}

pub(in crate::builtins::math::linalg::factor) async fn output_list<const N: usize>(
    runtime: Option<&RuntimeContext>,
    mut values: [Option<runmat_value::Value>; N],
    count: usize,
) -> BuiltinResult<runmat_value::Value> {
    // Numeric eig factors have no GC children; the array retains each complete
    // factor's physical owner until the final output collection is rooted.
    match runtime {
        Some(runtime) => {
            crate::execution::RootedValues::try_initialize_in_context(count, runtime, |i| {
                values[i]
                    .take()
                    .ok_or_else(|| super::eig_internal_error("eig: missing output factor"))
            })
            .map(|values| values.with_owned_output_storage(runmat_value::Value::OutputList))
        }
        None => {
            check(None, None)?;
            let mut output =
                runmat_value::exact_capacity_vec(count).map_err(|e| error(None, e.into()))?;
            for value in values.iter_mut().take(count) {
                output.push(
                    value
                        .take()
                        .ok_or_else(|| super::eig_internal_error("eig: missing output factor"))?,
                );
            }
            check(None, None)?;
            Ok(runmat_value::Value::output_list(output))
        }
    }
}
