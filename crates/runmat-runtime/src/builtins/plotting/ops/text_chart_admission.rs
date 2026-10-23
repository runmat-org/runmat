//! Fence ordinary chart scratch until its complete finite owner is available.
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use std::future::Future;

#[cfg(test)]
#[path = "text_chart_admission/tests.rs"]
mod tests;

pub(crate) struct UnboundedChartWork<'a> {
    runtime: Option<&'a RuntimeContext>,
    builtin: &'a str,
}
impl<'a> UnboundedChartWork<'a> {
    pub(crate) fn admit(
        runtime: Option<&'a RuntimeContext>,
        builtin: &'a str,
    ) -> BuiltinResult<Self> {
        let work = Self { runtime, builtin };
        work.check()?;
        Ok(work)
    }
    pub(crate) fn check(&self) -> BuiltinResult<()> {
        let cause = match HostAllocationReceipt::policy(self.runtime) {
            Ok(HostAllocationPolicy::Unbounded) => return Ok(()),
            Ok(HostAllocationPolicy::Managed) => HostAllocationError::MissingManagedOwner,
            Err(cause) => cause,
        };
        let mut error = host_allocation_runtime_error(
            self.runtime,
            "RunMat:plotting:TextChartWorkspace",
            format!(
                "{}: complete finite text-chart workspace ownership is not implemented; {cause}",
                self.builtin
            ),
            cause,
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(error)
    }
    pub(crate) fn finish<T>(&self, result: BuiltinResult<T>) -> BuiltinResult<T> {
        let output = result.map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?;
        self.check()?;
        Ok(output)
    }
    pub(crate) async fn wait<T>(
        &self,
        stage: impl Future<Output = BuiltinResult<T>>,
    ) -> BuiltinResult<T> {
        self.check()?;
        self.finish(stage.await)
    }
}
