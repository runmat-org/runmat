//! Invocation-local cooperation for image-owned host work. The execution
//! service chooses whether completed work yields a task turn.

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult};

/// Maximum uninterrupted image-owned scalar work. This is not a scheduler
/// quantum; `CpuCooperationBudget` retains that policy.
pub(crate) const IMAGE_WORK_CHUNK: usize = 4096;

pub(crate) struct ImageCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    builtin: &'static str,
}

impl ImageCooperation {
    pub(crate) fn new(runtime: Option<&RuntimeContext>, builtin: &'static str) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
            builtin,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_budget(budget: CpuCooperationBudget, builtin: &'static str) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".into()),
            builtin,
        }
    }

    pub(crate) async fn completed(&mut self, units: usize) -> BuiltinResult<()> {
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        budget
            .checkpoint(units as u64)
            .await
            .map_err(|error| match error {
                ExecutionServiceError::Cancelled => {
                    build_runtime_error(format!("{}: execution was cancelled", self.builtin))
                        .with_identifier(format!(
                            "{}:Cancelled",
                            self.namespace.as_deref().unwrap_or("RunMat")
                        ))
                        .with_builtin(self.builtin)
                        .build()
                }
                other => other.into_runtime_error(),
            })
    }

    /// Fill an already reserved native buffer without one uninterruptible
    /// `resize(len)` operation. A dropped future owns and releases its partial
    /// buffer; it cannot publish a partly filled image.
    pub(crate) async fn fill<T: Copy + Default>(
        &mut self,
        values: &mut Vec<T>,
        len: usize,
    ) -> BuiltinResult<()> {
        self.completed(0).await?;
        while values.len() < len {
            let next = values.len().saturating_add(IMAGE_WORK_CHUNK).min(len);
            let completed = next - values.len();
            values.resize(next, T::default());
            self.completed(completed).await?;
        }
        Ok(())
    }
}
