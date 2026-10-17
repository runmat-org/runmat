//! Invocation-owned CPU work budget for t-SNE preparation and optimization.

use super::{build_runtime_error, with_tsne_context, NAME};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::BuiltinResult;
use std::cmp::Ordering;

pub(super) struct TsneCooperation {
    budget: Option<CpuCooperationBudget>,
    runtime: Option<RuntimeContext>,
    namespace: Option<String>,
}

impl TsneCooperation {
    pub(super) async fn stable_sort_by<T: Copy>(
        &mut self,
        values: &mut Vec<T>,
        compare: impl FnMut(&T, &T) -> Ordering,
    ) -> BuiltinResult<()> {
        use crate::builtins::common::cooperative_sort::{
            SortProgress, SortResourceError, StableSortWork,
        };

        self.completed(0).await?;
        let mut work = StableSortWork::new(values, compare);
        loop {
            let progress = work.advance().map_err(|SortResourceError(source)| {
                super::internal_with_source(
                    "tsne: sorted observations exceed available memory",
                    source,
                )
            })?;
            let units = match progress {
                SortProgress::Pending(units) | SortProgress::Complete(units) => units,
            };
            self.completed(units).await?;
            if matches!(progress, SortProgress::Complete(_)) {
                return Ok(());
            }
        }
    }

    pub(super) fn new(runtime: Option<&RuntimeContext>) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            runtime: runtime.cloned(),
            namespace: runtime.map(RuntimeContext::error_namespace),
        }
    }

    pub(super) async fn normals(&mut self, len: usize) -> BuiltinResult<Vec<f64>> {
        let mut turn =
            crate::builtins::common::random::acquire_turn(self.runtime.as_ref(), NAME).await?;
        match self.budget.as_mut() {
            Some(budget) => crate::builtins::common::random::generate_normal_cooperative(
                &mut turn,
                len,
                NAME,
                self.runtime.as_ref(),
                budget,
            )
            .await
            .map_err(with_tsne_context),
            None => crate::builtins::common::random::generate_normal(&mut turn, len, NAME),
        }
    }

    pub(super) async fn completed(&mut self, units: u64) -> BuiltinResult<()> {
        if let Some(budget) = self.budget.as_mut() {
            budget
                .checkpoint(units)
                .await
                .map_err(|error| match error {
                    source @ ExecutionServiceError::Cancelled => {
                        let namespace = self.namespace.as_deref().unwrap_or("RunMat");
                        build_runtime_error("tsne: execution was cancelled")
                            .with_identifier(format!("{namespace}:Cancelled"))
                            .with_builtin(NAME)
                            .with_source(source)
                            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                            .build()
                    }
                    other => with_tsne_context(other.into_runtime_error()),
                })?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget) -> Self {
        Self {
            budget: Some(budget),
            runtime: None,
            namespace: Some("RunMat".to_string()),
        }
    }
}
