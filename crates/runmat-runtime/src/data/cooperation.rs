//! Dataset work units feed the existing execution-service quantum and token.

use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::BuiltinResult;

pub(crate) async fn cooperate_data_work(
    budget: &mut Option<CpuCooperationBudget>,
    completed_elements: u64,
) -> BuiltinResult<()> {
    let Some(budget) = budget.as_mut() else {
        return Ok(());
    };
    budget
        .checkpoint(completed_elements)
        .await
        .map_err(|error| match error {
            ExecutionServiceError::Cancelled => {
                let mut error = crate::runtime_error::semantic_error(
                    "RunMat:Cancelled",
                    "data invocation was cancelled",
                );
                error.source = Some(Box::new(ExecutionServiceError::Cancelled));
                error
            }
            other => other.into_runtime_error(),
        })
}
