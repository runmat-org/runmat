//! Borrow one exact original task independent of its existing worker route.
use super::model::{GraphBatchSubmission, ProgramBatchSubmission};
use crate::{NativeExecutionError, NativeExecutionResult};
pub(super) enum RetainedSubmission {
    Direct(Box<ProgramBatchSubmission>),
    Graph(Box<GraphBatchSubmission>),
}
impl RetainedSubmission {
    pub(super) fn task(&self) -> &runmat_execution_runner::TaskSubmission {
        match self {
            Self::Direct(s) => &s.task,
            Self::Graph(s) => s
                .task
                .as_ref()
                .expect("retained Graph requires original task"),
        }
    }
    pub(super) fn validate_response(
        &self,
        response: &crate::protocol::WorkerResponse,
    ) -> NativeExecutionResult<()> {
        match self {
            Self::Direct(s) => response.validate_against(&s.program_request()),
            Self::Graph(s) => s.request.validate_response(response),
        }
        .map_err(|e| NativeExecutionError::Protocol(e.to_string()))
    }
}
