use super::{ExecutionFailure, ExecutionRequest, ExecutionResponse};

pub trait TestExecutor {
    fn execute(
        &mut self,
        request: &ExecutionRequest,
    ) -> Result<ExecutionResponse, ExecutionFailure>;
}
