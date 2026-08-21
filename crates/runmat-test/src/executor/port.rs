use std::future::Future;
use std::pin::Pin;

use super::{ExecutionFailure, ExecutionRequest, ExecutionResponse};

pub type ExecutionFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ExecutionResponse, ExecutionFailure>> + 'a>>;

pub trait TestExecutor {
    fn execute<'a>(&'a mut self, request: &'a ExecutionRequest) -> ExecutionFuture<'a>;
}
