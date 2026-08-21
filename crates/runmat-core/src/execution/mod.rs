mod environment;
mod types;

use anyhow::Result;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub use environment::{
    program_environment, PROGRAM_COMPILER_SCHEMA, PROGRAM_RUNTIME_ABI_SCHEMA,
    PROGRAM_SEMANTIC_SCHEMA,
};
pub use types::*;

pub(crate) type SharedAsyncInputHandler = Arc<
    dyn Fn(InputRequest) -> Pin<Box<dyn Future<Output = Result<InputResponse, String>> + 'static>>
        + Send
        + Sync,
>;
