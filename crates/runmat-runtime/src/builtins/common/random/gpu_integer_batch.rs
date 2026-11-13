//! Bounded native GPU integer batches, selected before stream reservation.
//! Each retry is admitted and physically acknowledged before its successor.

mod execution;
mod progress;
mod selection;

#[cfg(test)]
mod protocol_tests;

pub(crate) use execution::execute_selected;
pub(crate) use selection::{select_initial_route, InitialIntegerRoute, SelectedInitialNativeBatch};

use crate::{build_runtime_error, RuntimeError};

fn error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("randi").build()
}

fn provider_error(source: runmat_accelerate_api::ProviderExecutionError) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        error(format!("randi: native GPU integer batch failed: {source}")),
        source,
    )
}
