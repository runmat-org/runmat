//! RunMat interactive engine wrapper.
//!
//! This crate now re-exports the host-agnostic session from `runmat-core` so that
//! existing callers (`runmat` CLI, tests, and editor integrations) can continue
//! using `ReplEngine` without pulling in CLI-specific dependencies.

pub use runmat_core::{
    execute_and_format, format_tokens, ExecutionResult, ExecutionStats, RunMatSession,
};

/// Backwards-compatible alias for the session type used by the CLI.
pub type ReplEngine = RunMatSession;
