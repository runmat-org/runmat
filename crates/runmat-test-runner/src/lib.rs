#![allow(clippy::result_large_err)]

pub mod artifact;
pub mod coordinator;
mod error;
pub mod host;
pub mod reporter;
pub mod schedule;
pub mod telemetry;
pub mod worker;

pub use coordinator::{CoordinatedRun, Coordinator, CoordinatorConfig};
pub use error::{RunnerError, RunnerResult};
