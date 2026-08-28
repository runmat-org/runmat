//! Portable deterministic execution driver and scheduler.

pub mod backend;
pub mod cancellation;
pub mod driver;
pub mod error;
pub mod pool;
pub mod port;
pub mod recovery;
pub mod scheduler;
pub mod task;
pub mod testing;

pub use driver::{Driver, DriverAction, DriverCommand, DriverConfig, DriverEvent, DriverSnapshot};
pub use error::{RunnerError, RunnerResult};
pub use pool::{PoolSpec, WorkerSpec};
pub use task::{AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess, TaskSubmission};
