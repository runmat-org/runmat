//! Native process composition for the portable RunMat execution driver.

mod config;
mod driver;
mod durable;
mod error;
mod local_store;
mod protocol;
mod service;
pub mod supervisor;
mod worker;

pub use config::NativeExecutionConfig;
pub use error::{NativeExecutionError, NativeExecutionResult};
pub use protocol::WorkerResponse;
pub use service::NativeExecutionService;
pub use worker::run_worker_stdio;
