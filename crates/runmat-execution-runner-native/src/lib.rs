//! Native process composition for the portable RunMat execution driver.

mod config;
mod driver;
mod error;
mod local_store;
mod protocol;
mod service;
mod worker;

pub use config::NativeExecutionConfig;
pub use error::{NativeExecutionError, NativeExecutionResult};
pub use service::NativeExecutionService;
pub use worker::run_worker_stdio;
