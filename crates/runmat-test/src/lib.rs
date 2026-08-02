//! Portable test-domain and lifecycle authority for RunMat.

pub mod context;
pub mod descriptor;
pub mod discovery;
pub mod error;
pub mod event;
pub mod executor;
pub mod identity;
pub mod lifecycle;
pub mod plan;
pub mod protocol;
pub mod result;
pub mod version;

pub use error::TestDomainError;
pub use version::{PROTOCOL_VERSION, TEST_IDENTITY_ALGORITHM_VERSION, TEST_PLAN_SCHEMA_VERSION};
