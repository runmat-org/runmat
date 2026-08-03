//! Portable execution contracts shared by native, browser, test, and remote hosts.

mod error;
pub mod handle;
pub mod identity;
pub mod protocol;
pub mod resource;
pub mod schema;
pub mod state;
pub mod task;
pub mod value;

pub use error::ContractError;
pub use identity::{Digest, DomainContribution, ProgramEnvironment, ProgramRevision};
