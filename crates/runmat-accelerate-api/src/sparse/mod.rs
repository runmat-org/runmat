//! Provider-neutral sparse linear-algebra contracts.
//!
//! Canonical numerical domains own their sparse operators. Providers receive validated patterns
//! and values and return opaque, device-local preparations; provider handles are never logical
//! matrix identities and cannot be transported as authored data.

mod capability;
mod execution;
mod handles;
mod multigrid;
mod operations;
mod pattern;
mod types;

pub use capability::*;
pub use execution::*;
pub use handles::*;
pub use multigrid::*;
pub use operations::*;
pub use pattern::*;
pub use types::*;

#[cfg(test)]
mod tests;
