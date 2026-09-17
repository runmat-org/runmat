//! Source-facing FEA compiler boundaries.
//!
//! This crate owns FEA module and host resolution plus semantic compilation
//! into canonical analysis contracts. It never owns numerical execution,
//! runtime values, filesystem implementations, or scheduler policy.

pub mod diagnostic;
pub mod operations;
pub mod source;

pub use diagnostic::*;
pub use operations::*;
pub use source::*;
