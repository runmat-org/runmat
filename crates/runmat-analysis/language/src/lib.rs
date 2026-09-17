//! Source-facing FEA compiler boundaries.
//!
//! This crate owns FEA module and host resolution plus semantic compilation
//! into canonical analysis contracts. It never owns numerical execution,
//! runtime values, filesystem implementations, or scheduler policy.

pub mod binding;
pub mod compiler;
pub mod diagnostic;
pub mod module_graph;
pub mod operations;
pub mod semantic;
pub mod source;

pub use binding::*;
pub use compiler::*;
pub use diagnostic::*;
pub use module_graph::*;
pub use operations::*;
pub use semantic::*;
pub use source::*;
