//! Source-facing FEA compiler boundaries.
//!
//! This crate owns FEA module and host resolution plus semantic compilation
//! into canonical analysis contracts. It never owns numerical execution,
//! runtime values, filesystem implementations, or scheduler policy.

pub mod authoring;
pub mod binding;
pub mod compiler;
pub mod diagnostic;
pub mod module_graph;
pub mod operations;
pub mod reference;
pub mod semantic;
pub mod source;

pub use authoring::{FeaAuthoringWordRole, FeaContextualWord};
pub use binding::*;
pub use compiler::*;
pub use diagnostic::*;
pub use module_graph::*;
pub use operations::*;
pub use runmat_design_syntax::FeaModelDefinitionKind;
pub use semantic::*;
pub use source::*;
