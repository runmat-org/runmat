//! Executor-neutral MATLAB indexing plans, selectors, reads, and mutations.

pub mod end_expr;
pub mod integer_assignment;
pub mod plan;
pub mod read_linear;
pub mod read_slice;
pub mod selectors;
pub mod write_linear;
pub mod write_slice;

pub use end_expr::{value_to_f64, EndExpr, ValueToF64Error};
