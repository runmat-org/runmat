//! Equation-problem public contracts.
mod contract;
pub use contract::{EquationProblemProperty, EQUATION_PROBLEM_CLASS};
mod descriptor;
mod documentation;
mod entry;
mod inference;
mod solve;
pub use descriptor::*;
pub use entry::*;
pub(super) use solve::infer_solve;
pub use solve::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &EQNPROBLEM_CATALOG_ENTRY,
    &EQNPROBLEM_SUBSREF_CATALOG_ENTRY,
    &EQNPROBLEM_SUBSASGN_CATALOG_ENTRY,
    &EQNPROBLEM_SOLVE_METHOD_CATALOG_ENTRY,
];
