//! Canonical structural load projection, independent of solver row placement.

mod apply;
mod compile;
mod contract;

pub(super) use apply::apply;
pub(super) use compile::{compile_explicit, compile_mixed, compile_solid};
pub(super) use contract::{
    CanonicalStructuralLoadEntry, CanonicalStructuralLoadPlan, CanonicalStructuralLoadTerm,
    StructuralLoadPlanError,
};
