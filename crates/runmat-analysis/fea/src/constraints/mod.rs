//! Canonical, topology-resolved constraint plans.
//!
//! Authored constraints belong to `runmat-analysis-core`. This module owns the one-way
//! projection from that physical intent onto solver topology. Assembly consumes the resulting
//! plan; it must not reinterpret authored targets or coordinate frames.

mod contract;
mod plan;
mod result;
mod target;

pub use contract::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintRow, CanonicalStructuralDof,
    ConstraintEnforcement, ConstraintGroupEvidence, ConstraintPlanError, ConstraintPlanEvidence,
    ReactionRecovery, ResolvedConstraintTarget,
};
pub use result::{
    CanonicalStructuralConstraintResult, ConstraintGroupResult, ConstraintResultError,
};

#[cfg(test)]
mod tests;
