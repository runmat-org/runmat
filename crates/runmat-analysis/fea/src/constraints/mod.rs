//! Canonical, topology-resolved constraint plans.
//!
//! Authored constraints belong to `runmat-analysis-core`. This module owns the one-way
//! projection from that physical intent onto solver topology. Assembly consumes the resulting
//! plan; it must not reinterpret authored targets or coordinate frames.

mod algebra;
mod compile;
mod contract;
mod frame;
mod normal;
mod plan;
mod rank;
mod result;
mod row;
mod target;
mod vector;

pub use contract::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintRow,
    CanonicalStructuralConstraintTerm, CanonicalStructuralDof, ConstraintEnforcement,
    ConstraintGroupEvidence, ConstraintPlanError, ConstraintPlanEvidence, ReactionRecovery,
    ResolvedConstraintTarget,
};
pub use result::{
    CanonicalStructuralConstraintResult, ConstraintGroupResult, ConstraintResultError,
    ConstraintRowReaction,
};

#[cfg(test)]
mod tests;
pub use algebra::{AffineConstraintReduction, ConstraintAlgebraError, ResolvedConstraintMatrix};
