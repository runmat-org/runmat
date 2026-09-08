//! Canonical, topology-resolved constraint plans.
//!
//! Authored constraints belong to `runmat-analysis-core`. This module owns the one-way
//! projection from that physical intent onto solver topology. Assembly consumes the resulting
//! plan; it must not reinterpret authored targets or coordinate frames.

mod algebra;
mod compile;
mod contract;
mod coupling;
mod frame;
mod normal;
mod pairing;
mod plan;
mod rank;
mod reaction;
mod result;
mod result_state;
mod row;
mod target;
mod vector;

pub use contract::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintRow,
    CanonicalStructuralConstraintTerm, CanonicalStructuralDof, CanonicalStructuralDofOwner,
    CanonicalStructuralReferencePoint, ConstraintCouplingEvidence, ConstraintCouplingKind,
    ConstraintCouplingWeight, ConstraintEnforcement, ConstraintGroupEvidence,
    ConstraintPairingEvidence, ConstraintPairingWeight, ConstraintPlanError,
    ConstraintPlanEvidence, ReactionRecovery, ResolvedConstraintTarget,
};
pub use result::{
    CanonicalStructuralConstraintResult, ConstraintGroupResult, ConstraintReferencePointResult,
    ConstraintResultError, ConstraintRowReaction,
};

#[cfg(test)]
mod tests;
pub use algebra::{AffineConstraintReduction, ConstraintAlgebraError, ResolvedConstraintMatrix};
