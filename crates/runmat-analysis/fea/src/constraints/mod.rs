//! Canonical, topology-resolved constraint plans.
//!
//! Authored constraints belong to `runmat-analysis-core`. This module owns the one-way
//! projection from that physical intent onto solver topology. Assembly consumes the resulting
//! plan; it must not reinterpret authored targets or coordinate frames.

mod algebra;
mod compile;
mod contract;
mod coupling;
mod enforcement;
mod explicit_surface;
mod frame;
mod matrix_apply;
mod mixed;
mod mixed_surface;
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

pub use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;
pub use contract::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintRow,
    CanonicalStructuralConstraintTerm, CanonicalStructuralDof, CanonicalStructuralDofAddress,
    CanonicalStructuralDofOwner, CanonicalStructuralReferencePoint, ConstraintCouplingEvidence,
    ConstraintCouplingKind, ConstraintCouplingWeight, ConstraintEnforcement,
    ConstraintGroupEvidence, ConstraintPairingEvidence, ConstraintPairingWeight,
    ConstraintPlanError, ConstraintPlanEvidence, ConstraintResidualQuantity, ReactionRecovery,
    ResolvedConstraintTarget,
};
pub use rank::{
    classify_constraint_rows, constraint_rank_graph_workspace_bytes,
    constraint_rank_workspace_profile, ConstraintRankClassification,
    ConstraintRankWorkspaceProfile,
};
pub use result::{
    CanonicalStructuralConstraintResult, ConstraintGroupResult, ConstraintReferencePointResult,
    ConstraintResultError, ConstraintRowReaction, ConstraintViolation,
};

#[cfg(test)]
mod tests;
pub use algebra::{AffineConstraintReduction, ConstraintAlgebraError, ResolvedConstraintMatrix};
pub(crate) use reaction::solve_constraint_multipliers;
