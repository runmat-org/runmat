//! Canonical, topology-resolved constraint plans.
//!
//! Authored constraints belong to `runmat-analysis-core`. This module owns the one-way
//! projection from that physical intent onto solver topology. Assembly consumes the resulting
//! plan; it must not reinterpret authored targets or coordinate frames.

mod algebra;
#[path = "algebra/translation_selection.rs"]
mod translation_selection;
pub(crate) use translation_selection::HomogeneousTranslationSelection;
mod compilation;
mod compile;
mod contract;
mod coupling;
mod enforcement;
mod explicit_surface;
mod frame;
mod lowering;
pub use lowering::{constraint_row_expansion_profile, ConstraintRowExpansionProfile};
mod matrix_apply;
mod mixed;
mod mixed_surface;
mod normal;
mod pairing;
mod phase;
mod plan;
mod prepared;
mod rank;
mod reaction;
mod result;
mod result_state;
mod row;
mod target;
mod vector;
mod workspace;
pub use compile::{
    constraint_reference_point_workspace_profile, ConstraintReferencePointWorkspaceProfile,
};
pub use coupling::{constraint_coupling_workspace_profile, ConstraintCouplingWorkspaceProfile};
pub use normal::{constraint_normal_workspace_profile, ConstraintNormalWorkspaceProfile};
pub(crate) use pairing::pair_targets;
pub use pairing::{constraint_pairing_workspace_profile, ConstraintPairingWorkspaceProfile};
pub use phase::{build_phase_reduction, UnitPhaseConstraintError, UnitPhaseRelation};
pub(crate) use target::resolve_nodal_target;
pub use target::{constraint_target_workspace_profile, ConstraintTargetWorkspaceProfile};

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
mod compilation_tests;
#[cfg(test)]
mod construction_tests;
#[cfg(test)]
mod tests;
pub use algebra::{AffineConstraintReduction, ConstraintAlgebraError, ResolvedConstraintMatrix};
pub(crate) use reaction::solve_constraint_multipliers;

pub use plan::{
    constraint_authored_workspace_profile, constraint_plan_finalization_workspace_profile,
    ConstraintAuthoredWorkspaceProfile, ConstraintPlanFinalizationWorkspaceProfile,
};
pub use row::{constraint_row_workspace_profile, ConstraintRowWorkspaceProfile};
