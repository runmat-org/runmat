use runmat_analysis_core::{
    BoundaryTransferImplementation, RegionTarget, StructuralRotationDof, StructuralTranslationDof,
};
use runmat_execution::Digest;
use runmat_meshing_core::{PersistentEntityId, StableDigest};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalStructuralDof {
    Translation(StructuralTranslationDof),
    Rotation(StructuralRotationDof),
}

impl CanonicalStructuralDof {
    pub(super) const TRANSLATIONS: [Self; 3] = [
        Self::Translation(StructuralTranslationDof::Ux),
        Self::Translation(StructuralTranslationDof::Uy),
        Self::Translation(StructuralTranslationDof::Uz),
    ];
}

impl From<StructuralTranslationDof> for CanonicalStructuralDof {
    fn from(value: StructuralTranslationDof) -> Self {
        Self::Translation(value)
    }
}

impl From<StructuralRotationDof> for CanonicalStructuralDof {
    fn from(value: StructuralRotationDof) -> Self {
        Self::Rotation(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintEnforcement {
    DirectElimination,
    AffineReduction,
    LagrangeMultiplier,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReactionRecovery {
    UnconstrainedEquilibriumResidual,
    ConstraintMultiplier,
    RedundantEquation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedConstraintTarget {
    pub authored: RegionTarget,
    pub matched_entities: Vec<PersistentEntityId>,
    pub matched_node_identities: Vec<StableDigest>,
    pub selection_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintGroupEvidence {
    pub constraint_id: String,
    pub source_digest: Digest,
    pub implementation: BoundaryTransferImplementation,
    pub targets: Vec<ResolvedConstraintTarget>,
    pub reference_point_identities: Vec<Digest>,
    pub coupling: Option<ConstraintCouplingEvidence>,
    pub pairings: Vec<ConstraintPairingEvidence>,
    pub row_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintCouplingKind {
    Rigid,
    Distributing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintCouplingEvidence {
    pub kind: ConstraintCouplingKind,
    pub reference_point_identity: Digest,
    pub target_weights: Vec<ConstraintCouplingWeight>,
    pub characteristic_length_meters: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintCouplingWeight {
    pub node_identity: StableDigest,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintPairingEvidence {
    pub secondary_node_identity: StableDigest,
    pub primary_weights: Vec<ConstraintPairingWeight>,
    pub geometric_error_meters: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintPairingWeight {
    pub primary_node_identity: StableDigest,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalStructuralConstraintTerm {
    pub owner: CanonicalStructuralDofOwner,
    pub dof: CanonicalStructuralDof,
    pub coefficient: f64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalStructuralDofOwner {
    StructuralNode(StableDigest),
    ReferencePoint(Digest),
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CanonicalStructuralDofAddress {
    pub owner: CanonicalStructuralDofOwner,
    pub dof: CanonicalStructuralDof,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalStructuralReferencePoint {
    pub identity: Digest,
    pub reference_point_id: String,
    pub position_meters: [f64; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalStructuralConstraintRow {
    pub identity: Digest,
    pub constraint_id: String,
    pub terms: Vec<CanonicalStructuralConstraintTerm>,
    pub right_hand_side: f64,
    pub residual_quantity: ConstraintResidualQuantity,
    pub reaction_recovery: ReactionRecovery,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintResidualQuantity {
    Displacement,
    Rotation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintPlanEvidence {
    pub authored_constraint_count: usize,
    pub active_row_count: usize,
    pub redundant_row_count: usize,
    pub conflicting_row_count: usize,
    pub rank: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalStructuralConstraintPlan {
    pub schema_version: u16,
    pub topology: CanonicalStructuralTopologyIdentity,
    pub authored_constraints_digest: Digest,
    pub enforcement: ConstraintEnforcement,
    pub reference_points: Vec<CanonicalStructuralReferencePoint>,
    pub groups: Vec<ConstraintGroupEvidence>,
    pub rows: Vec<CanonicalStructuralConstraintRow>,
    pub evidence: ConstraintPlanEvidence,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ConstraintPlanError {
    #[error("constraint numerical accounting failed: {0}")]
    Accounting(#[from] runmat_simulation_usage::LedgerError),
    #[error("constraint compilation {phase:?} requires {required} requested bytes, above caller limit {maximum}")]
    WorkspaceLimit {
        phase: super::compilation::ConstraintCompilationPhase,
        required: u64,
        maximum: u64,
    },
    #[error("constraint construction cancelled")]
    Cancelled,
    #[error("constraint `{constraint_id}` uses unsupported frame `{frame}`")]
    UnsupportedFrame {
        constraint_id: String,
        frame: String,
    },
    #[error("constraint `{constraint_id}` defines a degenerate coordinate frame")]
    DegenerateFrame { constraint_id: String },
    #[error("constraint `{constraint_id}` requires an unambiguous boundary normal but resolved {entity_count} entities")]
    AmbiguousNormal {
        constraint_id: String,
        entity_count: usize,
    },
    #[error("constraint `{constraint_id}` boundary normal could not be recovered")]
    NormalUnavailable { constraint_id: String },
    #[error("constraint `{constraint_id}` could not pair secondary node {secondary_node_identity:?}: {reason}")]
    PairingUnavailable {
        constraint_id: String,
        secondary_node_identity: StableDigest,
        reason: &'static str,
    },
    #[error("constraint `{constraint_id}` matching pairing is not bijective")]
    NonBijectivePairing { constraint_id: String },
    #[error("constraint `{constraint_id}` targets unsupported topology: {reason}")]
    UnsupportedTarget {
        constraint_id: String,
        reason: &'static str,
    },
    #[error(
        "constraint `{constraint_id}` target `{region_id}` did not resolve to solver topology"
    )]
    EmptyTarget {
        constraint_id: String,
        region_id: String,
    },
    #[error("constraint `{constraint_id}` expected exactly one entity but resolved {actual}")]
    Cardinality {
        constraint_id: String,
        actual: usize,
    },
    #[error("constraint id `{constraint_id}` is duplicated")]
    DuplicateConstraintId { constraint_id: String },
    #[error("constraint `{constraint_id}` prescribes an unavailable rotational DOF")]
    UnavailableRotationalDof { constraint_id: String },
    #[error("constraint `{constraint_id}` requires a study-constant prescribed value")]
    ScheduledValue { constraint_id: String },
    #[error("constraint `{constraint_id}` contains a non-finite prescribed value")]
    NonFiniteValue { constraint_id: String },
    #[error("constraints `{first_constraint_id}` and `{second_constraint_id}` prescribe conflicting values in the same canonical row space")]
    ConflictingRow {
        first_constraint_id: String,
        second_constraint_id: String,
        conflicting_dof: Option<CanonicalStructuralDofAddress>,
    },
    #[error("constraint plan topology does not match the solver mesh")]
    TopologyMismatch,
    #[error("constraint plan does not match the authored structural constraints")]
    AuthoredConstraintsMismatch,
    #[error("constraint plan is malformed: {0}")]
    Invalid(String),
    #[error("constraint plan encoding failed: {0}")]
    Encoding(String),
}
