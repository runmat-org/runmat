use runmat_execution::Digest;
use thiserror::Error;

use crate::{
    assembly::dofs::StructuralDofKind,
    constraints::{AffineConstraintReduction, ResolvedConstraintMatrix},
    contracts::StructuralInitialStateEvidence,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct StructuralInitialConstraintView<'a> {
    pub displacement_constraints: Option<&'a ResolvedConstraintMatrix>,
    pub velocity_constraints: Option<&'a ResolvedConstraintMatrix>,
    pub affine_reduction: Option<&'a AffineConstraintReduction>,
    pub direct_constrained_dofs: &'a [bool],
    pub direct_prescribed_displacement: &'a [f64],
}

impl StructuralInitialConstraintView<'_> {
    #[cfg(test)]
    pub(crate) const fn unconstrained() -> Self {
        Self {
            displacement_constraints: None,
            velocity_constraints: None,
            affine_reduction: None,
            direct_constrained_dofs: &[],
            direct_prescribed_displacement: &[],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StructuralInitialStateRealizationBasis {
    CanonicalTopology,
    OperatorBoundZeroDefault {
        dof_count: usize,
        operator_identity: Digest,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RealizedStructuralInitialState {
    pub displacement: Vec<f64>,
    pub velocity: Vec<f64>,
    pub evidence: StructuralInitialStateEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum StructuralInitialStateError {
    #[error(transparent)]
    Storage(crate::topology::storage::TopologyStorageError),
    #[error(
        "structural initial-state realization cannot resolve procedure step `{procedure_step_id}`"
    )]
    MissingStep { procedure_step_id: String },
    #[error("procedure step id `{procedure_step_id}` is not unique")]
    DuplicateStep { procedure_step_id: String },
    #[error("structural initial-state realization requires a solid mesh or explicit structural topology")]
    MissingDiscretization,
    #[error("operator-bound initial-state DOF count {supplied_dof_count} does not match the canonical layout DOF count {layout_dof_count}")]
    OperatorBoundDofCountMismatch {
        supplied_dof_count: usize,
        layout_dof_count: usize,
    },
    #[error("active initial condition `{initial_condition_id}` requires canonical structural topology and cannot use an operator-bound zero-default state")]
    OperatorBoundInitialConditionRequiresTopology { initial_condition_id: String },
    #[error("the admitted operator-bound initial-state envelope requires an all-zero constraint-compatible state")]
    OperatorBoundZeroStateConstraintConflict,
    #[error("structural initial-state topology is invalid: {detail}")]
    InvalidTopology { detail: String },
    #[error("structural DOF layout has {layout_node_count} nodes but canonical topology has {topology_node_count}")]
    LayoutTopologyMismatch {
        layout_node_count: usize,
        topology_node_count: usize,
    },
    #[error("active initial condition `{initial_condition_id}` is not a structural displacement, velocity, or angular velocity")]
    UnsupportedInitialCondition { initial_condition_id: String },
    #[error("structural initial condition `{initial_condition_id}` uses unsupported frame `{frame}`; only the global frame is admitted")]
    UnsupportedFrame {
        initial_condition_id: String,
        frame: String,
    },
    #[error("structural initial condition `{initial_condition_id}` must provide a vector3 value")]
    InvalidValueShape { initial_condition_id: String },
    #[error("structural initial condition `{initial_condition_id}` references unmaterialized field artifact `{artifact_id}`")]
    UnmaterializedFieldArtifact {
        initial_condition_id: String,
        artifact_id: String,
    },
    #[error("structural initial condition `{initial_condition_id}` has invalid field artifact: {detail}")]
    InvalidFieldArtifact {
        initial_condition_id: String,
        detail: String,
    },
    #[error("structural initial condition `{initial_condition_id}` references missing signal `{signal_id}`")]
    MissingSignal {
        initial_condition_id: String,
        signal_id: String,
    },
    #[error("structural initial condition `{initial_condition_id}` references a signal with an incompatible physical quantity")]
    IncompatibleSignal { initial_condition_id: String },
    #[error("structural initial condition `{initial_condition_id}` signal uses unsupported frame `{frame}`; only the global frame is admitted")]
    UnsupportedSignalFrame {
        initial_condition_id: String,
        frame: String,
    },
    #[error("structural initial condition `{initial_condition_id}` cannot use a frequency-domain signal")]
    UnsupportedSignalDomain { initial_condition_id: String },
    #[error(
        "structural initial condition `{initial_condition_id}` cannot sample its signal: {detail}"
    )]
    SignalSample {
        initial_condition_id: String,
        detail: String,
    },
    #[error("structural initial condition `{initial_condition_id}` realizes a complex or non-finite value")]
    InvalidRealizedValue { initial_condition_id: String },
    #[error("structural initial condition `{initial_condition_id}` target `{selector}` is invalid: {detail}")]
    InvalidTarget {
        initial_condition_id: String,
        selector: String,
        detail: String,
    },
    #[error("structural initial condition `{initial_condition_id}` targets node {node_index}, which has no {dof:?} DOF")]
    MissingDof {
        initial_condition_id: String,
        node_index: usize,
        dof: StructuralDofKind,
    },
    #[error("structural initial conditions `{first_initial_condition_id}` and `{second_initial_condition_id}` conflict at DOF row {row}")]
    ConflictingValue {
        first_initial_condition_id: String,
        second_initial_condition_id: String,
        row: usize,
    },
    #[error("structural initial constraint inputs are invalid: {detail}")]
    InvalidConstraintInputs { detail: String },
    #[error("structural initial condition `{initial_condition_id}` conflicts with prescribed displacement at DOF row {row}")]
    PrescribedDisplacementConflict {
        initial_condition_id: String,
        row: usize,
    },
    #[error("structural initial condition `{initial_condition_id}` conflicts with prescribed velocity at DOF row {row}")]
    PrescribedVelocityConflict {
        initial_condition_id: String,
        row: usize,
    },
    #[error("structural initial displacement violates constraint `{constraint_id}` at equation row {equation_row}")]
    DisplacementConstraintViolation {
        constraint_id: String,
        equation_row: usize,
        authored_initial_condition_ids: Vec<String>,
    },
    #[error("structural initial velocity violates constraint `{constraint_id}` at equation row {equation_row}")]
    VelocityConstraintViolation {
        constraint_id: String,
        equation_row: usize,
        authored_initial_condition_ids: Vec<String>,
    },
}
