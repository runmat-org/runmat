use runmat_analysis_core::ObservationId;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::results::observation::{
    ObservationHistoryCheckpointIdentity, ResolvedObservationPlanIdentity,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity,
    StateIdentity, StepIdentity, StructuralDampingOperatorIdentity, StructuralDofLayoutIdentity,
    StructuralExcitationSetIdentity, StructuralFormulationIdentity, StructuralKinematicsIdentity,
    StructuralMassOperatorIdentity, StructuralStiffnessOperatorIdentity,
    StructuralTransientCheckpointIdentity, StructuralTransientCheckpointScopeIdentity,
    StructuralTransientContinuationApproach, StructuralTransientIntegrator,
    StructuralTransientPrecision, StructuralTransientSchedule,
    StructuralTransientSolverOptionsIdentity, StudyIdentity,
};
use crate::{
    execution::{ModalProjectionEvidence, ModalTransientCoordinateState},
    pipeline::modal_basis::ModalBasisIdentity,
};

const TRANSIENT_CHECKPOINT_PREFIX: &[u8] = b"runmat-structural-transient-checkpoint/v1\0";
const TRANSIENT_SCOPE_PREFIX: &[u8] = b"runmat-structural-transient-checkpoint-scope/v1\0";
const TRANSIENT_CHECKPOINT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));
const TRANSIENT_SCOPE_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralTransientCoordinateSpace {
    Direct,
    ModalReduced {
        basis_state: StateIdentity,
        basis: ModalBasisIdentity,
        projection: ModalProjectionEvidence,
        coordinate_count: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralTransientNumericalState {
    Direct,
    ModalReduced {
        basis_state: StateIdentity,
        coordinates: ModalTransientCoordinateState,
    },
}

impl StructuralTransientNumericalState {
    fn validate_against(
        &self,
        coordinate_space: &StructuralTransientCoordinateSpace,
        physical_state: &StructuralTransientContinuationState,
    ) -> AnalysisExecutionResult<()> {
        match (self, coordinate_space) {
            (Self::Direct, StructuralTransientCoordinateSpace::Direct) => Ok(()),
            (
                Self::ModalReduced {
                    basis_state,
                    coordinates,
                },
                StructuralTransientCoordinateSpace::ModalReduced {
                    basis_state: expected_basis,
                    coordinate_count,
                    ..
                },
            ) if basis_state == expected_basis => {
                if coordinates.displacement.canonical_values().len() != *coordinate_count
                    || coordinates.velocity.canonical_values().len() != *coordinate_count
                    || coordinates.acceleration.canonical_values().len() != *coordinate_count
                    || coordinates
                        .displacement
                        .canonical_values()
                        .iter()
                        .chain(coordinates.velocity.canonical_values())
                        .chain(coordinates.acceleration.canonical_values())
                        .any(|value| !value.is_finite())
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "modal transient checkpoint has invalid reduced coordinates".into(),
                    ));
                }
                if physical_state.displacement.is_empty() {
                    return Err(AnalysisExecutionError::Invalid(
                        "modal transient checkpoint is missing reconstructed physical state".into(),
                    ));
                }
                Ok(())
            }
            _ => Err(AnalysisExecutionError::Identity(
                "structural transient checkpoint numerical state differs from its coordinate space",
            )),
        }
    }
}

impl StructuralTransientCoordinateSpace {
    pub fn coordinate_count(&self, physical_dof_count: usize) -> usize {
        match self {
            Self::Direct => physical_dof_count,
            Self::ModalReduced {
                coordinate_count, ..
            } => *coordinate_count,
        }
    }

    fn validate(&self, physical_dof_count: usize) -> AnalysisExecutionResult<()> {
        match self {
            Self::Direct => Ok(()),
            Self::ModalReduced {
                basis,
                projection,
                coordinate_count,
                ..
            } => {
                projection
                    .validate_for(projection.operator_digest, physical_dof_count)
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                if *coordinate_count == 0
                    || *coordinate_count != projection.mode_count
                    || *coordinate_count > physical_dof_count
                    || *basis != projection.basis
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "modal transient coordinate space differs from its projection evidence"
                            .into(),
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientCheckpointScope {
    pub study: StudyIdentity,
    pub step: StepIdentity,
    pub execution_plan: ExecutionPlanIdentity,
    pub source_workload: AnalysisWorkloadIdentity,
    pub numerical_plan: NumericalPlanIdentity,
    pub discretization: DiscretizationIdentity,
    pub constraint_plan: ConstraintPlanIdentity,
    pub stiffness_operator: StructuralStiffnessOperatorIdentity,
    pub mass_operator: StructuralMassOperatorIdentity,
    pub damping_operator: StructuralDampingOperatorIdentity,
    pub excitations: StructuralExcitationSetIdentity,
    pub dof_layout: StructuralDofLayoutIdentity,
    pub formulation: StructuralFormulationIdentity,
    pub kinematics: StructuralKinematicsIdentity,
    pub dof_count: usize,
    pub coordinate_space: StructuralTransientCoordinateSpace,
    pub integrator: StructuralTransientIntegrator,
    pub precision: StructuralTransientPrecision,
    pub solver_options: StructuralTransientSolverOptionsIdentity,
    pub schedule: StructuralTransientSchedule,
}

impl StructuralTransientCheckpointScope {
    pub(super) fn validate(&self) -> AnalysisExecutionResult<()> {
        self.schedule.validate()?;
        if self.dof_count == 0 {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient checkpoint scope must contain physical DOFs".into(),
            ));
        }
        self.coordinate_space.validate(self.dof_count)?;
        Ok(())
    }

    pub fn identity(&self) -> AnalysisExecutionResult<StructuralTransientCheckpointScopeIdentity> {
        self.validate()?;
        StructuralTransientCheckpointScopeIdentity::derive(
            "structural-transient-checkpoint-scope",
            self,
        )
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            TRANSIENT_SCOPE_PREFIX,
            "structural-transient-checkpoint-scope",
            self,
            TRANSIENT_SCOPE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let scope: Self = decode_contract(
            TRANSIENT_SCOPE_PREFIX,
            "structural-transient-checkpoint-scope",
            bytes,
            TRANSIENT_SCOPE_LIMITS,
        )?;
        scope.validate()?;
        Ok(scope)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientContinuationPoint {
    pub schedule_ordinal: u64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub time_s: runmat_types::physical::Time,
    pub approach: StructuralTransientContinuationApproach,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientContinuationState {
    pub displacement: Vec<f64>,
    pub velocity: Vec<f64>,
    pub acceleration: Vec<f64>,
}

impl StructuralTransientContinuationState {
    pub(super) fn validate(&self, dof_count: usize) -> AnalysisExecutionResult<()> {
        if self.displacement.len() != dof_count
            || self.velocity.len() != dof_count
            || self.acceleration.len() != dof_count
            || self
                .displacement
                .iter()
                .chain(&self.velocity)
                .chain(&self.acceleration)
                .any(|value| !value.is_finite())
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient state has invalid shape or values".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientBalanceAccumulator {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub initial_mechanical_energy_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub applied_load_work_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub support_work_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub damping_dissipation_j: runmat_types::physical::Energy,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientSolverCounters {
    pub host_sync_count: u64,
    pub device_apply_k_count: u64,
    pub device_apply_k_attempt_count: u64,
    pub dense_factorization_count: u64,
    pub dense_solve_count: u64,
}

impl StructuralTransientSolverCounters {
    pub(super) fn application_counts_are_valid(&self) -> bool {
        self.device_apply_k_count <= self.device_apply_k_attempt_count
    }

    pub(super) fn direct_counts_are_valid(&self) -> bool {
        self.dense_factorization_count == 0 && self.dense_solve_count == 0
    }

    pub(super) fn modal_dense_counts_are_valid(&self) -> bool {
        self.dense_factorization_count > 0
            && self.dense_factorization_count == self.dense_solve_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientObservationCursor {
    pub plan: ResolvedObservationPlanIdentity,
    pub observation_id: ObservationId,
    pub last_consumed_schedule_ordinal: u64,
    pub produced_sample_count: u64,
    pub history_checkpoint: Option<ObservationHistoryCheckpointIdentity>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralTransientCheckpointWire")]
pub struct StructuralTransientCheckpoint {
    pub schema_version: u16,
    pub identity: StructuralTransientCheckpointIdentity,
    pub scope: StructuralTransientCheckpointScope,
    pub predecessor: Option<StructuralTransientCheckpointIdentity>,
    pub block_ordinal: u64,
    pub continuation: StructuralTransientContinuationPoint,
    pub state: StructuralTransientContinuationState,
    pub numerical_state: StructuralTransientNumericalState,
    pub balance: StructuralTransientBalanceAccumulator,
    pub counters: StructuralTransientSolverCounters,
    pub observation_cursors: Vec<StructuralTransientObservationCursor>,
}

pub struct StructuralTransientCheckpointInput {
    pub scope: StructuralTransientCheckpointScope,
    pub block_ordinal: u64,
    pub continuation: StructuralTransientContinuationPoint,
    pub state: StructuralTransientContinuationState,
    pub numerical_state: StructuralTransientNumericalState,
    pub balance: StructuralTransientBalanceAccumulator,
    pub counters: StructuralTransientSolverCounters,
    pub observation_cursors: Vec<StructuralTransientObservationCursor>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientCheckpointWire {
    schema_version: u16,
    identity: StructuralTransientCheckpointIdentity,
    scope: StructuralTransientCheckpointScope,
    predecessor: Option<StructuralTransientCheckpointIdentity>,
    block_ordinal: u64,
    continuation: StructuralTransientContinuationPoint,
    state: StructuralTransientContinuationState,
    numerical_state: StructuralTransientNumericalState,
    balance: StructuralTransientBalanceAccumulator,
    counters: StructuralTransientSolverCounters,
    observation_cursors: Vec<StructuralTransientObservationCursor>,
}

impl TryFrom<StructuralTransientCheckpointWire> for StructuralTransientCheckpoint {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralTransientCheckpointWire) -> Result<Self, Self::Error> {
        let checkpoint = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            predecessor: wire.predecessor,
            block_ordinal: wire.block_ordinal,
            continuation: wire.continuation,
            state: wire.state,
            numerical_state: wire.numerical_state,
            balance: wire.balance,
            counters: wire.counters,
            observation_cursors: wire.observation_cursors,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }
}

impl StructuralTransientCheckpoint {
    pub fn new(
        input: StructuralTransientCheckpointInput,
        predecessor: Option<&Self>,
    ) -> AnalysisExecutionResult<Self> {
        let predecessor_identity = predecessor.map(|value| value.identity);
        let identity = checkpoint_identity(CheckpointIdentityInput {
            schema_version: 1,
            scope: &input.scope,
            predecessor: predecessor_identity,
            block_ordinal: input.block_ordinal,
            continuation: &input.continuation,
            state: &input.state,
            numerical_state: &input.numerical_state,
            balance: &input.balance,
            counters: &input.counters,
            observation_cursors: &input.observation_cursors,
        })?;
        let checkpoint = Self {
            schema_version: 1,
            identity,
            scope: input.scope,
            predecessor: predecessor_identity,
            block_ordinal: input.block_ordinal,
            continuation: input.continuation,
            state: input.state,
            numerical_state: input.numerical_state,
            balance: input.balance,
            counters: input.counters,
            observation_cursors: input.observation_cursors,
        };
        checkpoint.validate()?;
        if let Some(predecessor) = predecessor {
            checkpoint.validate_against_predecessor(predecessor)?;
        }
        Ok(checkpoint)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        if self.schema_version != 1
            || (self.block_ordinal == 0) != self.predecessor.is_none()
            || self.continuation.schedule_ordinal == 0
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient checkpoint has an invalid schema, lineage, or accepted-boundary cursor"
                    .into(),
            ));
        }
        let ordinal = usize::try_from(self.continuation.schedule_ordinal).map_err(|_| {
            AnalysisExecutionError::Invalid(
                "structural transient continuation ordinal exceeds the platform index range".into(),
            )
        })?;
        let point = self.scope.schedule.points.get(ordinal).ok_or_else(|| {
            AnalysisExecutionError::Invalid(
                "structural transient continuation lies outside its exact schedule".into(),
            )
        })?;
        if point.time_s.canonical_value().to_bits()
            != self.continuation.time_s.canonical_value().to_bits()
            || point.continuation != self.continuation.approach
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient continuation differs from its exact schedule",
            ));
        }
        self.state.validate(self.scope.dof_count)?;
        self.numerical_state
            .validate_against(&self.scope.coordinate_space, &self.state)?;
        let balance = &self.balance;
        if [
            balance.initial_mechanical_energy_j.canonical_value(),
            balance.applied_load_work_j.canonical_value(),
            balance.support_work_j.canonical_value(),
            balance.damping_dissipation_j.canonical_value(),
        ]
        .into_iter()
        .any(|value| !value.is_finite())
            || balance.initial_mechanical_energy_j.canonical_value() < 0.0
            || balance.damping_dissipation_j.canonical_value() < 0.0
            || !self.counters.application_counts_are_valid()
            || match &self.scope.coordinate_space {
                StructuralTransientCoordinateSpace::Direct => {
                    !self.counters.direct_counts_are_valid()
                }
                StructuralTransientCoordinateSpace::ModalReduced { .. } => {
                    !self.counters.modal_dense_counts_are_valid()
                }
            }
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient checkpoint balance or counters are invalid".into(),
            ));
        }
        for (index, cursor) in self.observation_cursors.iter().enumerate() {
            if cursor.last_consumed_schedule_ordinal > self.continuation.schedule_ordinal
                || (cursor.produced_sample_count == 0) != cursor.history_checkpoint.is_none()
                || index > 0
                    && (
                        &self.observation_cursors[index - 1].plan,
                        &self.observation_cursors[index - 1].observation_id,
                    ) >= (&cursor.plan, &cursor.observation_id)
            {
                return Err(AnalysisExecutionError::Invalid(
                    "structural transient observation cursors are invalid or noncanonical".into(),
                ));
            }
        }
        if self.identity
            != checkpoint_identity(CheckpointIdentityInput {
                schema_version: self.schema_version,
                scope: &self.scope,
                predecessor: self.predecessor,
                block_ordinal: self.block_ordinal,
                continuation: &self.continuation,
                state: &self.state,
                numerical_state: &self.numerical_state,
                balance: &self.balance,
                counters: &self.counters,
                observation_cursors: &self.observation_cursors,
            })?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient checkpoint identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn validate_against_predecessor(&self, predecessor: &Self) -> AnalysisExecutionResult<()> {
        self.validate()?;
        predecessor.validate()?;
        let expected_block_ordinal =
            predecessor
                .block_ordinal
                .checked_add(1)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient checkpoint block ordinal overflow",
                ))?;
        if self.scope != predecessor.scope
            || self.predecessor != Some(predecessor.identity)
            || self.block_ordinal != expected_block_ordinal
            || self.continuation.schedule_ordinal <= predecessor.continuation.schedule_ordinal
            || self
                .balance
                .initial_mechanical_energy_j
                .canonical_value()
                .to_bits()
                != predecessor
                    .balance
                    .initial_mechanical_energy_j
                    .canonical_value()
                    .to_bits()
            || self.balance.damping_dissipation_j.canonical_value()
                < predecessor.balance.damping_dissipation_j.canonical_value()
            || self.counters.host_sync_count < predecessor.counters.host_sync_count
            || self.counters.device_apply_k_count < predecessor.counters.device_apply_k_count
            || self.counters.device_apply_k_attempt_count
                < predecessor.counters.device_apply_k_attempt_count
            || self.counters.dense_factorization_count
                < predecessor.counters.dense_factorization_count
            || self.counters.dense_solve_count < predecessor.counters.dense_solve_count
            || !observation_cursors_continue(
                &predecessor.observation_cursors,
                &self.observation_cursors,
            )
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient checkpoint does not continue its exact predecessor",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            TRANSIENT_CHECKPOINT_PREFIX,
            "structural-transient-checkpoint",
            self,
            TRANSIENT_CHECKPOINT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            TRANSIENT_CHECKPOINT_PREFIX,
            "structural-transient-checkpoint",
            bytes,
            TRANSIENT_CHECKPOINT_LIMITS,
        )?)
    }
}

#[derive(Serialize)]
struct CheckpointIdentityInput<'a> {
    schema_version: u16,
    scope: &'a StructuralTransientCheckpointScope,
    predecessor: Option<StructuralTransientCheckpointIdentity>,
    block_ordinal: u64,
    continuation: &'a StructuralTransientContinuationPoint,
    state: &'a StructuralTransientContinuationState,
    numerical_state: &'a StructuralTransientNumericalState,
    balance: &'a StructuralTransientBalanceAccumulator,
    counters: &'a StructuralTransientSolverCounters,
    observation_cursors: &'a [StructuralTransientObservationCursor],
}

fn observation_cursors_continue(
    previous: &[StructuralTransientObservationCursor],
    current: &[StructuralTransientObservationCursor],
) -> bool {
    previous.len() == current.len()
        && previous.iter().zip(current).all(|(previous, current)| {
            previous.plan == current.plan
                && previous.observation_id == current.observation_id
                && previous.last_consumed_schedule_ordinal <= current.last_consumed_schedule_ordinal
                && previous.produced_sample_count <= current.produced_sample_count
                && (previous.produced_sample_count != current.produced_sample_count
                    || previous.history_checkpoint == current.history_checkpoint)
        })
}

fn checkpoint_identity(
    input: CheckpointIdentityInput<'_>,
) -> AnalysisExecutionResult<StructuralTransientCheckpointIdentity> {
    StructuralTransientCheckpointIdentity::derive_content("structural-transient-checkpoint", &input)
}

#[cfg(test)]
#[path = "transient_checkpoint_tests.rs"]
pub(in crate::execution) mod tests;
