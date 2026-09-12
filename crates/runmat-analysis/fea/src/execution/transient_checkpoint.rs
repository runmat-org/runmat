use runmat_analysis_core::ObservationId;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::results::observation::{
    ObservationHistoryCheckpointIdentity, ResolvedObservationPlanIdentity,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity,
    StepIdentity, StructuralDampingOperatorIdentity, StructuralDofLayoutIdentity,
    StructuralExcitationSetIdentity, StructuralFormulationIdentity, StructuralKinematicsIdentity,
    StructuralMassOperatorIdentity, StructuralStiffnessOperatorIdentity,
    StructuralTransientCheckpointIdentity, StructuralTransientContinuationApproach,
    StructuralTransientIntegrator, StructuralTransientPrecision, StructuralTransientSchedule,
    StructuralTransientSolverOptionsIdentity, StudyIdentity,
};

const TRANSIENT_CHECKPOINT_PREFIX: &[u8] = b"runmat-structural-transient-checkpoint/v1\0";
const TRANSIENT_CHECKPOINT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

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
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientContinuationPoint {
    pub schedule_ordinal: u64,
    pub time_s: f64,
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
    pub initial_mechanical_energy_j: f64,
    pub applied_load_work_j: f64,
    pub support_work_j: f64,
    pub damping_dissipation_j: f64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientSolverCounters {
    pub host_sync_count: u64,
    pub device_apply_k_count: u64,
    pub device_apply_k_attempt_count: u64,
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
    pub balance: StructuralTransientBalanceAccumulator,
    pub counters: StructuralTransientSolverCounters,
    pub observation_cursors: Vec<StructuralTransientObservationCursor>,
}

pub struct StructuralTransientCheckpointInput {
    pub scope: StructuralTransientCheckpointScope,
    pub block_ordinal: u64,
    pub continuation: StructuralTransientContinuationPoint,
    pub state: StructuralTransientContinuationState,
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
        if point.time_s.to_bits() != self.continuation.time_s.to_bits()
            || point.continuation != self.continuation.approach
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient continuation differs from its exact schedule",
            ));
        }
        self.state.validate(self.scope.dof_count)?;
        let balance = &self.balance;
        if [
            balance.initial_mechanical_energy_j,
            balance.applied_load_work_j,
            balance.support_work_j,
            balance.damping_dissipation_j,
        ]
        .into_iter()
        .any(|value| !value.is_finite())
            || balance.initial_mechanical_energy_j < 0.0
            || balance.damping_dissipation_j < 0.0
            || self.counters.device_apply_k_count > self.counters.device_apply_k_attempt_count
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
            || self.balance.initial_mechanical_energy_j.to_bits()
                != predecessor.balance.initial_mechanical_energy_j.to_bits()
            || self.balance.damping_dissipation_j < predecessor.balance.damping_dissipation_j
            || self.counters.host_sync_count < predecessor.counters.host_sync_count
            || self.counters.device_apply_k_count < predecessor.counters.device_apply_k_count
            || self.counters.device_apply_k_attempt_count
                < predecessor.counters.device_apply_k_attempt_count
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
    StructuralTransientCheckpointIdentity::derive("structural-transient-checkpoint", &input)
}

#[cfg(test)]
pub(in crate::execution) mod tests {
    use crate::execution::{
        OperatorIdentity, SolverExcitationIdentity, StructuralTransientBlockRequest,
        StructuralTransientScheduleArrival, StructuralTransientSchedulePoint,
    };

    use super::*;

    fn schedule() -> StructuralTransientSchedule {
        StructuralTransientSchedule::new(vec![
            StructuralTransientSchedulePoint {
                time_s: 0.0,
                arrival: StructuralTransientScheduleArrival::Initial,
                continuation: StructuralTransientContinuationApproach::At,
            },
            StructuralTransientSchedulePoint {
                time_s: 0.5,
                arrival: StructuralTransientScheduleArrival::LeftLimit,
                continuation: StructuralTransientContinuationApproach::RightLimit,
            },
            StructuralTransientSchedulePoint {
                time_s: 1.0,
                arrival: StructuralTransientScheduleArrival::Continuous,
                continuation: StructuralTransientContinuationApproach::At,
            },
        ])
        .unwrap()
    }

    pub(in crate::execution) fn scope() -> StructuralTransientCheckpointScope {
        StructuralTransientCheckpointScope {
            study: StudyIdentity::derive("test-study", &1_u8).unwrap(),
            step: StepIdentity::derive("test-step", &2_u8).unwrap(),
            execution_plan: ExecutionPlanIdentity::derive("test-execution", &3_u8).unwrap(),
            source_workload: AnalysisWorkloadIdentity::derive("test-workload", &4_u8).unwrap(),
            numerical_plan: NumericalPlanIdentity::derive("test-numerical", &5_u8).unwrap(),
            discretization: DiscretizationIdentity::derive("test-discretization", &6_u8).unwrap(),
            constraint_plan: ConstraintPlanIdentity::derive("test-constraints", &7_u8).unwrap(),
            stiffness_operator: StructuralStiffnessOperatorIdentity::new(
                OperatorIdentity::derive("test-stiffness", &8_u8).unwrap(),
            )
            .unwrap(),
            mass_operator: StructuralMassOperatorIdentity::new(
                OperatorIdentity::derive("test-mass", &9_u8).unwrap(),
            )
            .unwrap(),
            damping_operator: StructuralDampingOperatorIdentity::new(
                OperatorIdentity::derive("test-damping", &10_u8).unwrap(),
            )
            .unwrap(),
            excitations: StructuralExcitationSetIdentity::new(vec![
                SolverExcitationIdentity::derive("test-excitation", &11_u8).unwrap(),
            ])
            .unwrap(),
            dof_layout: StructuralDofLayoutIdentity::new(
                DiscretizationIdentity::derive("test-discretization", &6_u8).unwrap(),
                runmat_execution::Digest::sha256(b"test-dof-layout"),
            )
            .unwrap(),
            formulation: StructuralFormulationIdentity::new(
                DiscretizationIdentity::derive("test-discretization", &6_u8).unwrap(),
                runmat_execution::Digest::sha256(b"test-formulation"),
            )
            .unwrap(),
            kinematics: StructuralKinematicsIdentity::new(
                ConstraintPlanIdentity::derive("test-constraints", &7_u8).unwrap(),
                StructuralExcitationSetIdentity::new(vec![SolverExcitationIdentity::derive(
                    "test-excitation",
                    &11_u8,
                )
                .unwrap()])
                .unwrap(),
                runmat_execution::Digest::sha256(b"test-kinematics"),
            )
            .unwrap(),
            dof_count: 2,
            integrator: StructuralTransientIntegrator::NewmarkAverageAcceleration,
            precision: StructuralTransientPrecision::Real64,
            solver_options: StructuralTransientSolverOptionsIdentity::new(
                runmat_execution::Digest::sha256(b"test-solver-options"),
            )
            .unwrap(),
            schedule: schedule(),
        }
    }

    pub(in crate::execution) fn checkpoint(
        predecessor: Option<&StructuralTransientCheckpoint>,
        ordinal: u64,
    ) -> StructuralTransientCheckpoint {
        StructuralTransientCheckpoint::new(
            StructuralTransientCheckpointInput {
                scope: scope(),
                block_ordinal: predecessor.map_or(0, |value| value.block_ordinal + 1),
                continuation: StructuralTransientContinuationPoint {
                    schedule_ordinal: ordinal,
                    time_s: if ordinal == 1 { 0.5 } else { 1.0 },
                    approach: if ordinal == 1 {
                        StructuralTransientContinuationApproach::RightLimit
                    } else {
                        StructuralTransientContinuationApproach::At
                    },
                },
                state: StructuralTransientContinuationState {
                    displacement: vec![ordinal as f64, 0.0],
                    velocity: vec![0.5, 0.0],
                    acceleration: vec![1.0, 0.0],
                },
                balance: StructuralTransientBalanceAccumulator {
                    initial_mechanical_energy_j: 1.0,
                    applied_load_work_j: ordinal as f64,
                    support_work_j: -0.25 * ordinal as f64,
                    damping_dissipation_j: 0.1 * ordinal as f64,
                },
                counters: StructuralTransientSolverCounters {
                    host_sync_count: ordinal,
                    device_apply_k_count: ordinal,
                    device_apply_k_attempt_count: ordinal,
                },
                observation_cursors: Vec::new(),
            },
            predecessor,
        )
        .unwrap()
    }

    #[test]
    fn schedule_binds_arrival_to_the_exact_continuation_side() {
        let canonical = schedule();
        assert_eq!(
            StructuralTransientSchedule::from_canonical_bytes(
                &canonical.canonical_bytes().unwrap()
            )
            .unwrap(),
            canonical
        );
        let mut invalid = schedule();
        invalid.points[1].continuation = StructuralTransientContinuationApproach::At;
        assert!(invalid.validate().is_err());
        invalid = schedule();
        invalid.points.swap(1, 2);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn checkpoint_round_trips_and_preserves_right_limit_continuation() {
        let first = checkpoint(None, 1);
        let decoded =
            StructuralTransientCheckpoint::from_canonical_bytes(&first.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(decoded, first);
        assert_eq!(
            decoded.continuation.approach,
            StructuralTransientContinuationApproach::RightLimit
        );

        let second = checkpoint(Some(&first), 2);
        second.validate_against_predecessor(&first).unwrap();
        assert_eq!(second.predecessor, Some(first.identity));
    }

    #[test]
    fn checkpoint_rejects_state_schedule_scope_and_lineage_substitution() {
        let first = checkpoint(None, 1);
        let mut invalid = first.clone();
        invalid.state.velocity.pop();
        assert!(invalid.validate().is_err());

        let mut invalid = first.clone();
        invalid.continuation.approach = StructuralTransientContinuationApproach::At;
        assert!(invalid.validate().is_err());

        let mut invalid = first.clone();
        invalid.scope.mass_operator = StructuralMassOperatorIdentity::new(
            OperatorIdentity::derive("other-mass", &1_u8).unwrap(),
        )
        .unwrap();
        assert!(invalid.validate().is_err());

        let second = checkpoint(Some(&first), 2);
        let mut other_scope = scope();
        other_scope.kinematics = StructuralKinematicsIdentity::new(
            other_scope.constraint_plan,
            other_scope.excitations,
            runmat_execution::Digest::sha256(b"other-kinematics"),
        )
        .unwrap();
        let other = StructuralTransientCheckpoint::new(
            StructuralTransientCheckpointInput {
                scope: other_scope,
                block_ordinal: 0,
                continuation: first.continuation.clone(),
                state: first.state.clone(),
                balance: first.balance.clone(),
                counters: first.counters.clone(),
                observation_cursors: first.observation_cursors.clone(),
            },
            None,
        )
        .unwrap();
        assert!(second.validate_against_predecessor(&other).is_err());
    }

    #[test]
    fn block_request_is_exactly_bounded_by_its_predecessor_and_schedule() {
        let first = StructuralTransientBlockRequest::new(scope(), None, 1).unwrap();
        assert_eq!(first.start_schedule_ordinal, 0);
        assert_eq!(first.terminal_schedule_ordinal, 1);
        assert_eq!(
            StructuralTransientBlockRequest::from_canonical_bytes(
                &first.canonical_bytes().unwrap()
            )
            .unwrap(),
            first
        );

        let checkpoint = checkpoint(None, 1);
        let second = StructuralTransientBlockRequest::new(scope(), Some(&checkpoint), 2).unwrap();
        assert_eq!(second.start_schedule_ordinal, 1);
        assert_eq!(second.block_ordinal, 1);

        assert!(StructuralTransientBlockRequest::new(scope(), Some(&checkpoint), 1).is_err());
        assert!(StructuralTransientBlockRequest::new(scope(), Some(&checkpoint), 3).is_err());
    }
}
