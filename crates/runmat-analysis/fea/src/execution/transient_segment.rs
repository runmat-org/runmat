use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::{
    contracts::{SolverMethod, SolverPreconditioner, StructuralDynamicBalanceSnapshot},
    solve::{accelerator::AcceleratorSolveEvidence, backend::kind::LinearAlgebraBackendKind},
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, StructuralTransientBlockRequest,
    StructuralTransientBlockRequestIdentity, StructuralTransientCheckpoint,
    StructuralTransientCheckpointScope, StructuralTransientCheckpointScopeIdentity,
    StructuralTransientContinuationApproach, StructuralTransientContinuationState,
    StructuralTransientPhysicalSampleIdentity, StructuralTransientSegmentIdentity,
    StructuralTransientSegmentSampleIdentity, StructuralTransientSolverCounters,
};

const TRANSIENT_SEGMENT_PREFIX: &[u8] = b"runmat-structural-transient-segment/v1\0";
const TRANSIENT_SEGMENT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralTransientSampleStep {
    Initial,
    Accepted { time_step_s: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralTransientPhysicalSampleWire"
)]
pub struct StructuralTransientPhysicalSample {
    pub schema_version: u16,
    pub identity: StructuralTransientPhysicalSampleIdentity,
    pub scope: StructuralTransientCheckpointScopeIdentity,
    pub schedule_ordinal: u64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub time_s: runmat_types::physical::Time,
    pub step: StructuralTransientSampleStep,
    pub state: StructuralTransientContinuationState,
    pub projected_residual_norm: f64,
    pub reaction: Vec<f64>,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub strain_energy_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub kinetic_energy_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub applied_load_work_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub support_work_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub damping_dissipation_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub energy_balance_residual_j: runmat_types::physical::Energy,
    pub relative_energy_balance_error: f64,
    pub dynamic_balance: StructuralDynamicBalanceSnapshot,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientPhysicalSampleWire {
    schema_version: u16,
    identity: StructuralTransientPhysicalSampleIdentity,
    scope: StructuralTransientCheckpointScopeIdentity,
    schedule_ordinal: u64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    time_s: runmat_types::physical::Time,
    step: StructuralTransientSampleStep,
    state: StructuralTransientContinuationState,
    projected_residual_norm: f64,
    reaction: Vec<f64>,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    strain_energy_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    kinetic_energy_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    applied_load_work_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    support_work_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    damping_dissipation_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    energy_balance_residual_j: runmat_types::physical::Energy,
    relative_energy_balance_error: f64,
    dynamic_balance: StructuralDynamicBalanceSnapshot,
}

impl TryFrom<StructuralTransientPhysicalSampleWire> for StructuralTransientPhysicalSample {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralTransientPhysicalSampleWire) -> Result<Self, Self::Error> {
        let sample = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            schedule_ordinal: wire.schedule_ordinal,
            time_s: runmat_types::physical::Time::new(wire.time_s.canonical_value()),
            step: wire.step,
            state: wire.state,
            projected_residual_norm: wire.projected_residual_norm,
            reaction: wire.reaction,
            strain_energy_j: runmat_types::physical::Energy::new(
                wire.strain_energy_j.canonical_value(),
            ),
            kinetic_energy_j: runmat_types::physical::Energy::new(
                wire.kinetic_energy_j.canonical_value(),
            ),
            applied_load_work_j: runmat_types::physical::Energy::new(
                wire.applied_load_work_j.canonical_value(),
            ),
            support_work_j: runmat_types::physical::Energy::new(
                wire.support_work_j.canonical_value(),
            ),
            damping_dissipation_j: runmat_types::physical::Energy::new(
                wire.damping_dissipation_j.canonical_value(),
            ),
            energy_balance_residual_j: runmat_types::physical::Energy::new(
                wire.energy_balance_residual_j.canonical_value(),
            ),
            relative_energy_balance_error: wire.relative_energy_balance_error,
            dynamic_balance: wire.dynamic_balance,
        };
        sample.validate_content()?;
        Ok(sample)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTransientPhysicalSampleInput {
    pub schedule_ordinal: u64,
    pub time_s: runmat_types::physical::Time,
    pub step: StructuralTransientSampleStep,
    pub state: StructuralTransientContinuationState,
    pub projected_residual_norm: f64,
    pub reaction: Vec<f64>,
    pub strain_energy_j: runmat_types::physical::Energy,
    pub kinetic_energy_j: runmat_types::physical::Energy,
    pub applied_load_work_j: runmat_types::physical::Energy,
    pub support_work_j: runmat_types::physical::Energy,
    pub damping_dissipation_j: runmat_types::physical::Energy,
    pub energy_balance_residual_j: runmat_types::physical::Energy,
    pub relative_energy_balance_error: f64,
    pub dynamic_balance: StructuralDynamicBalanceSnapshot,
}

impl StructuralTransientPhysicalSample {
    pub fn new(
        scope: &StructuralTransientCheckpointScope,
        input: StructuralTransientPhysicalSampleInput,
    ) -> AnalysisExecutionResult<Self> {
        scope.validate()?;
        let scope_identity = scope.identity()?;
        let identity = physical_sample_input_identity(1, scope_identity, &input)?;
        let sample = Self {
            schema_version: 1,
            identity,
            scope: scope_identity,
            schedule_ordinal: input.schedule_ordinal,
            time_s: runmat_types::physical::Time::new(input.time_s.canonical_value()),
            step: input.step,
            state: input.state,
            projected_residual_norm: input.projected_residual_norm,
            reaction: input.reaction,
            strain_energy_j: runmat_types::physical::Energy::new(
                input.strain_energy_j.canonical_value(),
            ),
            kinetic_energy_j: runmat_types::physical::Energy::new(
                input.kinetic_energy_j.canonical_value(),
            ),
            applied_load_work_j: runmat_types::physical::Energy::new(
                input.applied_load_work_j.canonical_value(),
            ),
            support_work_j: runmat_types::physical::Energy::new(
                input.support_work_j.canonical_value(),
            ),
            damping_dissipation_j: runmat_types::physical::Energy::new(
                input.damping_dissipation_j.canonical_value(),
            ),
            energy_balance_residual_j: runmat_types::physical::Energy::new(
                input.energy_balance_residual_j.canonical_value(),
            ),
            relative_energy_balance_error: input.relative_energy_balance_error,
            dynamic_balance: input.dynamic_balance,
        };
        sample.validate_against(scope)?;
        Ok(sample)
    }

    pub fn validate_against(
        &self,
        scope: &StructuralTransientCheckpointScope,
    ) -> AnalysisExecutionResult<()> {
        scope.validate()?;
        self.validate_content()?;
        let ordinal = usize::try_from(self.schedule_ordinal).map_err(|_| {
            AnalysisExecutionError::Invalid(
                "structural transient segment ordinal exceeds the platform index range".into(),
            )
        })?;
        let schedule_point = scope.schedule.points.get(ordinal).ok_or_else(|| {
            AnalysisExecutionError::Invalid(
                "structural transient segment sample lies outside its exact schedule".into(),
            )
        })?;
        if self.schema_version != 1
            || self.scope != scope.identity()?
            || self.time_s.canonical_value().to_bits()
                != schedule_point.time_s.canonical_value().to_bits()
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient physical sample differs from its exact scope or schedule",
            ));
        }
        match self.step {
            StructuralTransientSampleStep::Initial
                if self.schedule_ordinal == 0
                    && schedule_point.arrival
                        == super::StructuralTransientScheduleArrival::Initial => {}
            StructuralTransientSampleStep::Accepted { time_step_s }
                if self.schedule_ordinal > 0 =>
            {
                let previous = &scope.schedule.points[ordinal - 1];
                if time_step_s.to_bits()
                    != (schedule_point.time_s.canonical_value() - previous.time_s.canonical_value())
                        .to_bits()
                {
                    return Err(AnalysisExecutionError::Identity(
                        "structural transient physical sample time step differs from its schedule",
                    ));
                }
            }
            _ => {
                return Err(AnalysisExecutionError::Identity(
                    "structural transient physical sample step kind differs from its ordinal",
                ));
            }
        }
        if self.identity != physical_sample_identity(self)? {
            return Err(AnalysisExecutionError::Identity(
                "structural transient physical sample identity does not converge",
            ));
        }
        Ok(())
    }

    fn validate_content(&self) -> AnalysisExecutionResult<()> {
        self.state.validate(self.reaction.len())?;
        self.dynamic_balance
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let scalars = [
            self.projected_residual_norm,
            self.strain_energy_j.canonical_value(),
            self.kinetic_energy_j.canonical_value(),
            self.applied_load_work_j.canonical_value(),
            self.support_work_j.canonical_value(),
            self.damping_dissipation_j.canonical_value(),
            self.energy_balance_residual_j.canonical_value(),
            self.relative_energy_balance_error,
        ];
        if self.reaction.is_empty()
            || self.dynamic_balance.applied_load.len() != self.reaction.len()
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient physical sample vectors have the wrong shape".into(),
            ));
        }
        if self.reaction.iter().any(|value| !value.is_finite())
            || scalars.iter().any(|value| !value.is_finite())
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient physical sample contains non-finite values".into(),
            ));
        }
        if self.projected_residual_norm < 0.0
            || self.strain_energy_j.canonical_value() < 0.0
            || self.kinetic_energy_j.canonical_value() < 0.0
            || self.damping_dissipation_j.canonical_value() < 0.0
            || self.relative_energy_balance_error < 0.0
        {
            return Err(AnalysisExecutionError::Invalid(format!(
                "structural transient physical sample has a negative nonnegative quantity: residual={}, strain_energy_j={}, kinetic_energy_j={}, damping_dissipation_j={}, relative_energy_balance_error={}",
                self.projected_residual_norm,
                self.strain_energy_j.canonical_value(),
                self.kinetic_energy_j.canonical_value(),
                self.damping_dissipation_j.canonical_value(),
                self.relative_energy_balance_error,
            )));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_content()?;
        if self.schema_version != 1 || self.identity != physical_sample_identity(self)? {
            return Err(AnalysisExecutionError::Identity(
                "structural transient physical sample identity does not converge",
            ));
        }
        Ok(encode_contract(
            TRANSIENT_SEGMENT_PREFIX,
            "structural-transient-physical-sample",
            self,
            TRANSIENT_SEGMENT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let sample: Self = decode_contract(
            TRANSIENT_SEGMENT_PREFIX,
            "structural-transient-physical-sample",
            bytes,
            TRANSIENT_SEGMENT_LIMITS,
        )?;
        if sample.schema_version != 1 || sample.identity != physical_sample_identity(&sample)? {
            return Err(AnalysisExecutionError::Identity(
                "structural transient physical sample identity does not converge",
            ));
        }
        Ok(sample)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralTransientSegmentSampleWire")]
pub struct StructuralTransientSegmentSample {
    pub schema_version: u16,
    pub identity: StructuralTransientSegmentSampleIdentity,
    pub request: StructuralTransientBlockRequestIdentity,
    pub physical: StructuralTransientPhysicalSample,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientSegmentSampleWire {
    schema_version: u16,
    identity: StructuralTransientSegmentSampleIdentity,
    request: StructuralTransientBlockRequestIdentity,
    physical: StructuralTransientPhysicalSample,
}

impl TryFrom<StructuralTransientSegmentSampleWire> for StructuralTransientSegmentSample {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralTransientSegmentSampleWire) -> Result<Self, Self::Error> {
        let sample = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            request: wire.request,
            physical: wire.physical,
        };
        if sample.schema_version != 1 || sample.identity != segment_sample_identity(&sample)? {
            return Err(AnalysisExecutionError::Identity(
                "structural transient segment sample identity does not converge",
            ));
        }
        Ok(sample)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTransientSegmentSampleInput {
    pub schedule_ordinal: u64,
    pub time_s: runmat_types::physical::Time,
    pub accepted_time_step_s: runmat_types::physical::Time,
    pub state: StructuralTransientContinuationState,
    pub projected_residual_norm: f64,
    pub reaction: Vec<f64>,
    pub strain_energy_j: runmat_types::physical::Energy,
    pub kinetic_energy_j: runmat_types::physical::Energy,
    pub applied_load_work_j: runmat_types::physical::Energy,
    pub support_work_j: runmat_types::physical::Energy,
    pub damping_dissipation_j: runmat_types::physical::Energy,
    pub energy_balance_residual_j: runmat_types::physical::Energy,
    pub relative_energy_balance_error: f64,
    pub dynamic_balance: StructuralDynamicBalanceSnapshot,
}

impl StructuralTransientSegmentSample {
    pub fn new(
        request: &StructuralTransientBlockRequest,
        input: StructuralTransientSegmentSampleInput,
    ) -> AnalysisExecutionResult<Self> {
        let physical = StructuralTransientPhysicalSample::new(
            &request.scope,
            StructuralTransientPhysicalSampleInput {
                schedule_ordinal: input.schedule_ordinal,
                time_s: runmat_types::physical::Time::new(input.time_s.canonical_value()),
                step: StructuralTransientSampleStep::Accepted {
                    time_step_s: input.accepted_time_step_s.canonical_value(),
                },
                state: input.state,
                projected_residual_norm: input.projected_residual_norm,
                reaction: input.reaction,
                strain_energy_j: runmat_types::physical::Energy::new(
                    input.strain_energy_j.canonical_value(),
                ),
                kinetic_energy_j: runmat_types::physical::Energy::new(
                    input.kinetic_energy_j.canonical_value(),
                ),
                applied_load_work_j: runmat_types::physical::Energy::new(
                    input.applied_load_work_j.canonical_value(),
                ),
                support_work_j: runmat_types::physical::Energy::new(
                    input.support_work_j.canonical_value(),
                ),
                damping_dissipation_j: runmat_types::physical::Energy::new(
                    input.damping_dissipation_j.canonical_value(),
                ),
                energy_balance_residual_j: runmat_types::physical::Energy::new(
                    input.energy_balance_residual_j.canonical_value(),
                ),
                relative_energy_balance_error: input.relative_energy_balance_error,
                dynamic_balance: input.dynamic_balance,
            },
        )?;
        let sample = Self {
            schema_version: 1,
            identity: StructuralTransientSegmentSampleIdentity::derive_content(
                "structural-transient-segment-sample",
                &(1_u16, request.identity, &physical),
            )?,
            request: request.identity,
            physical,
        };
        sample.validate_against(request)?;
        Ok(sample)
    }

    pub fn validate_against(
        &self,
        request: &StructuralTransientBlockRequest,
    ) -> AnalysisExecutionResult<()> {
        request.validate()?;
        self.physical.validate_against(&request.scope)?;
        if self.schema_version != 1
            || self.request != request.identity
            || !matches!(
                self.physical.step,
                StructuralTransientSampleStep::Accepted { .. }
            )
            || self.physical.schedule_ordinal <= request.start_schedule_ordinal
            || self.physical.schedule_ordinal > request.terminal_schedule_ordinal
            || self.identity != segment_sample_identity(self)?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient segment sample differs from its exact request",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        if self.schema_version != 1 || self.identity != segment_sample_identity(self)? {
            return Err(AnalysisExecutionError::Identity(
                "structural transient segment sample identity does not converge",
            ));
        }
        Ok(encode_contract(
            TRANSIENT_SEGMENT_PREFIX,
            "structural-transient-segment-sample",
            self,
            TRANSIENT_SEGMENT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            TRANSIENT_SEGMENT_PREFIX,
            "structural-transient-segment-sample",
            bytes,
            TRANSIENT_SEGMENT_LIMITS,
        )?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientSegmentEntry {
    pub schedule_ordinal: u64,
    pub sample: StructuralTransientSegmentSampleIdentity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientBlockSolverEvidence {
    pub solver_method: SolverMethod,
    pub solver_backend: LinearAlgebraBackendKind,
    pub preconditioner: SolverPreconditioner,
    pub counters: StructuralTransientSolverCounters,
    pub accelerator_evidence: Vec<AcceleratorSolveEvidence>,
}

impl StructuralTransientBlockSolverEvidence {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if !matches!(
            self.solver_method,
            SolverMethod::NewmarkAverageAccelerationPcg
                | SolverMethod::StructuralModalTransientNewmarkDenseFactorization
        ) || !self.counters.application_counts_are_valid()
            || self
                .accelerator_evidence
                .iter()
                .any(|evidence| evidence.validate().is_err() || !evidence.satisfies_tolerance())
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient block solver evidence is invalid".into(),
            ));
        }
        match self.solver_method {
            SolverMethod::NewmarkAverageAccelerationPcg
                if !self.counters.direct_counts_are_valid() =>
            {
                return Err(AnalysisExecutionError::Invalid(
                    "direct structural transient evidence contains modal dense solves".into(),
                ));
            }
            SolverMethod::StructuralModalTransientNewmarkDenseFactorization
                if self.solver_backend != LinearAlgebraBackendKind::CpuReference
                    || self.preconditioner != SolverPreconditioner::None
                    || self.counters.host_sync_count != 0
                    || self.counters.device_apply_k_count != 0
                    || self.counters.device_apply_k_attempt_count != 0
                    || !self.counters.modal_dense_counts_are_valid()
                    || !self.accelerator_evidence.is_empty() =>
            {
                return Err(AnalysisExecutionError::Invalid(
                    "modal structural transient evidence does not describe CPU dense-direct execution"
                        .into(),
                ));
            }
            _ => {}
        }
        match self.solver_backend {
            LinearAlgebraBackendKind::CpuReference
                if self.counters.device_apply_k_count != 0
                    || self.counters.device_apply_k_attempt_count != 0
                    || !self.accelerator_evidence.is_empty() =>
            {
                Err(AnalysisExecutionError::Invalid(
                    "CPU structural transient block evidence contains accelerator execution".into(),
                ))
            }
            LinearAlgebraBackendKind::Accelerator if self.accelerator_evidence.is_empty() => {
                Err(AnalysisExecutionError::Invalid(
                    "accelerated structural transient block evidence is missing provider records"
                        .into(),
                ))
            }
            _ => Ok(()),
        }
    }

    pub fn validate_for(
        &self,
        coordinate_space: &super::StructuralTransientCoordinateSpace,
    ) -> AnalysisExecutionResult<()> {
        let method_matches = matches!(
            (coordinate_space, self.solver_method),
            (
                super::StructuralTransientCoordinateSpace::Direct,
                SolverMethod::NewmarkAverageAccelerationPcg
            ) | (
                super::StructuralTransientCoordinateSpace::ModalReduced { .. },
                SolverMethod::StructuralModalTransientNewmarkDenseFactorization
            )
        );
        self.validate()?;
        if !method_matches {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient block solver evidence is invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralTransientSegmentManifestWire"
)]
pub struct StructuralTransientSegmentManifest {
    pub schema_version: u16,
    pub identity: StructuralTransientSegmentIdentity,
    pub request: StructuralTransientBlockRequest,
    pub terminal_checkpoint: StructuralTransientCheckpoint,
    pub solver_evidence: StructuralTransientBlockSolverEvidence,
    pub initial_sample: Option<StructuralTransientPhysicalSampleIdentity>,
    pub samples: Vec<StructuralTransientSegmentEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientSegmentManifestWire {
    schema_version: u16,
    identity: StructuralTransientSegmentIdentity,
    request: StructuralTransientBlockRequest,
    terminal_checkpoint: StructuralTransientCheckpoint,
    solver_evidence: StructuralTransientBlockSolverEvidence,
    initial_sample: Option<StructuralTransientPhysicalSampleIdentity>,
    samples: Vec<StructuralTransientSegmentEntry>,
}

impl TryFrom<StructuralTransientSegmentManifestWire> for StructuralTransientSegmentManifest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralTransientSegmentManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            request: wire.request,
            terminal_checkpoint: wire.terminal_checkpoint,
            solver_evidence: wire.solver_evidence,
            initial_sample: wire.initial_sample,
            samples: wire.samples,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl StructuralTransientSegmentManifest {
    fn new(
        request: StructuralTransientBlockRequest,
        terminal_checkpoint: StructuralTransientCheckpoint,
        solver_evidence: StructuralTransientBlockSolverEvidence,
        initial_sample: Option<&StructuralTransientPhysicalSample>,
        samples: &[StructuralTransientSegmentSample],
    ) -> AnalysisExecutionResult<Self> {
        let entries = samples
            .iter()
            .map(|sample| StructuralTransientSegmentEntry {
                schedule_ordinal: sample.physical.schedule_ordinal,
                sample: sample.identity,
            })
            .collect::<Vec<_>>();
        let identity = segment_identity(
            1,
            &request,
            &terminal_checkpoint,
            &solver_evidence,
            initial_sample.map(|sample| sample.identity),
            &entries,
        )?;
        let manifest = Self {
            schema_version: 1,
            identity,
            request,
            terminal_checkpoint,
            solver_evidence,
            initial_sample: initial_sample.map(|sample| sample.identity),
            samples: entries,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.request.validate()?;
        self.terminal_checkpoint.validate()?;
        self.solver_evidence
            .validate_for(&self.request.scope.coordinate_space)?;
        let expected_sample_count = self
            .request
            .terminal_schedule_ordinal
            .checked_sub(self.request.start_schedule_ordinal)
            .and_then(|count| usize::try_from(count).ok())
            .ok_or(AnalysisExecutionError::Limit(
                "structural transient segment sample count exceeds platform limits",
            ))?;
        let expected_terminal_counters = StructuralTransientSolverCounters {
            host_sync_count: self
                .request
                .start_counters
                .host_sync_count
                .checked_add(self.solver_evidence.counters.host_sync_count)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient host synchronization count overflow",
                ))?,
            device_apply_k_count: self
                .request
                .start_counters
                .device_apply_k_count
                .checked_add(self.solver_evidence.counters.device_apply_k_count)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient device apply count overflow",
                ))?,
            device_apply_k_attempt_count: self
                .request
                .start_counters
                .device_apply_k_attempt_count
                .checked_add(self.solver_evidence.counters.device_apply_k_attempt_count)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient device apply-attempt count overflow",
                ))?,
            dense_factorization_count: self
                .request
                .start_counters
                .dense_factorization_count
                .checked_add(self.solver_evidence.counters.dense_factorization_count)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient dense factorization count overflow",
                ))?,
            dense_solve_count: self
                .request
                .start_counters
                .dense_solve_count
                .checked_add(self.solver_evidence.counters.dense_solve_count)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient dense solve count overflow",
                ))?,
        };
        if self.schema_version != 1
            || self.samples.len() != expected_sample_count
            || self.initial_sample.is_some() != (self.request.block_ordinal == 0)
            || self.terminal_checkpoint.scope != self.request.scope
            || self.terminal_checkpoint.predecessor != self.request.predecessor
            || self.terminal_checkpoint.block_ordinal != self.request.block_ordinal
            || self.terminal_checkpoint.continuation.schedule_ordinal
                != self.request.terminal_schedule_ordinal
            || self.terminal_checkpoint.counters != expected_terminal_counters
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient segment does not close its exact block request",
            ));
        }
        for (index, entry) in self.samples.iter().enumerate() {
            let expected_ordinal = self
                .request
                .start_schedule_ordinal
                .checked_add(u64::try_from(index).map_err(|_| {
                    AnalysisExecutionError::Limit(
                        "structural transient segment index exceeds canonical limits",
                    )
                })?)
                .and_then(|ordinal| ordinal.checked_add(1))
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient segment ordinal overflow",
                ))?;
            if entry.schedule_ordinal != expected_ordinal {
                return Err(AnalysisExecutionError::Identity(
                    "structural transient segment entries are not contiguous",
                ));
            }
        }
        if self.identity
            != segment_identity(
                self.schema_version,
                &self.request,
                &self.terminal_checkpoint,
                &self.solver_evidence,
                self.initial_sample,
                &self.samples,
            )?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient segment identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            TRANSIENT_SEGMENT_PREFIX,
            "structural-transient-segment-manifest",
            self,
            TRANSIENT_SEGMENT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            TRANSIENT_SEGMENT_PREFIX,
            "structural-transient-segment-manifest",
            bytes,
            TRANSIENT_SEGMENT_LIMITS,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTransientSegment {
    manifest: StructuralTransientSegmentManifest,
    initial_sample: Option<StructuralTransientPhysicalSample>,
    samples: Vec<StructuralTransientSegmentSample>,
}

impl StructuralTransientSegment {
    pub fn new(
        request: StructuralTransientBlockRequest,
        terminal_checkpoint: StructuralTransientCheckpoint,
        solver_evidence: StructuralTransientBlockSolverEvidence,
        initial_sample: Option<StructuralTransientPhysicalSample>,
        samples: Vec<StructuralTransientSegmentSample>,
    ) -> AnalysisExecutionResult<Self> {
        let manifest = StructuralTransientSegmentManifest::new(
            request,
            terminal_checkpoint,
            solver_evidence,
            initial_sample.as_ref(),
            &samples,
        )?;
        Self::from_manifest_and_samples(manifest, initial_sample, samples)
    }

    pub fn from_manifest_and_samples(
        manifest: StructuralTransientSegmentManifest,
        initial_sample: Option<StructuralTransientPhysicalSample>,
        samples: Vec<StructuralTransientSegmentSample>,
    ) -> AnalysisExecutionResult<Self> {
        let segment = Self {
            manifest,
            initial_sample,
            samples,
        };
        segment.validate()?;
        Ok(segment)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.manifest.validate()?;
        if self.samples.len() != self.manifest.samples.len()
            || self.initial_sample.as_ref().map(|sample| sample.identity)
                != self.manifest.initial_sample
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient segment payload count differs from its manifest",
            ));
        }
        if let Some(initial) = &self.initial_sample {
            initial.validate_against(&self.manifest.request.scope)?;
            if !matches!(initial.step, StructuralTransientSampleStep::Initial)
                || self.manifest.request.block_ordinal != 0
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural transient initial sample is not owned by block zero",
                ));
            }
        }
        for (entry, sample) in self.manifest.samples.iter().zip(&self.samples) {
            sample.validate_against(&self.manifest.request)?;
            if entry.schedule_ordinal != sample.physical.schedule_ordinal
                || entry.sample != sample.identity
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural transient segment sample differs from its manifest entry",
                ));
            }
        }
        let terminal = self.samples.last().ok_or_else(|| {
            AnalysisExecutionError::Invalid(
                "structural transient segment must contain an accepted sample".into(),
            )
        })?;
        let checkpoint = &self.manifest.terminal_checkpoint;
        if terminal.physical.state.displacement != checkpoint.state.displacement
            || terminal.physical.state.velocity != checkpoint.state.velocity
            || checkpoint.continuation.approach == StructuralTransientContinuationApproach::At
                && terminal.physical.state.acceleration != checkpoint.state.acceleration
            || terminal
                .physical
                .applied_load_work_j
                .canonical_value()
                .to_bits()
                != checkpoint
                    .balance
                    .applied_load_work_j
                    .canonical_value()
                    .to_bits()
            || terminal.physical.support_work_j.canonical_value().to_bits()
                != checkpoint
                    .balance
                    .support_work_j
                    .canonical_value()
                    .to_bits()
            || terminal
                .physical
                .damping_dissipation_j
                .canonical_value()
                .to_bits()
                != checkpoint
                    .balance
                    .damping_dissipation_j
                    .canonical_value()
                    .to_bits()
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient segment terminal state differs from its checkpoint",
            ));
        }
        Ok(())
    }

    pub fn manifest(&self) -> &StructuralTransientSegmentManifest {
        &self.manifest
    }

    pub fn initial_sample(&self) -> Option<&StructuralTransientPhysicalSample> {
        self.initial_sample.as_ref()
    }

    pub fn samples(&self) -> &[StructuralTransientSegmentSample] {
        &self.samples
    }
}

fn segment_sample_identity(
    sample: &StructuralTransientSegmentSample,
) -> AnalysisExecutionResult<StructuralTransientSegmentSampleIdentity> {
    StructuralTransientSegmentSampleIdentity::derive_content(
        "structural-transient-segment-sample",
        &(sample.schema_version, sample.request, &sample.physical),
    )
}

fn physical_sample_input_identity(
    schema_version: u16,
    scope: StructuralTransientCheckpointScopeIdentity,
    input: &StructuralTransientPhysicalSampleInput,
) -> AnalysisExecutionResult<StructuralTransientPhysicalSampleIdentity> {
    StructuralTransientPhysicalSampleIdentity::derive_content(
        "structural-transient-physical-sample",
        &PhysicalSampleIdentityInput {
            schema_version,
            scope,
            schedule_ordinal: input.schedule_ordinal,
            time_s: runmat_types::physical::Time::new(input.time_s.canonical_value()),
            step: input.step,
            state: &input.state,
            projected_residual_norm: input.projected_residual_norm,
            reaction: &input.reaction,
            strain_energy_j: runmat_types::physical::Energy::new(
                input.strain_energy_j.canonical_value(),
            ),
            kinetic_energy_j: runmat_types::physical::Energy::new(
                input.kinetic_energy_j.canonical_value(),
            ),
            applied_load_work_j: runmat_types::physical::Energy::new(
                input.applied_load_work_j.canonical_value(),
            ),
            support_work_j: runmat_types::physical::Energy::new(
                input.support_work_j.canonical_value(),
            ),
            damping_dissipation_j: runmat_types::physical::Energy::new(
                input.damping_dissipation_j.canonical_value(),
            ),
            energy_balance_residual_j: runmat_types::physical::Energy::new(
                input.energy_balance_residual_j.canonical_value(),
            ),
            relative_energy_balance_error: input.relative_energy_balance_error,
            dynamic_balance: &input.dynamic_balance,
        },
    )
}

fn physical_sample_identity(
    sample: &StructuralTransientPhysicalSample,
) -> AnalysisExecutionResult<StructuralTransientPhysicalSampleIdentity> {
    StructuralTransientPhysicalSampleIdentity::derive_content(
        "structural-transient-physical-sample",
        &PhysicalSampleIdentityInput {
            schema_version: sample.schema_version,
            scope: sample.scope,
            schedule_ordinal: sample.schedule_ordinal,
            time_s: runmat_types::physical::Time::new(sample.time_s.canonical_value()),
            step: sample.step,
            state: &sample.state,
            projected_residual_norm: sample.projected_residual_norm,
            reaction: &sample.reaction,
            strain_energy_j: runmat_types::physical::Energy::new(
                sample.strain_energy_j.canonical_value(),
            ),
            kinetic_energy_j: runmat_types::physical::Energy::new(
                sample.kinetic_energy_j.canonical_value(),
            ),
            applied_load_work_j: runmat_types::physical::Energy::new(
                sample.applied_load_work_j.canonical_value(),
            ),
            support_work_j: runmat_types::physical::Energy::new(
                sample.support_work_j.canonical_value(),
            ),
            damping_dissipation_j: runmat_types::physical::Energy::new(
                sample.damping_dissipation_j.canonical_value(),
            ),
            energy_balance_residual_j: runmat_types::physical::Energy::new(
                sample.energy_balance_residual_j.canonical_value(),
            ),
            relative_energy_balance_error: sample.relative_energy_balance_error,
            dynamic_balance: &sample.dynamic_balance,
        },
    )
}

#[derive(Serialize)]
struct PhysicalSampleIdentityInput<'a> {
    schema_version: u16,
    scope: StructuralTransientCheckpointScopeIdentity,
    schedule_ordinal: u64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    time_s: runmat_types::physical::Time,
    step: StructuralTransientSampleStep,
    state: &'a StructuralTransientContinuationState,
    projected_residual_norm: f64,
    reaction: &'a [f64],
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    strain_energy_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    kinetic_energy_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    applied_load_work_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    support_work_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    damping_dissipation_j: runmat_types::physical::Energy,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    energy_balance_residual_j: runmat_types::physical::Energy,
    relative_energy_balance_error: f64,
    dynamic_balance: &'a StructuralDynamicBalanceSnapshot,
}

fn segment_identity(
    schema_version: u16,
    request: &StructuralTransientBlockRequest,
    terminal_checkpoint: &StructuralTransientCheckpoint,
    solver_evidence: &StructuralTransientBlockSolverEvidence,
    initial_sample: Option<StructuralTransientPhysicalSampleIdentity>,
    samples: &[StructuralTransientSegmentEntry],
) -> AnalysisExecutionResult<StructuralTransientSegmentIdentity> {
    StructuralTransientSegmentIdentity::derive_content(
        "structural-transient-segment",
        &(
            schema_version,
            request,
            terminal_checkpoint,
            solver_evidence,
            initial_sample,
            samples,
        ),
    )
}

#[cfg(test)]
mod tests {
    use crate::execution::transient_checkpoint::tests::{checkpoint, scope};

    use super::*;

    fn solver_evidence() -> StructuralTransientBlockSolverEvidence {
        StructuralTransientBlockSolverEvidence {
            solver_method: SolverMethod::NewmarkAverageAccelerationPcg,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            preconditioner: SolverPreconditioner::None,
            counters: StructuralTransientSolverCounters {
                host_sync_count: 1,
                ..StructuralTransientSolverCounters::default()
            },
            accelerator_evidence: Vec::new(),
        }
    }

    fn sample(
        request: &StructuralTransientBlockRequest,
        ordinal: u64,
        acceleration: Vec<f64>,
    ) -> StructuralTransientSegmentSample {
        StructuralTransientSegmentSample::new(
            request,
            StructuralTransientSegmentSampleInput {
                schedule_ordinal: ordinal,
                time_s: runmat_types::physical::Time::new(if ordinal == 1 { 0.5 } else { 1.0 }),
                accepted_time_step_s: runmat_types::physical::Time::new(0.5),
                state: StructuralTransientContinuationState {
                    displacement: vec![ordinal as f64, 0.0],
                    velocity: vec![0.5, 0.0],
                    acceleration,
                },
                projected_residual_norm: 1.0e-12,
                reaction: vec![2.0, 0.0],
                strain_energy_j: runmat_types::physical::Energy::new(0.6),
                kinetic_energy_j: runmat_types::physical::Energy::new(0.4),
                applied_load_work_j: runmat_types::physical::Energy::new(ordinal as f64),
                support_work_j: runmat_types::physical::Energy::new(-0.25 * ordinal as f64),
                damping_dissipation_j: runmat_types::physical::Energy::new(0.1 * ordinal as f64),
                energy_balance_residual_j: runmat_types::physical::Energy::new(0.0),
                relative_energy_balance_error: 0.0,
                dynamic_balance: StructuralDynamicBalanceSnapshot {
                    applied_load: vec![1.0, 0.0],
                    inertia_force: vec![0.25, 0.0],
                    damping_force: vec![0.25, 0.0],
                    elastic_force: vec![0.5, 0.0],
                    equilibrium_residual: vec![0.0, 0.0],
                },
            },
        )
        .unwrap()
    }

    fn initial_sample(
        scope: &StructuralTransientCheckpointScope,
    ) -> StructuralTransientPhysicalSample {
        StructuralTransientPhysicalSample::new(
            scope,
            StructuralTransientPhysicalSampleInput {
                schedule_ordinal: 0,
                time_s: runmat_types::physical::Time::new(0.0),
                step: StructuralTransientSampleStep::Initial,
                state: StructuralTransientContinuationState {
                    displacement: vec![0.0, 0.0],
                    velocity: vec![0.0, 0.0],
                    acceleration: vec![0.0, 0.0],
                },
                projected_residual_norm: 0.0,
                reaction: vec![0.0, 0.0],
                strain_energy_j: runmat_types::physical::Energy::new(0.0),
                kinetic_energy_j: runmat_types::physical::Energy::new(0.0),
                applied_load_work_j: runmat_types::physical::Energy::new(0.0),
                support_work_j: runmat_types::physical::Energy::new(0.0),
                damping_dissipation_j: runmat_types::physical::Energy::new(0.0),
                energy_balance_residual_j: runmat_types::physical::Energy::new(0.0),
                relative_energy_balance_error: 0.0,
                dynamic_balance: StructuralDynamicBalanceSnapshot {
                    applied_load: vec![0.0, 0.0],
                    inertia_force: vec![0.0, 0.0],
                    damping_force: vec![0.0, 0.0],
                    elastic_force: vec![0.0, 0.0],
                    equilibrium_residual: vec![0.0, 0.0],
                },
            },
        )
        .unwrap()
    }

    #[test]
    fn partitioned_segment_round_trips_and_preserves_right_limit_continuation() {
        let request = StructuralTransientBlockRequest::new(scope(), None, 1).unwrap();
        let segment = StructuralTransientSegment::new(
            request.clone(),
            checkpoint(None, 1),
            solver_evidence(),
            Some(initial_sample(&request.scope)),
            vec![sample(&request, 1, vec![7.0, 0.0])],
        )
        .unwrap();
        assert_ne!(
            segment.samples()[0].physical.state.acceleration,
            segment.manifest().terminal_checkpoint.state.acceleration
        );
        let manifest = StructuralTransientSegmentManifest::from_canonical_bytes(
            &segment.manifest().canonical_bytes().unwrap(),
        )
        .unwrap();
        let initial = StructuralTransientPhysicalSample::from_canonical_bytes(
            &segment.initial_sample().unwrap().canonical_bytes().unwrap(),
        )
        .unwrap();
        let samples = segment
            .samples()
            .iter()
            .map(|sample| {
                StructuralTransientSegmentSample::from_canonical_bytes(
                    &sample.canonical_bytes().unwrap(),
                )
                .unwrap()
            })
            .collect();
        assert_eq!(
            StructuralTransientSegment::from_manifest_and_samples(
                manifest,
                Some(initial),
                samples,
            )
            .unwrap(),
            segment
        );
    }

    #[test]
    fn segment_rejects_schedule_payload_balance_and_manifest_substitution() {
        let request = StructuralTransientBlockRequest::new(scope(), None, 1).unwrap();
        let segment = StructuralTransientSegment::new(
            request.clone(),
            checkpoint(None, 1),
            solver_evidence(),
            Some(initial_sample(&request.scope)),
            vec![sample(&request, 1, vec![7.0, 0.0])],
        )
        .unwrap();

        let mut samples = segment.samples().to_vec();
        samples[0].physical.time_s = runmat_types::physical::Time::new(0.75);
        assert!(StructuralTransientSegment::from_manifest_and_samples(
            segment.manifest().clone(),
            segment.initial_sample().cloned(),
            samples
        )
        .is_err());

        let mut samples = segment.samples().to_vec();
        samples[0].physical.reaction.pop();
        assert!(StructuralTransientSegment::from_manifest_and_samples(
            segment.manifest().clone(),
            segment.initial_sample().cloned(),
            samples
        )
        .is_err());

        let mut samples = segment.samples().to_vec();
        samples[0] = sample(&request, 1, vec![8.0, 0.0]);
        assert!(StructuralTransientSegment::from_manifest_and_samples(
            segment.manifest().clone(),
            segment.initial_sample().cloned(),
            samples
        )
        .is_err());

        let mut manifest = segment.manifest().clone();
        manifest.terminal_checkpoint.state.displacement[0] = 9.0;
        assert!(StructuralTransientSegment::from_manifest_and_samples(
            manifest,
            segment.initial_sample().cloned(),
            segment.samples().to_vec()
        )
        .is_err());
    }
}
