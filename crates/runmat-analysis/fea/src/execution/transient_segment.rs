use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::contracts::StructuralDynamicBalanceSnapshot;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, StructuralTransientBlockRequest,
    StructuralTransientBlockRequestIdentity, StructuralTransientCheckpoint,
    StructuralTransientContinuationApproach, StructuralTransientContinuationState,
    StructuralTransientSegmentIdentity, StructuralTransientSegmentSampleIdentity,
};

const TRANSIENT_SEGMENT_PREFIX: &[u8] = b"runmat-structural-transient-segment/v1\0";
const TRANSIENT_SEGMENT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralTransientSegmentSampleWire")]
pub struct StructuralTransientSegmentSample {
    pub schema_version: u16,
    pub identity: StructuralTransientSegmentSampleIdentity,
    pub request: StructuralTransientBlockRequestIdentity,
    pub schedule_ordinal: u64,
    pub time_s: f64,
    pub accepted_time_step_s: f64,
    pub state: StructuralTransientContinuationState,
    pub projected_residual_norm: f64,
    pub reaction: Vec<f64>,
    pub strain_energy_j: f64,
    pub kinetic_energy_j: f64,
    pub applied_load_work_j: f64,
    pub support_work_j: f64,
    pub damping_dissipation_j: f64,
    pub energy_balance_residual_j: f64,
    pub relative_energy_balance_error: f64,
    pub dynamic_balance: StructuralDynamicBalanceSnapshot,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientSegmentSampleWire {
    schema_version: u16,
    identity: StructuralTransientSegmentSampleIdentity,
    request: StructuralTransientBlockRequestIdentity,
    schedule_ordinal: u64,
    time_s: f64,
    accepted_time_step_s: f64,
    state: StructuralTransientContinuationState,
    projected_residual_norm: f64,
    reaction: Vec<f64>,
    strain_energy_j: f64,
    kinetic_energy_j: f64,
    applied_load_work_j: f64,
    support_work_j: f64,
    damping_dissipation_j: f64,
    energy_balance_residual_j: f64,
    relative_energy_balance_error: f64,
    dynamic_balance: StructuralDynamicBalanceSnapshot,
}

impl TryFrom<StructuralTransientSegmentSampleWire> for StructuralTransientSegmentSample {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralTransientSegmentSampleWire) -> Result<Self, Self::Error> {
        Ok(Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            request: wire.request,
            schedule_ordinal: wire.schedule_ordinal,
            time_s: wire.time_s,
            accepted_time_step_s: wire.accepted_time_step_s,
            state: wire.state,
            projected_residual_norm: wire.projected_residual_norm,
            reaction: wire.reaction,
            strain_energy_j: wire.strain_energy_j,
            kinetic_energy_j: wire.kinetic_energy_j,
            applied_load_work_j: wire.applied_load_work_j,
            support_work_j: wire.support_work_j,
            damping_dissipation_j: wire.damping_dissipation_j,
            energy_balance_residual_j: wire.energy_balance_residual_j,
            relative_energy_balance_error: wire.relative_energy_balance_error,
            dynamic_balance: wire.dynamic_balance,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTransientSegmentSampleInput {
    pub schedule_ordinal: u64,
    pub time_s: f64,
    pub accepted_time_step_s: f64,
    pub state: StructuralTransientContinuationState,
    pub projected_residual_norm: f64,
    pub reaction: Vec<f64>,
    pub strain_energy_j: f64,
    pub kinetic_energy_j: f64,
    pub applied_load_work_j: f64,
    pub support_work_j: f64,
    pub damping_dissipation_j: f64,
    pub energy_balance_residual_j: f64,
    pub relative_energy_balance_error: f64,
    pub dynamic_balance: StructuralDynamicBalanceSnapshot,
}

impl StructuralTransientSegmentSample {
    pub fn new(
        request: &StructuralTransientBlockRequest,
        input: StructuralTransientSegmentSampleInput,
    ) -> AnalysisExecutionResult<Self> {
        let identity = sample_input_identity(1, request.identity, &input)?;
        let sample = Self {
            schema_version: 1,
            identity,
            request: request.identity,
            schedule_ordinal: input.schedule_ordinal,
            time_s: input.time_s,
            accepted_time_step_s: input.accepted_time_step_s,
            state: input.state,
            projected_residual_norm: input.projected_residual_norm,
            reaction: input.reaction,
            strain_energy_j: input.strain_energy_j,
            kinetic_energy_j: input.kinetic_energy_j,
            applied_load_work_j: input.applied_load_work_j,
            support_work_j: input.support_work_j,
            damping_dissipation_j: input.damping_dissipation_j,
            energy_balance_residual_j: input.energy_balance_residual_j,
            relative_energy_balance_error: input.relative_energy_balance_error,
            dynamic_balance: input.dynamic_balance,
        };
        sample.validate_against(request)?;
        Ok(sample)
    }

    pub fn validate_against(
        &self,
        request: &StructuralTransientBlockRequest,
    ) -> AnalysisExecutionResult<()> {
        request.validate()?;
        let ordinal = usize::try_from(self.schedule_ordinal).map_err(|_| {
            AnalysisExecutionError::Invalid(
                "structural transient segment ordinal exceeds the platform index range".into(),
            )
        })?;
        let schedule_point = request.scope.schedule.points.get(ordinal).ok_or_else(|| {
            AnalysisExecutionError::Invalid(
                "structural transient segment sample lies outside its exact schedule".into(),
            )
        })?;
        let previous = request
            .scope
            .schedule
            .points
            .get(ordinal.checked_sub(1).ok_or_else(|| {
                AnalysisExecutionError::Invalid(
                    "structural transient segment cannot publish the initial point".into(),
                )
            })?)
            .ok_or_else(|| {
                AnalysisExecutionError::Invalid(
                    "structural transient segment sample lacks its preceding schedule point".into(),
                )
            })?;
        if self.schema_version != 1
            || self.request != request.identity
            || self.schedule_ordinal <= request.start_schedule_ordinal
            || self.schedule_ordinal > request.terminal_schedule_ordinal
            || self.time_s.to_bits() != schedule_point.time_s.to_bits()
            || self.accepted_time_step_s.to_bits()
                != (schedule_point.time_s - previous.time_s).to_bits()
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient segment sample differs from its exact request or schedule",
            ));
        }
        self.state.validate(request.scope.dof_count)?;
        self.dynamic_balance
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let scalars = [
            self.projected_residual_norm,
            self.strain_energy_j,
            self.kinetic_energy_j,
            self.applied_load_work_j,
            self.support_work_j,
            self.damping_dissipation_j,
            self.energy_balance_residual_j,
            self.relative_energy_balance_error,
        ];
        if self.reaction.len() != request.scope.dof_count
            || self.dynamic_balance.applied_load.len() != request.scope.dof_count
            || self.reaction.iter().any(|value| !value.is_finite())
            || scalars.iter().any(|value| !value.is_finite())
            || self.projected_residual_norm < 0.0
            || self.strain_energy_j < 0.0
            || self.kinetic_energy_j < 0.0
            || self.damping_dissipation_j < 0.0
            || self.relative_energy_balance_error < 0.0
            || self.identity != sample_identity(self)?
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient segment sample has invalid physical values or identity"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        if self.schema_version != 1 || self.identity != sample_identity(self)? {
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
        let sample: Self = decode_contract(
            TRANSIENT_SEGMENT_PREFIX,
            "structural-transient-segment-sample",
            bytes,
            TRANSIENT_SEGMENT_LIMITS,
        )?;
        if sample.schema_version != 1 || sample.identity != sample_identity(&sample)? {
            return Err(AnalysisExecutionError::Identity(
                "structural transient segment sample identity does not converge",
            ));
        }
        Ok(sample)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientSegmentEntry {
    pub schedule_ordinal: u64,
    pub sample: StructuralTransientSegmentSampleIdentity,
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
    pub samples: Vec<StructuralTransientSegmentEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientSegmentManifestWire {
    schema_version: u16,
    identity: StructuralTransientSegmentIdentity,
    request: StructuralTransientBlockRequest,
    terminal_checkpoint: StructuralTransientCheckpoint,
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
        samples: &[StructuralTransientSegmentSample],
    ) -> AnalysisExecutionResult<Self> {
        let entries = samples
            .iter()
            .map(|sample| StructuralTransientSegmentEntry {
                schedule_ordinal: sample.schedule_ordinal,
                sample: sample.identity,
            })
            .collect::<Vec<_>>();
        let identity = segment_identity(1, &request, &terminal_checkpoint, &entries)?;
        let manifest = Self {
            schema_version: 1,
            identity,
            request,
            terminal_checkpoint,
            samples: entries,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.request.validate()?;
        self.terminal_checkpoint.validate()?;
        let expected_sample_count = self
            .request
            .terminal_schedule_ordinal
            .checked_sub(self.request.start_schedule_ordinal)
            .and_then(|count| usize::try_from(count).ok())
            .ok_or(AnalysisExecutionError::Limit(
                "structural transient segment sample count exceeds platform limits",
            ))?;
        if self.schema_version != 1
            || self.samples.len() != expected_sample_count
            || self.terminal_checkpoint.scope != self.request.scope
            || self.terminal_checkpoint.predecessor != self.request.predecessor
            || self.terminal_checkpoint.block_ordinal != self.request.block_ordinal
            || self.terminal_checkpoint.continuation.schedule_ordinal
                != self.request.terminal_schedule_ordinal
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
    samples: Vec<StructuralTransientSegmentSample>,
}

impl StructuralTransientSegment {
    pub fn new(
        request: StructuralTransientBlockRequest,
        terminal_checkpoint: StructuralTransientCheckpoint,
        samples: Vec<StructuralTransientSegmentSample>,
    ) -> AnalysisExecutionResult<Self> {
        let manifest =
            StructuralTransientSegmentManifest::new(request, terminal_checkpoint, &samples)?;
        Self::from_manifest_and_samples(manifest, samples)
    }

    pub fn from_manifest_and_samples(
        manifest: StructuralTransientSegmentManifest,
        samples: Vec<StructuralTransientSegmentSample>,
    ) -> AnalysisExecutionResult<Self> {
        let segment = Self { manifest, samples };
        segment.validate()?;
        Ok(segment)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.manifest.validate()?;
        if self.samples.len() != self.manifest.samples.len() {
            return Err(AnalysisExecutionError::Identity(
                "structural transient segment payload count differs from its manifest",
            ));
        }
        for (entry, sample) in self.manifest.samples.iter().zip(&self.samples) {
            sample.validate_against(&self.manifest.request)?;
            if entry.schedule_ordinal != sample.schedule_ordinal || entry.sample != sample.identity
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
        if terminal.state.displacement != checkpoint.state.displacement
            || terminal.state.velocity != checkpoint.state.velocity
            || checkpoint.continuation.approach == StructuralTransientContinuationApproach::At
                && terminal.state.acceleration != checkpoint.state.acceleration
            || terminal.applied_load_work_j.to_bits()
                != checkpoint.balance.applied_load_work_j.to_bits()
            || terminal.support_work_j.to_bits() != checkpoint.balance.support_work_j.to_bits()
            || terminal.damping_dissipation_j.to_bits()
                != checkpoint.balance.damping_dissipation_j.to_bits()
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

    pub fn samples(&self) -> &[StructuralTransientSegmentSample] {
        &self.samples
    }
}

fn sample_identity(
    sample: &StructuralTransientSegmentSample,
) -> AnalysisExecutionResult<StructuralTransientSegmentSampleIdentity> {
    StructuralTransientSegmentSampleIdentity::derive(
        "structural-transient-segment-sample",
        &SegmentSampleIdentityInput {
            schema_version: sample.schema_version,
            request: sample.request,
            schedule_ordinal: sample.schedule_ordinal,
            time_s: sample.time_s,
            accepted_time_step_s: sample.accepted_time_step_s,
            state: &sample.state,
            projected_residual_norm: sample.projected_residual_norm,
            reaction: &sample.reaction,
            strain_energy_j: sample.strain_energy_j,
            kinetic_energy_j: sample.kinetic_energy_j,
            applied_load_work_j: sample.applied_load_work_j,
            support_work_j: sample.support_work_j,
            damping_dissipation_j: sample.damping_dissipation_j,
            energy_balance_residual_j: sample.energy_balance_residual_j,
            relative_energy_balance_error: sample.relative_energy_balance_error,
            dynamic_balance: &sample.dynamic_balance,
        },
    )
}

fn sample_input_identity(
    schema_version: u16,
    request: StructuralTransientBlockRequestIdentity,
    input: &StructuralTransientSegmentSampleInput,
) -> AnalysisExecutionResult<StructuralTransientSegmentSampleIdentity> {
    StructuralTransientSegmentSampleIdentity::derive(
        "structural-transient-segment-sample",
        &SegmentSampleIdentityInput {
            schema_version,
            request,
            schedule_ordinal: input.schedule_ordinal,
            time_s: input.time_s,
            accepted_time_step_s: input.accepted_time_step_s,
            state: &input.state,
            projected_residual_norm: input.projected_residual_norm,
            reaction: &input.reaction,
            strain_energy_j: input.strain_energy_j,
            kinetic_energy_j: input.kinetic_energy_j,
            applied_load_work_j: input.applied_load_work_j,
            support_work_j: input.support_work_j,
            damping_dissipation_j: input.damping_dissipation_j,
            energy_balance_residual_j: input.energy_balance_residual_j,
            relative_energy_balance_error: input.relative_energy_balance_error,
            dynamic_balance: &input.dynamic_balance,
        },
    )
}

#[derive(Serialize)]
struct SegmentSampleIdentityInput<'a> {
    schema_version: u16,
    request: StructuralTransientBlockRequestIdentity,
    schedule_ordinal: u64,
    time_s: f64,
    accepted_time_step_s: f64,
    state: &'a StructuralTransientContinuationState,
    projected_residual_norm: f64,
    reaction: &'a [f64],
    strain_energy_j: f64,
    kinetic_energy_j: f64,
    applied_load_work_j: f64,
    support_work_j: f64,
    damping_dissipation_j: f64,
    energy_balance_residual_j: f64,
    relative_energy_balance_error: f64,
    dynamic_balance: &'a StructuralDynamicBalanceSnapshot,
}

fn segment_identity(
    schema_version: u16,
    request: &StructuralTransientBlockRequest,
    terminal_checkpoint: &StructuralTransientCheckpoint,
    samples: &[StructuralTransientSegmentEntry],
) -> AnalysisExecutionResult<StructuralTransientSegmentIdentity> {
    StructuralTransientSegmentIdentity::derive(
        "structural-transient-segment",
        &(schema_version, request, terminal_checkpoint, samples),
    )
}

#[cfg(test)]
mod tests {
    use crate::execution::transient_checkpoint::tests::{checkpoint, scope};

    use super::*;

    fn sample(
        request: &StructuralTransientBlockRequest,
        ordinal: u64,
        acceleration: Vec<f64>,
    ) -> StructuralTransientSegmentSample {
        StructuralTransientSegmentSample::new(
            request,
            StructuralTransientSegmentSampleInput {
                schedule_ordinal: ordinal,
                time_s: if ordinal == 1 { 0.5 } else { 1.0 },
                accepted_time_step_s: 0.5,
                state: StructuralTransientContinuationState {
                    displacement: vec![ordinal as f64, 0.0],
                    velocity: vec![0.5, 0.0],
                    acceleration,
                },
                projected_residual_norm: 1.0e-12,
                reaction: vec![2.0, 0.0],
                strain_energy_j: 0.6,
                kinetic_energy_j: 0.4,
                applied_load_work_j: ordinal as f64,
                support_work_j: -0.25 * ordinal as f64,
                damping_dissipation_j: 0.1 * ordinal as f64,
                energy_balance_residual_j: 0.0,
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

    #[test]
    fn partitioned_segment_round_trips_and_preserves_right_limit_continuation() {
        let request = StructuralTransientBlockRequest::new(scope(), None, 1).unwrap();
        let segment = StructuralTransientSegment::new(
            request.clone(),
            checkpoint(None, 1),
            vec![sample(&request, 1, vec![7.0, 0.0])],
        )
        .unwrap();
        assert_ne!(
            segment.samples()[0].state.acceleration,
            segment.manifest().terminal_checkpoint.state.acceleration
        );
        let manifest = StructuralTransientSegmentManifest::from_canonical_bytes(
            &segment.manifest().canonical_bytes().unwrap(),
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
            StructuralTransientSegment::from_manifest_and_samples(manifest, samples).unwrap(),
            segment
        );
    }

    #[test]
    fn segment_rejects_schedule_payload_balance_and_manifest_substitution() {
        let request = StructuralTransientBlockRequest::new(scope(), None, 1).unwrap();
        let segment = StructuralTransientSegment::new(
            request.clone(),
            checkpoint(None, 1),
            vec![sample(&request, 1, vec![7.0, 0.0])],
        )
        .unwrap();

        let mut samples = segment.samples().to_vec();
        samples[0].time_s = 0.75;
        assert!(StructuralTransientSegment::from_manifest_and_samples(
            segment.manifest().clone(),
            samples
        )
        .is_err());

        let mut samples = segment.samples().to_vec();
        samples[0].reaction.pop();
        assert!(StructuralTransientSegment::from_manifest_and_samples(
            segment.manifest().clone(),
            samples
        )
        .is_err());

        let mut samples = segment.samples().to_vec();
        samples[0] = sample(&request, 1, vec![8.0, 0.0]);
        assert!(StructuralTransientSegment::from_manifest_and_samples(
            segment.manifest().clone(),
            samples
        )
        .is_err());

        let mut manifest = segment.manifest().clone();
        manifest.terminal_checkpoint.state.displacement[0] = 9.0;
        assert!(StructuralTransientSegment::from_manifest_and_samples(
            manifest,
            segment.samples().to_vec()
        )
        .is_err());
    }
}
