use runmat_analysis_core::{RandomExcitationId, RandomExcitationSpectrum};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    contracts::FeaRunError,
    pipeline::modal_basis::{ModalBasisArtifact, ModalBasisIdentity},
    post::StructuralLinearFieldQuantity,
};

use super::{
    plan::RandomResponsePlan,
    quadrature::{compensated_add, integrate_interval},
    random_excitation_spectrum_digest, ModalRandomResponseResult, ModalRandomResponseResultParts,
    RandomResponseFieldStatistics, RandomResponseIntegrationEvidence,
    RandomResponseIntegrationPolicy,
};

const INTERVAL_PREFIX: &[u8] = b"runmat-modal-random-response-interval\0";
const INTERVAL_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 40_000_000, 65_536, 24);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomResponseFieldLayout {
    quantity: StructuralLinearFieldQuantity,
    shape: Vec<usize>,
    component_count: usize,
}

impl RandomResponseFieldLayout {
    pub const fn quantity(&self) -> StructuralLinearFieldQuantity {
        self.quantity
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub const fn component_count(&self) -> usize {
        self.component_count
    }
}

/// Placement-neutral numerical inventory used to partition and resource-admit interval work.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomResponsePlanDescription {
    basis: ModalBasisIdentity,
    excitation: RandomExcitationId,
    excitation_digest: Digest,
    policy: RandomResponseIntegrationPolicy,
    selected_mode_count: usize,
    captured_mass_ratio_by_channel: Vec<f64>,
    breakpoints_hz: Vec<f64>,
    authored_interval_count: usize,
    physical_dof_count: usize,
    fields: Vec<RandomResponseFieldLayout>,
}

impl RandomResponsePlanDescription {
    pub const fn basis(&self) -> ModalBasisIdentity {
        self.basis
    }

    pub fn excitation(&self) -> &RandomExcitationId {
        &self.excitation
    }

    pub const fn excitation_digest(&self) -> Digest {
        self.excitation_digest
    }

    pub const fn policy(&self) -> RandomResponseIntegrationPolicy {
        self.policy
    }

    pub const fn selected_mode_count(&self) -> usize {
        self.selected_mode_count
    }

    pub fn captured_mass_ratio_by_channel(&self) -> &[f64] {
        &self.captured_mass_ratio_by_channel
    }

    pub fn breakpoints_hz(&self) -> &[f64] {
        &self.breakpoints_hz
    }

    pub fn interval_count(&self) -> usize {
        self.breakpoints_hz.len().saturating_sub(1)
    }

    pub const fn authored_interval_count(&self) -> usize {
        self.authored_interval_count
    }

    pub const fn physical_dof_count(&self) -> usize {
        self.physical_dof_count
    }

    pub fn fields(&self) -> &[RandomResponseFieldLayout] {
        &self.fields
    }

    pub fn value_count(&self) -> usize {
        self.physical_dof_count
            + self
                .fields
                .iter()
                .map(RandomResponseFieldLayout::component_count)
                .sum::<usize>()
    }

    pub(crate) fn validate(&self) -> Result<(), FeaRunError> {
        self.policy.validate()?;
        if self.excitation.0.trim().is_empty()
            || self.selected_mode_count == 0
            || self.physical_dof_count == 0
            || self.captured_mass_ratio_by_channel.is_empty()
            || self
                .captured_mass_ratio_by_channel
                .iter()
                .any(|ratio| !ratio.is_finite() || !(0.0..=1.0 + 1.0e-12).contains(ratio))
            || self.breakpoints_hz.len() < 2
            || self.authored_interval_count == 0
            || self.authored_interval_count > self.interval_count()
            || self
                .breakpoints_hz
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || self
                .breakpoints_hz
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self.fields.is_empty()
            || self
                .fields
                .windows(2)
                .any(|pair| pair[0].quantity >= pair[1].quantity)
            || self.fields.iter().any(|field| {
                field.shape.is_empty()
                    || field.shape.contains(&0)
                    || field.component_count == 0
                    || field.shape.iter().product::<usize>() != field.component_count
            })
        {
            return Err(FeaRunError::Solve(
                "random-response plan description is invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomResponseFieldVariance {
    quantity: StructuralLinearFieldQuantity,
    shape: Vec<usize>,
    variance: Vec<f64>,
}

impl RandomResponseFieldVariance {
    pub const fn quantity(&self) -> StructuralLinearFieldQuantity {
        self.quantity
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn variance(&self) -> &[f64] {
        &self.variance
    }

    pub(crate) fn validate(&self) -> Result<(), FeaRunError> {
        let count = self
            .shape
            .iter()
            .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| {
                FeaRunError::Solve("random-response interval field shape overflows".into())
            })?;
        if self.shape.is_empty()
            || self.shape.contains(&0)
            || self.variance.len() != count
            || self
                .variance
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(FeaRunError::Solve(
                "random-response interval field variance is invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomResponseIntervalEvidence {
    pub evaluated_frequency_count: usize,
    pub refinement_level: u8,
    pub estimated_relative_error: f64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModalRandomResponseIntervalIdentity(Digest);

impl ModalRandomResponseIntervalIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

/// One independently executable canonical frequency-interval contribution.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModalRandomResponseIntervalResult {
    identity: ModalRandomResponseIntervalIdentity,
    basis: ModalBasisIdentity,
    excitation: RandomExcitationId,
    excitation_digest: Digest,
    policy: RandomResponseIntegrationPolicy,
    ordinal: usize,
    frequency_bounds_hz: [f64; 2],
    selected_mode_count: usize,
    captured_mass_ratio_by_channel: Vec<f64>,
    dof_variance: Vec<f64>,
    fields: Vec<RandomResponseFieldVariance>,
    evidence: RandomResponseIntervalEvidence,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModalRandomResponseIntervalResultWire {
    identity: ModalRandomResponseIntervalIdentity,
    basis: ModalBasisIdentity,
    excitation: RandomExcitationId,
    excitation_digest: Digest,
    policy: RandomResponseIntegrationPolicy,
    ordinal: usize,
    frequency_bounds_hz: [f64; 2],
    selected_mode_count: usize,
    captured_mass_ratio_by_channel: Vec<f64>,
    dof_variance: Vec<f64>,
    fields: Vec<RandomResponseFieldVariance>,
    evidence: RandomResponseIntervalEvidence,
}

impl TryFrom<ModalRandomResponseIntervalResultWire> for ModalRandomResponseIntervalResult {
    type Error = FeaRunError;

    fn try_from(wire: ModalRandomResponseIntervalResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            identity: wire.identity,
            basis: wire.basis,
            excitation: wire.excitation,
            excitation_digest: wire.excitation_digest,
            policy: wire.policy,
            ordinal: wire.ordinal,
            frequency_bounds_hz: wire.frequency_bounds_hz,
            selected_mode_count: wire.selected_mode_count,
            captured_mass_ratio_by_channel: wire.captured_mass_ratio_by_channel,
            dof_variance: wire.dof_variance,
            fields: wire.fields,
            evidence: wire.evidence,
        };
        result.validate()?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for ModalRandomResponseIntervalResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        ModalRandomResponseIntervalResultWire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

impl ModalRandomResponseIntervalResult {
    pub const fn identity(&self) -> ModalRandomResponseIntervalIdentity {
        self.identity
    }

    pub const fn basis(&self) -> ModalBasisIdentity {
        self.basis
    }

    pub fn excitation(&self) -> &RandomExcitationId {
        &self.excitation
    }

    pub const fn excitation_digest(&self) -> Digest {
        self.excitation_digest
    }

    pub const fn policy(&self) -> RandomResponseIntegrationPolicy {
        self.policy
    }

    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub const fn frequency_bounds_hz(&self) -> [f64; 2] {
        self.frequency_bounds_hz
    }

    pub const fn selected_mode_count(&self) -> usize {
        self.selected_mode_count
    }

    pub fn captured_mass_ratio_by_channel(&self) -> &[f64] {
        &self.captured_mass_ratio_by_channel
    }

    pub fn dof_variance(&self) -> &[f64] {
        &self.dof_variance
    }

    pub fn fields(&self) -> &[RandomResponseFieldVariance] {
        &self.fields
    }

    pub const fn evidence(&self) -> RandomResponseIntervalEvidence {
        self.evidence
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, FeaRunError> {
        self.validate()?;
        encode_contract(
            INTERVAL_PREFIX,
            "modal-random-response-interval-result",
            self,
            INTERVAL_LIMITS,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, FeaRunError> {
        decode_contract(
            INTERVAL_PREFIX,
            "modal-random-response-interval-result",
            bytes,
            INTERVAL_LIMITS,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }

    pub fn validate_for(
        &self,
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
        excitation: &RandomExcitationSpectrum,
    ) -> Result<(), FeaRunError> {
        self.validate()?;
        let regenerated = solve_modal_random_response_interval(
            summary,
            basis,
            excitation,
            self.policy,
            self.ordinal,
        )?;
        if &regenerated != self {
            return Err(FeaRunError::Solve(
                "random-response interval differs from exact numerical regeneration".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), FeaRunError> {
        self.policy.validate()?;
        let evidence = self.evidence;
        if self.excitation.0.trim().is_empty()
            || self.frequency_bounds_hz[0] < 0.0
            || !self.frequency_bounds_hz[0].is_finite()
            || !self.frequency_bounds_hz[1].is_finite()
            || self.frequency_bounds_hz[0] >= self.frequency_bounds_hz[1]
            || self.selected_mode_count == 0
            || self.captured_mass_ratio_by_channel.is_empty()
            || self
                .captured_mass_ratio_by_channel
                .iter()
                .any(|ratio| !ratio.is_finite() || !(0.0..=1.0 + 1.0e-12).contains(ratio))
            || self.dof_variance.is_empty()
            || self
                .dof_variance
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || self.fields.is_empty()
            || self.fields.iter().any(|field| field.validate().is_err())
            || self
                .fields
                .windows(2)
                .any(|pair| pair[0].quantity >= pair[1].quantity)
            || evidence.evaluated_frequency_count < 3
            || evidence.refinement_level == 0
            || evidence.refinement_level > self.policy.maximum_refinement_levels
            || !evidence.estimated_relative_error.is_finite()
            || evidence.estimated_relative_error < 0.0
            || self.identity != self.derive_identity()?
        {
            return Err(FeaRunError::Solve(
                "modal random-response interval is not canonical".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ModalRandomResponseIntervalIdentity, FeaRunError> {
        encode_contract(
            INTERVAL_PREFIX,
            "modal-random-response-interval-identity",
            &(
                self.basis,
                &self.excitation,
                self.excitation_digest,
                self.policy,
                self.ordinal,
                self.frequency_bounds_hz,
                self.selected_mode_count,
                &self.captured_mass_ratio_by_channel,
                &self.dof_variance,
                &self.fields,
                self.evidence,
            ),
            INTERVAL_LIMITS,
        )
        .map(Digest::sha256)
        .map(ModalRandomResponseIntervalIdentity)
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }
}

pub fn describe_modal_random_response(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    excitation: &RandomExcitationSpectrum,
    policy: RandomResponseIntegrationPolicy,
) -> Result<RandomResponsePlanDescription, FeaRunError> {
    let plan = RandomResponsePlan::new(summary, basis, excitation, policy)?;
    let response = plan.frequency_response(summary, basis, excitation)?;
    let description = RandomResponsePlanDescription {
        basis: basis.identity(),
        excitation: excitation.excitation_id.clone(),
        excitation_digest: random_excitation_spectrum_digest(excitation)?,
        policy,
        selected_mode_count: plan.selected_mode_count(),
        captured_mass_ratio_by_channel: plan.captured_mass_ratio_by_channel().to_vec(),
        breakpoints_hz: plan.breakpoints_hz().to_vec(),
        authored_interval_count: plan.authored_interval_count(),
        physical_dof_count: response.layout().dof_count,
        fields: response
            .layout()
            .fields
            .iter()
            .map(|field| RandomResponseFieldLayout {
                quantity: field.quantity,
                shape: field.shape.clone(),
                component_count: field.length,
            })
            .collect(),
    };
    description.validate()?;
    Ok(description)
}

pub fn random_response_interval_breakpoints(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    excitation: &RandomExcitationSpectrum,
    policy: RandomResponseIntegrationPolicy,
) -> Result<Vec<f64>, FeaRunError> {
    describe_modal_random_response(summary, basis, excitation, policy)
        .map(|description| description.breakpoints_hz)
}

pub fn solve_modal_random_response_interval(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    excitation: &RandomExcitationSpectrum,
    policy: RandomResponseIntegrationPolicy,
    ordinal: usize,
) -> Result<ModalRandomResponseIntervalResult, FeaRunError> {
    let plan = RandomResponsePlan::new(summary, basis, excitation, policy)?;
    let bounds = plan
        .breakpoints_hz()
        .get(ordinal..ordinal.saturating_add(2))
        .filter(|bounds| bounds.len() == 2)
        .ok_or_else(|| {
            FeaRunError::Solve("random-response interval ordinal is outside the plan".into())
        })?;
    let response = plan.frequency_response(summary, basis, excitation)?;
    let integrated = integrate_interval(bounds[0], bounds[1], policy, |frequency| {
        response.evaluate(frequency)
    })?;
    let layout = response.layout();
    let dof_variance = integrated.values[..layout.dof_count].to_vec();
    let fields = layout
        .fields
        .iter()
        .map(|field| RandomResponseFieldVariance {
            quantity: field.quantity,
            shape: field.shape.clone(),
            variance: integrated.values[field.offset..field.offset + field.length].to_vec(),
        })
        .collect();
    let mut result = ModalRandomResponseIntervalResult {
        identity: ModalRandomResponseIntervalIdentity(Digest::sha256([])),
        basis: basis.identity(),
        excitation: excitation.excitation_id.clone(),
        excitation_digest: random_excitation_spectrum_digest(excitation)?,
        policy,
        ordinal,
        frequency_bounds_hz: [bounds[0], bounds[1]],
        selected_mode_count: plan.selected_mode_count(),
        captured_mass_ratio_by_channel: plan.captured_mass_ratio_by_channel().to_vec(),
        dof_variance,
        fields,
        evidence: RandomResponseIntervalEvidence {
            evaluated_frequency_count: integrated.evaluated_frequency_count,
            refinement_level: integrated.refinement_level,
            estimated_relative_error: integrated.estimated_relative_error,
        },
    };
    result.identity = result.derive_identity()?;
    result.validate()?;
    Ok(result)
}

pub fn assemble_modal_random_response_intervals(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    excitation: &RandomExcitationSpectrum,
    policy: RandomResponseIntegrationPolicy,
    intervals: Vec<ModalRandomResponseIntervalResult>,
) -> Result<ModalRandomResponseResult, FeaRunError> {
    let description = describe_modal_random_response(summary, basis, excitation, policy)?;
    assemble_described_random_response_intervals(&description, intervals)
}

/// Deterministically assembles canonical interval contributions without requiring model residency.
///
/// This is the placement-neutral fan-in boundary used after local or distributed interval work.
/// The description binds the exact modal basis, excitation, integration policy, breakpoint set,
/// and result layout needed to reject missing, duplicated, substituted, or reordered work.
pub fn assemble_described_random_response_intervals(
    description: &RandomResponsePlanDescription,
    mut intervals: Vec<ModalRandomResponseIntervalResult>,
) -> Result<ModalRandomResponseResult, FeaRunError> {
    description.validate()?;
    intervals.sort_by_key(ModalRandomResponseIntervalResult::ordinal);
    let expected_count = description.interval_count();
    if intervals.len() != expected_count {
        return Err(FeaRunError::Solve(
            "random-response interval assembly is incomplete".into(),
        ));
    }
    for (ordinal, interval) in intervals.iter().enumerate() {
        interval.validate()?;
        let expected_bounds = [
            description.breakpoints_hz()[ordinal],
            description.breakpoints_hz()[ordinal + 1],
        ];
        if interval.ordinal != ordinal
            || interval.basis != description.basis()
            || interval.excitation != *description.excitation()
            || interval.excitation_digest != description.excitation_digest()
            || interval.policy != description.policy()
            || interval.frequency_bounds_hz != expected_bounds
            || interval.selected_mode_count != description.selected_mode_count()
            || interval.captured_mass_ratio_by_channel
                != description.captured_mass_ratio_by_channel()
            || interval.dof_variance.len() != description.physical_dof_count()
            || interval.fields.len() != description.fields().len()
            || interval
                .fields
                .iter()
                .zip(description.fields())
                .any(|(field, expected)| {
                    field.quantity != expected.quantity()
                        || field.shape != expected.shape()
                        || field.variance.len() != expected.component_count()
                })
        {
            return Err(FeaRunError::Solve(
                "random-response interval differs from its canonical assembly plan".into(),
            ));
        }
    }

    let flat_length = description.value_count();
    let mut total = vec![0.0; flat_length];
    let mut compensation = vec![0.0; flat_length];
    for interval in &intervals {
        let mut flat = Vec::with_capacity(flat_length);
        flat.extend_from_slice(&interval.dof_variance);
        for field in &interval.fields {
            flat.extend_from_slice(&field.variance);
        }
        compensated_add(&mut total, &mut compensation, &flat);
    }
    let dof_variance = total[..description.physical_dof_count()].to_vec();
    let mut offset = description.physical_dof_count();
    let fields = description
        .fields()
        .iter()
        .map(|field| {
            let end = offset + field.component_count();
            let variance = total[offset..end].to_vec();
            offset = end;
            let rms = variance.iter().map(|value| value.sqrt()).collect();
            RandomResponseFieldStatistics {
                quantity: field.quantity(),
                shape: field.shape().to_vec(),
                variance,
                rms,
            }
        })
        .collect();
    ModalRandomResponseResult::from_parts(ModalRandomResponseResultParts {
        basis: description.basis(),
        excitation: description.excitation().clone(),
        excitation_digest: description.excitation_digest(),
        policy: description.policy(),
        selected_mode_count: description.selected_mode_count(),
        captured_mass_ratio_by_channel: description.captured_mass_ratio_by_channel().to_vec(),
        dof_variance,
        fields,
        evidence: RandomResponseIntegrationEvidence {
            authored_interval_count: description.authored_interval_count(),
            resonance_split_count: expected_count
                .saturating_sub(description.authored_interval_count()),
            evaluated_frequency_count: intervals
                .iter()
                .map(|interval| interval.evidence.evaluated_frequency_count)
                .sum(),
            maximum_refinement_level_used: intervals
                .iter()
                .map(|interval| interval.evidence.refinement_level)
                .max()
                .unwrap_or(0),
            maximum_estimated_relative_variance_error: intervals
                .iter()
                .map(|interval| interval.evidence.estimated_relative_error)
                .fold(0.0_f64, f64::max),
        },
    })
}
