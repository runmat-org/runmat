mod complex;
mod integrate;
mod interval;
mod plan;
mod quadrature;
mod recovery;
mod spectrum;

#[cfg(test)]
mod tests;

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

pub use integrate::solve_modal_random_response;
pub use interval::{
    assemble_described_random_response_intervals, assemble_modal_random_response_intervals,
    describe_modal_random_response, random_response_interval_breakpoints,
    solve_modal_random_response_interval, ModalRandomResponseIntervalIdentity,
    ModalRandomResponseIntervalResult, RandomResponseFieldLayout, RandomResponseFieldVariance,
    RandomResponseIntervalEvidence, RandomResponsePlanDescription,
};

const RESULT_PREFIX: &[u8] = b"runmat-modal-random-response-result\0";
const EXCITATION_PREFIX: &[u8] = b"runmat-random-excitation-spectrum\0";
const RESULT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 24));

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomResponseIntegrationPolicy {
    pub modal_damping_ratio: f64,
    pub minimum_cumulative_mass_ratio: f64,
    pub relative_variance_tolerance: f64,
    pub maximum_refinement_levels: u8,
}

impl RandomResponseIntegrationPolicy {
    pub fn production_default() -> Self {
        Self {
            modal_damping_ratio: 0.05,
            minimum_cumulative_mass_ratio: 0.9,
            relative_variance_tolerance: 1.0e-5,
            maximum_refinement_levels: 12,
        }
    }

    pub(crate) fn validate(self) -> Result<(), FeaRunError> {
        if !self.modal_damping_ratio.is_finite()
            || !(0.0..1.0).contains(&self.modal_damping_ratio)
            || !self.minimum_cumulative_mass_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_cumulative_mass_ratio)
            || !self.relative_variance_tolerance.is_finite()
            || self.relative_variance_tolerance <= 0.0
            || self.maximum_refinement_levels == 0
            || self.maximum_refinement_levels > 24
        {
            return Err(FeaRunError::Solve(
                "random-response integration policy is invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomResponseFieldStatistics {
    quantity: StructuralLinearFieldQuantity,
    shape: Vec<usize>,
    variance: Vec<f64>,
    rms: Vec<f64>,
}

impl RandomResponseFieldStatistics {
    pub const fn quantity(&self) -> StructuralLinearFieldQuantity {
        self.quantity
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn variance(&self) -> &[f64] {
        &self.variance
    }

    pub fn rms(&self) -> &[f64] {
        &self.rms
    }

    fn validate(&self) -> Result<(), FeaRunError> {
        let count = self
            .shape
            .iter()
            .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| FeaRunError::Solve("random-response field shape overflows".into()))?;
        if self.shape.is_empty()
            || self.shape.contains(&0)
            || self.variance.len() != count
            || self.rms.len() != count
            || self
                .variance
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || self
                .rms
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || self
                .variance
                .iter()
                .zip(&self.rms)
                .any(|(variance, rms)| !close(*rms * *rms, *variance))
        {
            return Err(FeaRunError::Solve(
                "random-response field statistics are not canonical".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomResponseIntegrationEvidence {
    pub authored_interval_count: usize,
    pub resonance_split_count: usize,
    pub evaluated_frequency_count: usize,
    pub maximum_refinement_level_used: u8,
    pub maximum_estimated_relative_variance_error: f64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModalRandomResponseResultIdentity(Digest);

impl ModalRandomResponseResultIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

/// Zero-mean stationary random-response statistics for one exact modal basis and excitation PSD.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModalRandomResponseResult {
    identity: ModalRandomResponseResultIdentity,
    basis: ModalBasisIdentity,
    excitation: RandomExcitationId,
    excitation_digest: Digest,
    policy: RandomResponseIntegrationPolicy,
    selected_mode_count: usize,
    captured_mass_ratio_by_channel: Vec<f64>,
    dof_variance: Vec<f64>,
    dof_rms: Vec<f64>,
    fields: Vec<RandomResponseFieldStatistics>,
    evidence: RandomResponseIntegrationEvidence,
}

struct ModalRandomResponseResultParts {
    basis: ModalBasisIdentity,
    excitation: RandomExcitationId,
    excitation_digest: Digest,
    policy: RandomResponseIntegrationPolicy,
    selected_mode_count: usize,
    captured_mass_ratio_by_channel: Vec<f64>,
    dof_variance: Vec<f64>,
    fields: Vec<RandomResponseFieldStatistics>,
    evidence: RandomResponseIntegrationEvidence,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModalRandomResponseResultWire {
    identity: ModalRandomResponseResultIdentity,
    basis: ModalBasisIdentity,
    excitation: RandomExcitationId,
    excitation_digest: Digest,
    policy: RandomResponseIntegrationPolicy,
    selected_mode_count: usize,
    captured_mass_ratio_by_channel: Vec<f64>,
    dof_variance: Vec<f64>,
    dof_rms: Vec<f64>,
    fields: Vec<RandomResponseFieldStatistics>,
    evidence: RandomResponseIntegrationEvidence,
}

impl TryFrom<ModalRandomResponseResultWire> for ModalRandomResponseResult {
    type Error = FeaRunError;

    fn try_from(wire: ModalRandomResponseResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            identity: wire.identity,
            basis: wire.basis,
            excitation: wire.excitation,
            excitation_digest: wire.excitation_digest,
            policy: wire.policy,
            selected_mode_count: wire.selected_mode_count,
            captured_mass_ratio_by_channel: wire.captured_mass_ratio_by_channel,
            dof_variance: wire.dof_variance,
            dof_rms: wire.dof_rms,
            fields: wire.fields,
            evidence: wire.evidence,
        };
        result.validate()?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for ModalRandomResponseResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        ModalRandomResponseResultWire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

impl ModalRandomResponseResult {
    fn from_parts(parts: ModalRandomResponseResultParts) -> Result<Self, FeaRunError> {
        let dof_rms = parts
            .dof_variance
            .iter()
            .map(|variance| variance.sqrt())
            .collect();
        let mut result = Self {
            identity: ModalRandomResponseResultIdentity(Digest::sha256([])),
            basis: parts.basis,
            excitation: parts.excitation,
            excitation_digest: parts.excitation_digest,
            policy: parts.policy,
            selected_mode_count: parts.selected_mode_count,
            captured_mass_ratio_by_channel: parts.captured_mass_ratio_by_channel,
            dof_variance: parts.dof_variance,
            dof_rms,
            fields: parts.fields,
            evidence: parts.evidence,
        };
        result.identity = result.derive_identity()?;
        result.validate()?;
        Ok(result)
    }

    pub const fn identity(&self) -> ModalRandomResponseResultIdentity {
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

    pub const fn selected_mode_count(&self) -> usize {
        self.selected_mode_count
    }

    pub fn captured_mass_ratio_by_channel(&self) -> &[f64] {
        &self.captured_mass_ratio_by_channel
    }

    pub fn dof_variance(&self) -> &[f64] {
        &self.dof_variance
    }

    pub fn dof_rms(&self) -> &[f64] {
        &self.dof_rms
    }

    pub fn fields(&self) -> &[RandomResponseFieldStatistics] {
        &self.fields
    }

    pub fn field(
        &self,
        quantity: StructuralLinearFieldQuantity,
    ) -> Option<&RandomResponseFieldStatistics> {
        self.fields.iter().find(|field| field.quantity == quantity)
    }

    pub const fn evidence(&self) -> &RandomResponseIntegrationEvidence {
        &self.evidence
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, FeaRunError> {
        self.validate()?;
        encode_contract(
            RESULT_PREFIX,
            "modal-random-response-result",
            self,
            RESULT_LIMITS,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, FeaRunError> {
        decode_contract(
            RESULT_PREFIX,
            "modal-random-response-result",
            bytes,
            RESULT_LIMITS,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }

    /// Regenerates the complete numerical result against the exact operator, basis, and spectrum.
    pub fn validate_for(
        &self,
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
        excitation: &RandomExcitationSpectrum,
    ) -> Result<(), FeaRunError> {
        self.validate()?;
        if self.basis != basis.identity()
            || self.excitation != excitation.excitation_id
            || self.excitation_digest != random_excitation_spectrum_digest(excitation)?
        {
            return Err(FeaRunError::Solve(
                "random-response result does not name the supplied basis and excitation".into(),
            ));
        }
        let regenerated = solve_modal_random_response(summary, basis, excitation, self.policy)?;
        if &regenerated != self {
            return Err(FeaRunError::Solve(
                "random-response result differs from exact numerical regeneration".into(),
            ));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), FeaRunError> {
        self.policy.validate()?;
        if self.excitation.0.trim().is_empty()
            || self.selected_mode_count == 0
            || self.captured_mass_ratio_by_channel.is_empty()
            || self
                .captured_mass_ratio_by_channel
                .iter()
                .any(|ratio| !ratio.is_finite() || !(0.0..=1.0 + 1.0e-12).contains(ratio))
            || self.dof_variance.is_empty()
            || self.dof_variance.len() != self.dof_rms.len()
            || self
                .dof_variance
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || self
                .dof_rms
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || self
                .dof_variance
                .iter()
                .zip(&self.dof_rms)
                .any(|(variance, rms)| !close(*rms * *rms, *variance))
            || self.fields.is_empty()
            || self.fields.iter().any(|field| field.validate().is_err())
            || self
                .fields
                .windows(2)
                .any(|pair| pair[0].quantity >= pair[1].quantity)
            || !evidence_is_valid(&self.evidence)
            || self.identity != self.derive_identity()?
        {
            return Err(FeaRunError::Solve(
                "modal random-response result is not canonical".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ModalRandomResponseResultIdentity, FeaRunError> {
        encode_contract(
            RESULT_PREFIX,
            "modal-random-response-result-identity",
            &(
                self.basis,
                &self.excitation,
                self.excitation_digest,
                self.policy,
                self.selected_mode_count,
                &self.captured_mass_ratio_by_channel,
                &self.dof_variance,
                &self.dof_rms,
                &self.fields,
                &self.evidence,
            ),
            RESULT_LIMITS,
        )
        .map(Digest::sha256)
        .map(ModalRandomResponseResultIdentity)
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }
}

pub(crate) fn random_excitation_spectrum_digest(
    excitation: &RandomExcitationSpectrum,
) -> Result<Digest, FeaRunError> {
    excitation
        .validate_intrinsic()
        .map_err(FeaRunError::Solve)?;
    encode_contract(
        EXCITATION_PREFIX,
        "random-excitation-spectrum",
        excitation,
        RESULT_LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| FeaRunError::Solve(error.to_string()))
}

fn evidence_is_valid(evidence: &RandomResponseIntegrationEvidence) -> bool {
    evidence.authored_interval_count > 0
        && evidence.evaluated_frequency_count >= 2
        && evidence.maximum_refinement_level_used > 0
        && evidence
            .maximum_estimated_relative_variance_error
            .is_finite()
        && evidence.maximum_estimated_relative_variance_error >= 0.0
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 64.0 * f64::EPSILON * left.abs().max(right.abs()).max(f64::MIN_POSITIVE)
}
