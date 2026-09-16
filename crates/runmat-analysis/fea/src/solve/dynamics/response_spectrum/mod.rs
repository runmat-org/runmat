mod combine;
mod interpolate;

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use runmat_analysis_core::{
    CoordinateFrame, DirectionalCombination, ModalCombination, ResponseSpectrum,
    ResponseSpectrumId, SpectrumDamping, SpectrumEnvelope, SpectrumResponse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{contracts::FeaRunError, ModalBasisArtifact, ModalBasisIdentity};

use combine::{combine_directional_responses, combine_modal_responses};
use interpolate::interpolate_spectrum;

const RESULT_PREFIX: &[u8] = b"runmat-modal-response-spectrum-result\0";
const RESULT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 65_536, 24);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModalResponseSpectrumResultIdentity(Digest);

impl ModalResponseSpectrumResultIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalResponseSpectrumModeContribution {
    pub mode_index: usize,
    pub frequency_hz: f64,
    pub interpolated_ordinate: f64,
    pub directional_participation_factor: f64,
    pub directional_effective_mass: f64,
    pub cumulative_mass_ratio: f64,
    pub generalized_peak: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalResponseSpectrumTruncationEvidence {
    pub selected_mode_count: usize,
    pub available_mode_count: usize,
    pub required_cumulative_mass_ratio: f64,
    pub achieved_cumulative_mass_ratio: f64,
    pub directional_total_free_mass: f64,
    pub directional_captured_mass: f64,
    pub residual_mode_included: bool,
    pub rigid_response_included: bool,
    pub missing_mass_included: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalResponseSpectrumContribution {
    pub spectrum_id: ResponseSpectrumId,
    pub spectrum_provenance_digest: String,
    pub direction: [f64; 3],
    pub damping_ratio: f64,
    pub modal_combination: ModalCombination,
    pub modes: Vec<ModalResponseSpectrumModeContribution>,
    pub truncation: ModalResponseSpectrumTruncationEvidence,
    pub combined_response: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModalResponseSpectrumResult {
    identity: ModalResponseSpectrumResultIdentity,
    basis: ModalBasisIdentity,
    response: SpectrumResponse,
    directional_combination: DirectionalCombination,
    contributions: Vec<ModalResponseSpectrumContribution>,
    combined_response: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModalResponseSpectrumResultWire {
    identity: ModalResponseSpectrumResultIdentity,
    basis: ModalBasisIdentity,
    response: SpectrumResponse,
    directional_combination: DirectionalCombination,
    contributions: Vec<ModalResponseSpectrumContribution>,
    combined_response: Vec<f64>,
}

impl TryFrom<ModalResponseSpectrumResultWire> for ModalResponseSpectrumResult {
    type Error = FeaRunError;

    fn try_from(wire: ModalResponseSpectrumResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            identity: wire.identity,
            basis: wire.basis,
            response: wire.response,
            directional_combination: wire.directional_combination,
            contributions: wire.contributions,
            combined_response: wire.combined_response,
        };
        result.validate()?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for ModalResponseSpectrumResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        ModalResponseSpectrumResultWire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

impl ModalResponseSpectrumResult {
    pub const fn identity(&self) -> ModalResponseSpectrumResultIdentity {
        self.identity
    }

    pub const fn basis(&self) -> ModalBasisIdentity {
        self.basis
    }

    pub const fn response(&self) -> SpectrumResponse {
        self.response
    }

    pub const fn directional_combination(&self) -> DirectionalCombination {
        self.directional_combination
    }

    pub fn contributions(&self) -> &[ModalResponseSpectrumContribution] {
        &self.contributions
    }

    pub fn combined_response(&self) -> &[f64] {
        &self.combined_response
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, FeaRunError> {
        self.validate()?;
        encode_contract(
            RESULT_PREFIX,
            "modal-response-spectrum-result",
            self,
            RESULT_LIMITS,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, FeaRunError> {
        let result: Self = decode_contract(
            RESULT_PREFIX,
            "modal-response-spectrum-result",
            bytes,
            RESULT_LIMITS,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        result.validate()?;
        Ok(result)
    }

    pub fn validate_for(
        &self,
        basis: &ModalBasisArtifact,
        spectra: &[ResponseSpectrum],
    ) -> Result<(), FeaRunError> {
        self.validate()?;
        if solve_modal_response_spectrum(basis, spectra)? != *self {
            return Err(FeaRunError::Solve(
                "modal response-spectrum result differs from its basis or spectrum inputs".into(),
            ));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), FeaRunError> {
        if self.contributions.is_empty()
            || self.combined_response.is_empty()
            || self
                .combined_response
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || self.contributions.iter().any(|contribution| {
                !contribution_is_valid(contribution, self.combined_response.len())
            })
            || self.identity != self.derive_identity()?
        {
            return Err(FeaRunError::Solve(
                "modal response-spectrum result is not canonical".into(),
            ));
        }
        let ids = self
            .contributions
            .iter()
            .map(|contribution| &contribution.spectrum_id)
            .collect::<BTreeSet<_>>();
        if ids.len() != self.contributions.len() {
            return Err(FeaRunError::Solve(
                "modal response-spectrum result repeats a spectrum identity".into(),
            ));
        }
        let recombined = combine_directional_responses(
            &self
                .contributions
                .iter()
                .map(|contribution| contribution.combined_response.as_slice())
                .collect::<Vec<_>>(),
            self.directional_combination,
        )?;
        if !vectors_close(&recombined, &self.combined_response) {
            return Err(FeaRunError::Solve(
                "modal response-spectrum directional combination is inconsistent".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ModalResponseSpectrumResultIdentity, FeaRunError> {
        encode_contract(
            RESULT_PREFIX,
            "modal-response-spectrum-result-identity",
            &(
                self.basis,
                self.response,
                self.directional_combination,
                &self.contributions,
                &self.combined_response,
            ),
            RESULT_LIMITS,
        )
        .map(Digest::sha256)
        .map(ModalResponseSpectrumResultIdentity)
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }
}

pub fn solve_modal_response_spectrum(
    basis: &ModalBasisArtifact,
    spectra: &[ResponseSpectrum],
) -> Result<ModalResponseSpectrumResult, FeaRunError> {
    basis.validate()?;
    if spectra.is_empty() {
        return Err(FeaRunError::Solve(
            "modal response requires at least one response spectrum".into(),
        ));
    }
    let response = spectra[0].response;
    let directional_combination = spectra[0].directional_combination;
    if spectra.iter().any(|spectrum| {
        spectrum.response != response || spectrum.directional_combination != directional_combination
    }) {
        return Err(FeaRunError::Solve(
            "directional response combination requires one response quantity and policy".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut contributions = Vec::with_capacity(spectra.len());
    for spectrum in spectra {
        spectrum.validate_intrinsic().map_err(FeaRunError::Solve)?;
        if !ids.insert(spectrum.spectrum_id.clone()) {
            return Err(FeaRunError::Solve(
                "modal response repeats a response-spectrum identity".into(),
            ));
        }
        contributions.push(solve_spectrum_direction(basis, spectrum)?);
    }
    let combined_response = combine_directional_responses(
        &contributions
            .iter()
            .map(|contribution| contribution.combined_response.as_slice())
            .collect::<Vec<_>>(),
        directional_combination,
    )?;
    let mut result = ModalResponseSpectrumResult {
        identity: ModalResponseSpectrumResultIdentity(Digest::sha256([])),
        basis: basis.identity(),
        response,
        directional_combination,
        contributions,
        combined_response,
    };
    result.identity = result.derive_identity()?;
    result.validate()?;
    Ok(result)
}

fn solve_spectrum_direction(
    basis: &ModalBasisArtifact,
    spectrum: &ResponseSpectrum,
) -> Result<ModalResponseSpectrumContribution, FeaRunError> {
    if spectrum.direction.frame != CoordinateFrame::Global {
        return Err(FeaRunError::Solve(
            "modal response currently requires spectrum directions in the global frame".into(),
        ));
    }
    if spectrum.truncation.include_residual_mode
        || spectrum.truncation.include_rigid_response
        || spectrum.truncation.include_missing_mass
    {
        return Err(FeaRunError::Solve(
            "residual-mode, rigid-response, and missing-mass corrections require the physical operator and are not admitted by the basis-only solver"
                .into(),
        ));
    }
    let direction = normalized_direction(spectrum.direction.vector)?;
    let damping_ratio = damping_ratio(spectrum.damping)?;
    let total_free_mass = direction
        .iter()
        .zip(basis.total_free_mass())
        .map(|(component, mass)| component * component * mass)
        .sum::<f64>();
    if !total_free_mass.is_finite() || total_free_mass <= 0.0 {
        return Err(FeaRunError::Solve(
            "spectrum direction has no positive free structural mass".into(),
        ));
    }

    let mut captured_mass = 0.0;
    let mut modes = Vec::new();
    let mut modal_vectors = Vec::new();
    for (mode_offset, mode) in basis.modes().iter().enumerate() {
        let participation = direction
            .iter()
            .zip(mode.directional())
            .map(|(component, metric)| component * metric.participation_factor())
            .sum::<f64>();
        let effective_mass = participation * participation * mode.modal_mass();
        captured_mass += effective_mass;
        let cumulative_mass_ratio = captured_mass / total_free_mass;
        let ordinate = interpolate_spectrum(spectrum, mode.frequency_hz())?;
        let envelope_sign = match spectrum.envelope {
            SpectrumEnvelope::Negative => -1.0,
            SpectrumEnvelope::Positive | SpectrumEnvelope::Maximax => 1.0,
        };
        let generalized_peak = envelope_sign * participation * ordinate;
        modes.push(ModalResponseSpectrumModeContribution {
            mode_index: mode_offset + 1,
            frequency_hz: mode.frequency_hz(),
            interpolated_ordinate: ordinate,
            directional_participation_factor: participation,
            directional_effective_mass: effective_mass,
            cumulative_mass_ratio,
            generalized_peak,
        });
        modal_vectors.push(
            mode.dof_shape()
                .iter()
                .map(|shape| shape * generalized_peak)
                .collect::<Vec<_>>(),
        );
        if cumulative_mass_ratio + 64.0 * f64::EPSILON
            >= spectrum.truncation.minimum_cumulative_mass_ratio
        {
            break;
        }
    }
    let achieved = modes
        .last()
        .map(|mode| mode.cumulative_mass_ratio)
        .unwrap_or(0.0);
    if achieved + 64.0 * f64::EPSILON < spectrum.truncation.minimum_cumulative_mass_ratio {
        return Err(FeaRunError::Solve(format!(
            "modal basis captures {:.6} of directional mass but spectrum requires {:.6}",
            achieved, spectrum.truncation.minimum_cumulative_mass_ratio
        )));
    }
    let frequencies = modes
        .iter()
        .map(|mode| mode.frequency_hz)
        .collect::<Vec<_>>();
    let combined_response = combine_modal_responses(
        &modal_vectors,
        &frequencies,
        damping_ratio,
        spectrum.modal_combination,
    )?;
    Ok(ModalResponseSpectrumContribution {
        spectrum_id: spectrum.spectrum_id.clone(),
        spectrum_provenance_digest: spectrum.provenance_digest.clone(),
        direction,
        damping_ratio,
        modal_combination: spectrum.modal_combination,
        truncation: ModalResponseSpectrumTruncationEvidence {
            selected_mode_count: modes.len(),
            available_mode_count: basis.truncation().available_mode_count(),
            required_cumulative_mass_ratio: spectrum.truncation.minimum_cumulative_mass_ratio,
            achieved_cumulative_mass_ratio: achieved,
            directional_total_free_mass: total_free_mass,
            directional_captured_mass: captured_mass,
            residual_mode_included: false,
            rigid_response_included: false,
            missing_mass_included: false,
        },
        modes,
        combined_response,
    })
}

fn normalized_direction(vector: [f64; 3]) -> Result<[f64; 3], FeaRunError> {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(FeaRunError::Solve(
            "spectrum direction must be finite and nonzero".into(),
        ));
    }
    Ok(vector.map(|value| value / norm))
}

fn damping_ratio(damping: SpectrumDamping) -> Result<f64, FeaRunError> {
    let ratio = match damping {
        SpectrumDamping::Ratio(value) => value,
        SpectrumDamping::QualityFactor(value) => 1.0 / (2.0 * value),
    };
    if !ratio.is_finite() || !(0.0..1.0).contains(&ratio) {
        return Err(FeaRunError::Solve(
            "modal response requires a damping ratio in [0, 1)".into(),
        ));
    }
    Ok(ratio)
}

fn contribution_is_valid(
    contribution: &ModalResponseSpectrumContribution,
    response_length: usize,
) -> bool {
    let truncation = &contribution.truncation;
    let summed_mass = contribution
        .modes
        .iter()
        .map(|mode| mode.directional_effective_mass)
        .sum::<f64>();
    let expected_ratio = summed_mass / truncation.directional_total_free_mass;
    let direction_norm = contribution
        .direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    !contribution.spectrum_id.0.trim().is_empty()
        && !contribution.spectrum_provenance_digest.trim().is_empty()
        && contribution.direction.iter().all(|value| value.is_finite())
        && scalar_close(direction_norm, 1.0)
        && contribution.damping_ratio.is_finite()
        && (0.0..1.0).contains(&contribution.damping_ratio)
        && !contribution.modes.is_empty()
        && contribution.combined_response.len() == response_length
        && contribution
            .combined_response
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && contribution.modes.iter().enumerate().all(|(index, mode)| {
            mode.mode_index == index + 1
                && mode.frequency_hz.is_finite()
                && mode.frequency_hz >= 0.0
                && mode.interpolated_ordinate.is_finite()
                && mode.interpolated_ordinate >= 0.0
                && mode.directional_participation_factor.is_finite()
                && mode.directional_effective_mass.is_finite()
                && mode.directional_effective_mass >= 0.0
                && mode.cumulative_mass_ratio.is_finite()
                && mode.cumulative_mass_ratio >= 0.0
                && mode.generalized_peak.is_finite()
        })
        && contribution.modes.windows(2).all(|pair| {
            pair[0].frequency_hz <= pair[1].frequency_hz
                && pair[0].cumulative_mass_ratio <= pair[1].cumulative_mass_ratio
        })
        && truncation.selected_mode_count == contribution.modes.len()
        && truncation.available_mode_count >= contribution.modes.len()
        && truncation.required_cumulative_mass_ratio.is_finite()
        && (0.0..=1.0).contains(&truncation.required_cumulative_mass_ratio)
        && truncation.achieved_cumulative_mass_ratio.is_finite()
        && truncation.achieved_cumulative_mass_ratio >= 0.0
        && contribution
            .truncation
            .directional_total_free_mass
            .is_finite()
        && contribution.truncation.directional_total_free_mass > 0.0
        && contribution
            .truncation
            .directional_captured_mass
            .is_finite()
        && contribution.truncation.directional_captured_mass >= 0.0
        && !truncation.residual_mode_included
        && !truncation.rigid_response_included
        && !truncation.missing_mass_included
        && scalar_close(summed_mass, truncation.directional_captured_mass)
        && scalar_close(expected_ratio, truncation.achieved_cumulative_mass_ratio)
}

fn vectors_close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| scalar_close(*left, *right))
}

fn scalar_close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 128.0 * f64::EPSILON * left.abs().max(right.abs()).max(1.0)
}
