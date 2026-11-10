mod combine;
mod correction;
mod direction_result;
mod direction_solve;
mod interpolate;
mod recovery;
mod validation;

#[cfg(test)]
mod qualification;
#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use runmat_analysis_core::{
    DirectionalCombination, ModalCombination, ModalTruncationCorrection, ResponseSpectrum,
    ResponseSpectrumId, RigidResponsePolicy, SpectrumResponse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::assembly::AssemblySummary;
use crate::{contracts::FeaRunError, ModalBasisArtifact, ModalBasisIdentity};

pub use crate::post::StructuralLinearFieldQuantity;
use combine::combine_directional_responses;
#[cfg(test)]
use combine::{combine_modal_responses, combine_periodic_and_rigid_responses};
pub use direction_result::{
    ModalResponseSpectrumDirectionResult, ModalResponseSpectrumDirectionResultIdentity,
};
use direction_solve::solve_spectrum_direction;
#[cfg(test)]
use interpolate::interpolate_spectrum;
pub use recovery::{
    recover_modal_response_spectrum_fields, response_spectrum_deformation_recovery_unavailability,
    ResponseSpectrumDeformationRecovery, ResponseSpectrumDeformationRecoveryUnavailability,
    ResponseSpectrumFieldRecovery, ResponseSpectrumFieldRecoveryIdentity,
    ResponseSpectrumLinearField, ResponseSpectrumNodalEnvelope, ResponseSpectrumNodalField,
    ResponseSpectrumNodalQuantity,
};
use validation::{contribution_is_valid, vectors_close};

const RESULT_PREFIX: &[u8] = b"runmat-modal-response-spectrum-result\0";
const RESULT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::platform_content(65_536, 24));

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
    pub kind: ModalResponseSpectrumModeKind,
    pub mode_index: usize,
    pub frequency_hz: f64,
    pub interpolated_ordinate: f64,
    pub directional_participation_factor: f64,
    pub directional_effective_mass: f64,
    pub cumulative_mass_ratio: f64,
    pub generalized_peak: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalResponseSpectrumModeKind {
    Eigenmode,
    ResidualVector,
}

/// Response-spectrum vectors are component-wise statistical peak magnitudes, not one concurrent
/// signed structural state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseSpectrumResultSemantics {
    StatisticalComponentMagnitudeEnvelope,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseSpectrumDisplacementConversion {
    Direct,
    DivideByAngularFrequency,
    DivideByAngularFrequencySquared,
}

impl ResponseSpectrumDisplacementConversion {
    pub const fn for_response(response: SpectrumResponse) -> Option<Self> {
        match response {
            SpectrumResponse::RelativeDisplacement => Some(Self::Direct),
            SpectrumResponse::PseudoVelocity => Some(Self::DivideByAngularFrequency),
            SpectrumResponse::PseudoAcceleration => Some(Self::DivideByAngularFrequencySquared),
            SpectrumResponse::RelativeVelocity
            | SpectrumResponse::RelativeAcceleration
            | SpectrumResponse::AbsoluteAcceleration => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalResponseSpectrumTruncationEvidence {
    pub selected_eigenmode_count: usize,
    pub available_eigenmode_count: usize,
    pub residual_vector_count: usize,
    pub required_cumulative_mass_ratio: f64,
    pub achieved_cumulative_mass_ratio: f64,
    pub directional_total_free_mass: f64,
    pub directional_captured_mass: f64,
    pub correction: ModalTruncationCorrection,
    pub correction_response: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalResponseSpectrumContribution {
    pub spectrum_id: ResponseSpectrumId,
    pub spectrum_provenance_digest: String,
    pub direction: [f64; 3],
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub damping_ratio: runmat_types::physical::DampingRatio,
    pub modal_combination: ModalCombination,
    pub rigid_response_policy: RigidResponsePolicy,
    pub rigid_coefficients: Vec<f64>,
    pub modes: Vec<ModalResponseSpectrumModeContribution>,
    pub truncation: ModalResponseSpectrumTruncationEvidence,
    pub periodic_response: Vec<f64>,
    pub rigid_response: Vec<f64>,
    pub combined_response: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModalResponseSpectrumResult {
    identity: ModalResponseSpectrumResultIdentity,
    basis: ModalBasisIdentity,
    response: SpectrumResponse,
    semantics: ResponseSpectrumResultSemantics,
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
    semantics: ResponseSpectrumResultSemantics,
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
            semantics: wire.semantics,
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

    pub const fn semantics(&self) -> ResponseSpectrumResultSemantics {
        self.semantics
    }

    pub const fn displacement_conversion(&self) -> Option<ResponseSpectrumDisplacementConversion> {
        ResponseSpectrumDisplacementConversion::for_response(self.response)
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

    pub fn validate_for_operator(
        &self,
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
        spectra: &[ResponseSpectrum],
    ) -> Result<(), FeaRunError> {
        self.validate()?;
        if solve_modal_response_spectrum_with_operator(summary, basis, spectra)? != *self {
            return Err(FeaRunError::Solve(
                "modal response-spectrum result differs from its operator, basis, or spectrum inputs"
                    .into(),
            ));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), FeaRunError> {
        if self.semantics != ResponseSpectrumResultSemantics::StatisticalComponentMagnitudeEnvelope
            || self.contributions.is_empty()
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
        crate::contracts::canonical_digest::canonical_contract_digest(
            RESULT_PREFIX,
            "modal-response-spectrum-result-identity",
            &(
                self.basis,
                self.response,
                self.semantics,
                self.directional_combination,
                &self.contributions,
                &self.combined_response,
            ),
            RESULT_LIMITS,
        )
        .map(ModalResponseSpectrumResultIdentity)
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }
}

pub fn solve_modal_response_spectrum(
    basis: &ModalBasisArtifact,
    spectra: &[ResponseSpectrum],
) -> Result<ModalResponseSpectrumResult, FeaRunError> {
    solve_modal_response_spectrum_internal(None, basis, spectra)
}

pub fn solve_modal_response_spectrum_with_operator(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    spectra: &[ResponseSpectrum],
) -> Result<ModalResponseSpectrumResult, FeaRunError> {
    basis.validate_for(summary)?;
    solve_modal_response_spectrum_internal(Some(summary), basis, spectra)
}

pub fn solve_modal_response_spectrum_direction_with_operator(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    spectrum: &ResponseSpectrum,
) -> Result<ModalResponseSpectrumDirectionResult, FeaRunError> {
    basis.validate_for(summary)?;
    let contribution = solve_spectrum_direction(Some(summary), basis, spectrum)?;
    let result = ModalResponseSpectrumDirectionResult::from_contribution(
        basis.identity(),
        spectrum.clone(),
        contribution,
    )?;
    result.validate_for(summary, basis)?;
    Ok(result)
}

pub fn assemble_modal_response_spectrum_directions(
    basis: &ModalBasisArtifact,
    spectra: &[ResponseSpectrum],
    direction_results: &[ModalResponseSpectrumDirectionResult],
) -> Result<ModalResponseSpectrumResult, FeaRunError> {
    basis.validate()?;
    assemble_modal_response_spectrum_direction_results(basis.identity(), spectra, direction_results)
}

pub fn assemble_modal_response_spectrum_direction_results(
    basis: ModalBasisIdentity,
    spectra: &[ResponseSpectrum],
    direction_results: &[ModalResponseSpectrumDirectionResult],
) -> Result<ModalResponseSpectrumResult, FeaRunError> {
    if spectra.len() != direction_results.len() || spectra.is_empty() {
        return Err(FeaRunError::Solve(
            "response-spectrum assembly requires exactly one result per requested direction".into(),
        ));
    }
    let contributions = spectra
        .iter()
        .zip(direction_results)
        .map(|(spectrum, result)| {
            result.validate()?;
            if result.basis() != basis || result.spectrum() != spectrum {
                return Err(FeaRunError::Solve(
                    "response-spectrum assembly received a substituted direction result".into(),
                ));
            }
            Ok(result.contribution().clone())
        })
        .collect::<Result<Vec<_>, _>>()?;
    assemble_modal_response_spectrum_result(basis, spectra, contributions)
}

fn solve_modal_response_spectrum_internal(
    summary: Option<&AssemblySummary>,
    basis: &ModalBasisArtifact,
    spectra: &[ResponseSpectrum],
) -> Result<ModalResponseSpectrumResult, FeaRunError> {
    basis.validate()?;
    if spectra.is_empty() {
        return Err(FeaRunError::Solve(
            "modal response requires at least one response spectrum".into(),
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
        contributions.push(solve_spectrum_direction(summary, basis, spectrum)?);
    }
    assemble_modal_response_spectrum_result(basis.identity(), spectra, contributions)
}

fn assemble_modal_response_spectrum_result(
    basis: ModalBasisIdentity,
    spectra: &[ResponseSpectrum],
    contributions: Vec<ModalResponseSpectrumContribution>,
) -> Result<ModalResponseSpectrumResult, FeaRunError> {
    let response = spectra
        .first()
        .ok_or_else(|| FeaRunError::Solve("modal response requires a spectrum".into()))?
        .response;
    let directional_combination = spectra[0].directional_combination;
    if spectra.iter().any(|spectrum| {
        spectrum.response != response || spectrum.directional_combination != directional_combination
    }) {
        return Err(FeaRunError::Solve(
            "directional response combination requires one response quantity and policy".into(),
        ));
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
        basis,
        response,
        semantics: ResponseSpectrumResultSemantics::StatisticalComponentMagnitudeEnvelope,
        directional_combination,
        contributions,
        combined_response,
    };
    result.identity = result.derive_identity()?;
    result.validate()?;
    Ok(result)
}
