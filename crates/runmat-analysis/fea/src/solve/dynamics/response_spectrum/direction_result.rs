use runmat_analysis_core::ResponseSpectrum;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary, contracts::FeaRunError, ModalBasisArtifact, ModalBasisIdentity,
};

use super::{
    solve_spectrum_direction, validation::contribution_is_valid, ModalResponseSpectrumContribution,
};

const DIRECTION_RESULT_PREFIX: &[u8] = b"runmat-modal-response-spectrum-direction-result\0";
const DIRECTION_RESULT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::platform_content(65_536, 24));

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModalResponseSpectrumDirectionResultIdentity(Digest);

impl ModalResponseSpectrumDirectionResultIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

/// Canonical output of one independently executable spectrum direction.
///
/// The complete spectrum is retained deliberately: a worker result can therefore be authenticated
/// and revalidated without consulting mutable model state, while the final assembly can prove that
/// every requested direction appears exactly once.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModalResponseSpectrumDirectionResult {
    identity: ModalResponseSpectrumDirectionResultIdentity,
    basis: ModalBasisIdentity,
    spectrum: ResponseSpectrum,
    contribution: ModalResponseSpectrumContribution,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModalResponseSpectrumDirectionResultWire {
    identity: ModalResponseSpectrumDirectionResultIdentity,
    basis: ModalBasisIdentity,
    spectrum: ResponseSpectrum,
    contribution: ModalResponseSpectrumContribution,
}

impl TryFrom<ModalResponseSpectrumDirectionResultWire> for ModalResponseSpectrumDirectionResult {
    type Error = FeaRunError;

    fn try_from(wire: ModalResponseSpectrumDirectionResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            identity: wire.identity,
            basis: wire.basis,
            spectrum: wire.spectrum,
            contribution: wire.contribution,
        };
        result.validate()?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for ModalResponseSpectrumDirectionResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        ModalResponseSpectrumDirectionResultWire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

impl ModalResponseSpectrumDirectionResult {
    pub(super) fn from_contribution(
        basis: ModalBasisIdentity,
        spectrum: ResponseSpectrum,
        contribution: ModalResponseSpectrumContribution,
    ) -> Result<Self, FeaRunError> {
        let mut result = Self {
            identity: ModalResponseSpectrumDirectionResultIdentity(Digest::sha256([])),
            basis,
            spectrum,
            contribution,
        };
        result.identity = result.derive_identity()?;
        result.validate()?;
        Ok(result)
    }

    pub const fn identity(&self) -> ModalResponseSpectrumDirectionResultIdentity {
        self.identity
    }

    pub const fn basis(&self) -> ModalBasisIdentity {
        self.basis
    }

    pub fn spectrum(&self) -> &ResponseSpectrum {
        &self.spectrum
    }

    pub fn contribution(&self) -> &ModalResponseSpectrumContribution {
        &self.contribution
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, FeaRunError> {
        self.validate()?;
        encode_contract(
            DIRECTION_RESULT_PREFIX,
            "modal-response-spectrum-direction-result",
            self,
            DIRECTION_RESULT_LIMITS,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, FeaRunError> {
        decode_contract(
            DIRECTION_RESULT_PREFIX,
            "modal-response-spectrum-direction-result",
            bytes,
            DIRECTION_RESULT_LIMITS,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }

    pub fn validate_for(
        &self,
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
    ) -> Result<(), FeaRunError> {
        self.validate()?;
        basis.validate_for(summary)?;
        if self.basis != basis.identity()
            || self.contribution != solve_spectrum_direction(Some(summary), basis, &self.spectrum)?
        {
            return Err(FeaRunError::Solve(
                "response-spectrum direction result differs from its operator, basis, or spectrum"
                    .into(),
            ));
        }
        Ok(())
    }

    pub(super) fn validate(&self) -> Result<(), FeaRunError> {
        self.spectrum
            .validate_intrinsic()
            .map_err(FeaRunError::Solve)?;
        let spectrum_digest = self
            .spectrum
            .ordinates
            .numerical_digest()
            .map_err(FeaRunError::Solve)?;
        if self.contribution.spectrum_id != self.spectrum.spectrum_id
            || self.contribution.spectrum_provenance_digest != spectrum_digest
            || self.contribution.modal_combination != self.spectrum.modal_combination
            || self.contribution.rigid_response_policy != self.spectrum.rigid_response
            || !contribution_is_valid(
                &self.contribution,
                self.contribution.combined_response.len(),
            )
            || self.identity != self.derive_identity()?
        {
            return Err(FeaRunError::Solve(
                "response-spectrum direction result is not canonical".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ModalResponseSpectrumDirectionResultIdentity, FeaRunError> {
        crate::contracts::canonical_digest::canonical_contract_digest(
            DIRECTION_RESULT_PREFIX,
            "modal-response-spectrum-direction-result-identity",
            &(self.basis, &self.spectrum, &self.contribution),
            DIRECTION_RESULT_LIMITS,
        )
        .map(ModalResponseSpectrumDirectionResultIdentity)
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }
}
