use runmat_analysis_core::ResponseSpectrum;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::solve::dynamics::ModalResponseSpectrumDirectionResult;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult,
    StructuralModalResponseSpectrumDirectionRequestIdentity,
    StructuralModalResponseSpectrumDirectionResultIdentity, StructuralModalResponseSpectrumScope,
    StructuralModalResponseSpectrumWorkload,
};

const ENTRY_PREFIX: &[u8] = b"runmat-structural-modal-response-spectrum-direction/v1\0";
const ENTRY_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralModalResponseSpectrumDirectionRequestWire"
)]
pub struct StructuralModalResponseSpectrumDirectionRequest {
    schema_version: u16,
    identity: StructuralModalResponseSpectrumDirectionRequestIdentity,
    scope: StructuralModalResponseSpectrumScope,
    ordinal: usize,
    spectrum: ResponseSpectrum,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalResponseSpectrumDirectionRequestWire {
    schema_version: u16,
    identity: StructuralModalResponseSpectrumDirectionRequestIdentity,
    scope: StructuralModalResponseSpectrumScope,
    ordinal: usize,
    spectrum: ResponseSpectrum,
}

impl TryFrom<StructuralModalResponseSpectrumDirectionRequestWire>
    for StructuralModalResponseSpectrumDirectionRequest
{
    type Error = AnalysisExecutionError;

    fn try_from(
        wire: StructuralModalResponseSpectrumDirectionRequestWire,
    ) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            ordinal: wire.ordinal,
            spectrum: wire.spectrum,
        };
        request.validate()?;
        Ok(request)
    }
}

impl StructuralModalResponseSpectrumDirectionRequest {
    pub fn new(
        workload: &StructuralModalResponseSpectrumWorkload,
        scope: StructuralModalResponseSpectrumScope,
        ordinal: usize,
    ) -> AnalysisExecutionResult<Self> {
        scope.validate_for(workload)?;
        let spectrum = workload
            .spectra()?
            .get(ordinal)
            .ok_or_else(|| invalid("response-spectrum direction ordinal is outside the plan"))?
            .to_owned()
            .clone();
        let mut request = Self {
            schema_version: 1,
            identity: StructuralModalResponseSpectrumDirectionRequestIdentity::derive(
                "structural-modal-response-spectrum-direction-request-placeholder",
                &1_u8,
            )?,
            scope,
            ordinal,
            spectrum,
        };
        request.identity = request.derive_identity()?;
        request.validate_for(workload)?;
        Ok(request)
    }

    pub const fn identity(&self) -> StructuralModalResponseSpectrumDirectionRequestIdentity {
        self.identity
    }
    pub fn scope(&self) -> &StructuralModalResponseSpectrumScope {
        &self.scope
    }
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }
    pub fn spectrum(&self) -> &ResponseSpectrum {
        &self.spectrum
    }

    pub fn validate_for(
        &self,
        workload: &StructuralModalResponseSpectrumWorkload,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        self.scope.validate_for(workload)?;
        if workload
            .spectra()?
            .get(self.ordinal)
            .is_none_or(|expected| **expected != self.spectrum)
        {
            return Err(AnalysisExecutionError::Identity(
                "response-spectrum direction request differs from its workload",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            ENTRY_PREFIX,
            "structural-modal-response-spectrum-direction-request",
            self,
            ENTRY_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            ENTRY_PREFIX,
            "structural-modal-response-spectrum-direction-request",
            bytes,
            ENTRY_LIMITS,
        )?)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        self.spectrum.validate_intrinsic().map_err(invalid)?;
        if self.schema_version != 1
            || self.ordinal >= self.scope.spectrum_ids().len()
            || self.scope.spectrum_ids()[self.ordinal] != self.spectrum.spectrum_id
            || self.scope.response() != self.spectrum.response
            || self.scope.directional_combination() != self.spectrum.directional_combination
            || self.identity != self.derive_identity()?
        {
            return Err(invalid(
                "response-spectrum direction request is not canonical",
            ));
        }
        Ok(())
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<StructuralModalResponseSpectrumDirectionRequestIdentity> {
        StructuralModalResponseSpectrumDirectionRequestIdentity::derive(
            "structural-modal-response-spectrum-direction-request",
            &(
                self.schema_version,
                self.scope.identity(),
                self.ordinal,
                &self.spectrum,
            ),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralModalResponseSpectrumDirectionResultWire"
)]
pub struct StructuralModalResponseSpectrumDirectionResult {
    schema_version: u16,
    identity: StructuralModalResponseSpectrumDirectionResultIdentity,
    request: StructuralModalResponseSpectrumDirectionRequest,
    numerical: ModalResponseSpectrumDirectionResult,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalResponseSpectrumDirectionResultWire {
    schema_version: u16,
    identity: StructuralModalResponseSpectrumDirectionResultIdentity,
    request: StructuralModalResponseSpectrumDirectionRequest,
    numerical: ModalResponseSpectrumDirectionResult,
}

impl TryFrom<StructuralModalResponseSpectrumDirectionResultWire>
    for StructuralModalResponseSpectrumDirectionResult
{
    type Error = AnalysisExecutionError;

    fn try_from(
        wire: StructuralModalResponseSpectrumDirectionResultWire,
    ) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            request: wire.request,
            numerical: wire.numerical,
        };
        result.validate()?;
        Ok(result)
    }
}

impl StructuralModalResponseSpectrumDirectionResult {
    pub fn new(
        request: StructuralModalResponseSpectrumDirectionRequest,
        numerical: ModalResponseSpectrumDirectionResult,
    ) -> AnalysisExecutionResult<Self> {
        let mut result = Self {
            schema_version: 1,
            identity: StructuralModalResponseSpectrumDirectionResultIdentity::derive(
                "structural-modal-response-spectrum-direction-result-placeholder",
                &1_u8,
            )?,
            request,
            numerical,
        };
        result.identity = result.derive_identity()?;
        result.validate()?;
        Ok(result)
    }

    pub const fn identity(&self) -> StructuralModalResponseSpectrumDirectionResultIdentity {
        self.identity
    }
    pub fn request(&self) -> &StructuralModalResponseSpectrumDirectionRequest {
        &self.request
    }
    pub fn numerical(&self) -> &ModalResponseSpectrumDirectionResult {
        &self.numerical
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            ENTRY_PREFIX,
            "structural-modal-response-spectrum-direction-result",
            self,
            ENTRY_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            ENTRY_PREFIX,
            "structural-modal-response-spectrum-direction-result",
            bytes,
            ENTRY_LIMITS,
        )?)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.request.validate()?;
        if self.schema_version != 1
            || self.numerical.basis() != self.request.scope().basis()
            || self.numerical.spectrum() != self.request.spectrum()
            || self.numerical.contribution().combined_response.len()
                != self.request.scope().physical_dof_count()
            || self.identity != self.derive_identity()?
        {
            return Err(invalid(
                "response-spectrum direction result is not canonical",
            ));
        }
        Ok(())
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<StructuralModalResponseSpectrumDirectionResultIdentity> {
        StructuralModalResponseSpectrumDirectionResultIdentity::derive(
            "structural-modal-response-spectrum-direction-result",
            &(
                self.schema_version,
                self.request.identity(),
                self.numerical.identity(),
            ),
        )
    }
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}
