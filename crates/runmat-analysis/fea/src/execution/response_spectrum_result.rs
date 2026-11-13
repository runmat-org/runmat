use std::collections::BTreeSet;

use runmat_analysis_core::SpectrumResponse;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::solve::dynamics::{
    assemble_modal_response_spectrum_direction_results, ModalResponseSpectrumModeKind,
    ModalResponseSpectrumResult, ResponseSpectrumResultSemantics,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    NumericalReceiptIdentity, SolverResultIdentity,
    StructuralModalResponseSpectrumAssemblyRequestIdentity,
    StructuralModalResponseSpectrumDirectionResult, StructuralModalResponseSpectrumScope,
    StructuralModalResponseSpectrumWorkload,
};

const RESULT_PREFIX: &[u8] = b"runmat-structural-modal-response-spectrum-result/v1\0";
const RESULT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Deterministic fan-in request. Entry completion order is discarded in favor of authored order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralModalResponseSpectrumAssemblyRequestWire"
)]
pub struct StructuralModalResponseSpectrumAssemblyRequest {
    schema_version: u16,
    identity: StructuralModalResponseSpectrumAssemblyRequestIdentity,
    scope: StructuralModalResponseSpectrumScope,
    entries: Vec<StructuralModalResponseSpectrumDirectionResult>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalResponseSpectrumAssemblyRequestWire {
    schema_version: u16,
    identity: StructuralModalResponseSpectrumAssemblyRequestIdentity,
    scope: StructuralModalResponseSpectrumScope,
    entries: Vec<StructuralModalResponseSpectrumDirectionResult>,
}

impl TryFrom<StructuralModalResponseSpectrumAssemblyRequestWire>
    for StructuralModalResponseSpectrumAssemblyRequest
{
    type Error = AnalysisExecutionError;

    fn try_from(
        wire: StructuralModalResponseSpectrumAssemblyRequestWire,
    ) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            entries: wire.entries,
        };
        request.validate()?;
        Ok(request)
    }
}

impl StructuralModalResponseSpectrumAssemblyRequest {
    pub fn new(
        scope: StructuralModalResponseSpectrumScope,
        mut entries: Vec<StructuralModalResponseSpectrumDirectionResult>,
    ) -> AnalysisExecutionResult<Self> {
        entries.sort_by_key(|entry| entry.request().ordinal());
        let mut request = Self {
            schema_version: 1,
            identity: StructuralModalResponseSpectrumAssemblyRequestIdentity::derive(
                "structural-modal-response-spectrum-assembly-request-placeholder",
                &1_u8,
            )?,
            scope,
            entries,
        };
        request.identity = request.derive_identity()?;
        request.validate()?;
        Ok(request)
    }

    pub const fn identity(&self) -> StructuralModalResponseSpectrumAssemblyRequestIdentity {
        self.identity
    }
    pub fn scope(&self) -> &StructuralModalResponseSpectrumScope {
        &self.scope
    }
    pub fn entries(&self) -> &[StructuralModalResponseSpectrumDirectionResult] {
        &self.entries
    }

    pub fn assemble(&self) -> AnalysisExecutionResult<ModalResponseSpectrumResult> {
        self.validate()?;
        let spectra = self
            .entries
            .iter()
            .map(|entry| entry.request().spectrum().clone())
            .collect::<Vec<_>>();
        let numerical = self
            .entries
            .iter()
            .map(|entry| entry.numerical().clone())
            .collect::<Vec<_>>();
        assemble_modal_response_spectrum_direction_results(self.scope.basis(), &spectra, &numerical)
            .map_err(|error| invalid(error.to_string()))
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        let ordinals = self
            .entries
            .iter()
            .map(|entry| entry.request().ordinal())
            .collect::<BTreeSet<_>>();
        if self.schema_version != 1
            || self.entries.len() != self.scope.spectrum_ids().len()
            || ordinals.len() != self.entries.len()
            || self.entries.iter().enumerate().any(|(ordinal, entry)| {
                entry.validate().is_err()
                    || entry.request().ordinal() != ordinal
                    || entry.request().scope() != &self.scope
            })
            || self.identity != self.derive_identity()?
        {
            return Err(invalid(
                "response-spectrum assembly request is incomplete, unordered, or inconsistent",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            RESULT_PREFIX,
            "structural-modal-response-spectrum-assembly-request",
            self,
            RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            RESULT_PREFIX,
            "structural-modal-response-spectrum-assembly-request",
            bytes,
            RESULT_LIMITS,
        )?)
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<StructuralModalResponseSpectrumAssemblyRequestIdentity> {
        StructuralModalResponseSpectrumAssemblyRequestIdentity::derive(
            "structural-modal-response-spectrum-assembly-request",
            &(
                self.schema_version,
                self.scope.identity(),
                self.entries
                    .iter()
                    .map(StructuralModalResponseSpectrumDirectionResult::identity)
                    .collect::<Vec<_>>(),
            ),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralModalResponseSpectrumSolverResultWire"
)]
pub struct StructuralModalResponseSpectrumSolverResult {
    schema_version: u16,
    assembly: StructuralModalResponseSpectrumAssemblyRequest,
    numerical: ModalResponseSpectrumResult,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalResponseSpectrumSolverResultWire {
    schema_version: u16,
    assembly: StructuralModalResponseSpectrumAssemblyRequest,
    numerical: ModalResponseSpectrumResult,
}

impl TryFrom<StructuralModalResponseSpectrumSolverResultWire>
    for StructuralModalResponseSpectrumSolverResult
{
    type Error = AnalysisExecutionError;

    fn try_from(
        wire: StructuralModalResponseSpectrumSolverResultWire,
    ) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            assembly: wire.assembly,
            numerical: wire.numerical,
        };
        result.validate()?;
        Ok(result)
    }
}

impl StructuralModalResponseSpectrumSolverResult {
    pub fn new(
        workload: &StructuralModalResponseSpectrumWorkload,
        assembly: StructuralModalResponseSpectrumAssemblyRequest,
    ) -> AnalysisExecutionResult<Self> {
        assembly.scope().validate_for(workload)?;
        let numerical = assembly.assemble()?;
        let result = Self {
            schema_version: 1,
            assembly,
            numerical,
        };
        result.validate_for(workload)?;
        Ok(result)
    }

    pub fn assembly(&self) -> &StructuralModalResponseSpectrumAssemblyRequest {
        &self.assembly
    }
    pub fn numerical(&self) -> &ModalResponseSpectrumResult {
        &self.numerical
    }

    pub fn validate_for(
        &self,
        workload: &StructuralModalResponseSpectrumWorkload,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        self.assembly.scope().validate_for(workload)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.assembly.validate()?;
        if self.schema_version != 1 || self.numerical != self.assembly.assemble()? {
            return Err(invalid(
                "response-spectrum solver result differs from its direction assembly",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            RESULT_PREFIX,
            "structural-modal-response-spectrum-result",
            self,
            RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            RESULT_PREFIX,
            "structural-modal-response-spectrum-result",
            bytes,
            RESULT_LIMITS,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralModalResponseSpectrumNumericalReceiptWire"
)]
pub struct StructuralModalResponseSpectrumNumericalReceipt {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    source_payload_digest: Digest,
    spectrum_count: usize,
    eigenmode_contribution_count: usize,
    residual_vector_contribution_count: usize,
    minimum_achieved_mass_ratio: f64,
    maximum_response_magnitude: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalResponseSpectrumNumericalReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    source_payload_digest: Digest,
    spectrum_count: usize,
    eigenmode_contribution_count: usize,
    residual_vector_contribution_count: usize,
    minimum_achieved_mass_ratio: f64,
    maximum_response_magnitude: f64,
}

impl TryFrom<StructuralModalResponseSpectrumNumericalReceiptWire>
    for StructuralModalResponseSpectrumNumericalReceipt
{
    type Error = AnalysisExecutionError;

    fn try_from(
        wire: StructuralModalResponseSpectrumNumericalReceiptWire,
    ) -> Result<Self, Self::Error> {
        let receipt = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_workload: wire.source_workload,
            source_result: wire.source_result,
            source_payload_digest: wire.source_payload_digest,
            spectrum_count: wire.spectrum_count,
            eigenmode_contribution_count: wire.eigenmode_contribution_count,
            residual_vector_contribution_count: wire.residual_vector_contribution_count,
            minimum_achieved_mass_ratio: wire.minimum_achieved_mass_ratio,
            maximum_response_magnitude: wire.maximum_response_magnitude,
        };
        receipt.validate_source()?;
        Ok(receipt)
    }
}

impl StructuralModalResponseSpectrumNumericalReceipt {
    pub fn new(
        source_workload: AnalysisWorkloadIdentity,
        result: &StructuralModalResponseSpectrumSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate()?;
        let source_payload_digest = Digest::sha256(result.canonical_bytes()?);
        let source_result = SolverResultIdentity::new(source_workload, source_payload_digest)?;
        let contributions = result.numerical().contributions();
        let mut receipt = Self {
            schema_version: 1,
            identity: NumericalReceiptIdentity::new(source_result, Digest::sha256([]))?,
            source_workload,
            source_result,
            source_payload_digest,
            spectrum_count: contributions.len(),
            eigenmode_contribution_count: contributions
                .iter()
                .flat_map(|entry| &entry.modes)
                .filter(|mode| mode.kind == ModalResponseSpectrumModeKind::Eigenmode)
                .count(),
            residual_vector_contribution_count: contributions
                .iter()
                .flat_map(|entry| &entry.modes)
                .filter(|mode| mode.kind == ModalResponseSpectrumModeKind::ResidualVector)
                .count(),
            minimum_achieved_mass_ratio: contributions
                .iter()
                .map(|entry| entry.truncation.achieved_cumulative_mass_ratio)
                .fold(f64::INFINITY, f64::min),
            maximum_response_magnitude: result
                .numerical()
                .combined_response()
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
        };
        receipt.identity = receipt.derive_identity()?;
        receipt.validate_source()?;
        Ok(receipt)
    }

    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }
    pub const fn source_result(&self) -> SolverResultIdentity {
        self.source_result
    }

    pub const fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.source_workload
    }

    pub const fn minimum_achieved_mass_ratio(&self) -> f64 {
        self.minimum_achieved_mass_ratio
    }

    pub fn validate_against(
        &self,
        source_result: SolverResultIdentity,
        result: &StructuralModalResponseSpectrumSolverResult,
    ) -> AnalysisExecutionResult<()> {
        if self != &Self::new(self.source_workload, result)? || self.source_result != source_result
        {
            return Err(AnalysisExecutionError::Identity(
                "response-spectrum receipt differs from its source result",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_source()?;
        Ok(encode_contract(
            RESULT_PREFIX,
            "structural-modal-response-spectrum-receipt",
            self,
            RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            RESULT_PREFIX,
            "structural-modal-response-spectrum-receipt",
            bytes,
            RESULT_LIMITS,
        )?)
    }

    pub fn validate_source(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.spectrum_count == 0
            || !self.minimum_achieved_mass_ratio.is_finite()
            || !(0.0..=1.0 + 1.0e-10).contains(&self.minimum_achieved_mass_ratio)
            || !self.maximum_response_magnitude.is_finite()
            || self.maximum_response_magnitude < 0.0
            || self.source_result
                != SolverResultIdentity::new(self.source_workload, self.source_payload_digest)?
            || self.identity != self.derive_identity()?
        {
            return Err(invalid("response-spectrum numerical receipt is invalid"));
        }
        Ok(())
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
        let evidence = Digest::sha256(encode_contract(
            RESULT_PREFIX,
            "structural-modal-response-spectrum-receipt-evidence",
            &(
                self.spectrum_count,
                self.eigenmode_contribution_count,
                self.residual_vector_contribution_count,
                self.minimum_achieved_mass_ratio,
                self.maximum_response_magnitude,
            ),
            RESULT_LIMITS,
        )?);
        NumericalReceiptIdentity::new(self.source_result, evidence)
    }
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}

/// Compact terminal semantic state for downstream consumers and publication.
///
/// Direction-level modal contributions remain in the solver result. The state carries only the
/// final statistical envelope and the exact scope needed to interpret it, avoiding a second copy
/// of every direction artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralResponseEnvelope {
    scope: StructuralModalResponseSpectrumScope,
    response: SpectrumResponse,
    semantics: ResponseSpectrumResultSemantics,
    values: Vec<f64>,
}

impl StructuralResponseEnvelope {
    pub fn from_result(
        result: &StructuralModalResponseSpectrumSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate()?;
        let envelope = Self {
            scope: result.assembly().scope().clone(),
            response: result.numerical().response(),
            semantics: result.numerical().semantics(),
            values: result.numerical().combined_response().to_vec(),
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn scope(&self) -> &StructuralModalResponseSpectrumScope {
        &self.scope
    }

    pub const fn semantics(&self) -> ResponseSpectrumResultSemantics {
        self.semantics
    }

    pub const fn response(&self) -> SpectrumResponse {
        self.response
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn validate_for(
        &self,
        result: &StructuralModalResponseSpectrumSolverResult,
    ) -> AnalysisExecutionResult<()> {
        if self != &Self::from_result(result)? {
            return Err(AnalysisExecutionError::Identity(
                "structural response envelope differs from its solver result",
            ));
        }
        Ok(())
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        if self.semantics != ResponseSpectrumResultSemantics::StatisticalComponentMagnitudeEnvelope
            || self.values.len() != self.scope.physical_dof_count()
            || self
                .values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid("structural response envelope is not canonical"));
        }
        Ok(())
    }
}
