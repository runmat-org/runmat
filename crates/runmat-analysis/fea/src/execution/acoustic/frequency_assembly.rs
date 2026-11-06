//! Deterministic fan-in of original harmonic results; never invokes a solver.
use super::{AcousticFrequencyResult, AcousticFrequencyWorkload, AcousticHarmonicSolverResult};
use crate::execution::{
    AnalysisExecutionError, AnalysisExecutionResult, FrequencySamplePublicationManifest,
    FrequencySamplePublicationSample,
};
use crate::results::observation::ObservationArtifactManifest;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

const REQUEST_PREFIX: &[u8] = b"runmat-acoustic-frequency-assembly-request/v1\0";
const REQUEST_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Per-entry authored observation manifests, in exact frequency-plan order. Entries without
/// authored outputs carry `None`. The assembly publishes them beside each sample's fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "AssemblyRequestWire")]
pub struct AcousticFrequencyAssemblyRequest {
    schema_version: u16,
    identity: Digest,
    observations: Vec<Option<ObservationArtifactManifest>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AssemblyRequestWire {
    schema_version: u16,
    identity: Digest,
    observations: Vec<Option<ObservationArtifactManifest>>,
}

impl TryFrom<AssemblyRequestWire> for AcousticFrequencyAssemblyRequest {
    type Error = AnalysisExecutionError;
    fn try_from(wire: AssemblyRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            observations: wire.observations,
        };
        request.validate()?;
        Ok(request)
    }
}

impl AcousticFrequencyAssemblyRequest {
    pub fn new(
        observations: Vec<Option<ObservationArtifactManifest>>,
    ) -> AnalysisExecutionResult<Self> {
        let mut request = Self {
            schema_version: 1,
            identity: Digest::sha256([]),
            observations,
        };
        request.identity = request.derive_identity()?;
        request.validate()?;
        Ok(request)
    }
    pub const fn identity(&self) -> Digest {
        self.identity
    }
    pub fn observations(&self) -> &[Option<ObservationArtifactManifest>] {
        &self.observations
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.observations.is_empty()
            || self.identity != self.derive_identity()?
        {
            return Err(AnalysisExecutionError::Identity(
                "acoustic assembly request schema, inventory or identity differs",
            ));
        }
        for manifest in self.observations.iter().flatten() {
            manifest
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        }
        Ok(())
    }
    fn derive_identity(&self) -> AnalysisExecutionResult<Digest> {
        Ok(Digest::sha256(encode_contract(
            REQUEST_PREFIX,
            "acoustic-frequency-assembly-request-identity",
            &(self.schema_version, &self.observations),
            REQUEST_LIMITS,
        )?))
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            REQUEST_PREFIX,
            "acoustic-frequency-assembly-request",
            self,
            REQUEST_LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            REQUEST_PREFIX,
            "acoustic-frequency-assembly-request",
            bytes,
            REQUEST_LIMITS,
        )?)
    }
}

#[cfg(test)]
mod tests;

pub struct AcousticFrequencyAssemblyOutput {
    pub result: AcousticFrequencyResult,
    pub publication: FrequencySamplePublicationManifest,
    pub fields: Vec<crate::results::field::RecoveredFieldArtifact>,
}

pub fn assemble_acoustic_frequency_result(
    workload: AcousticFrequencyWorkload,
    mesh: &SolverMeshArtifact,
    entries: Vec<AcousticHarmonicSolverResult>,
    request: &AcousticFrequencyAssemblyRequest,
) -> AnalysisExecutionResult<AcousticFrequencyAssemblyOutput> {
    request.validate()?;
    let result = AcousticFrequencyResult::assemble(workload, mesh, entries)?;
    if request.observations().len() != result.results().len() {
        return Err(AnalysisExecutionError::Identity(
            "acoustic observation inventory differs from the frequency plan",
        ));
    }
    let mut original_fields = Vec::with_capacity(result.results().len());
    let samples = result
        .workload()
        .plan()
        .entries()
        .map_err(|error| crate::execution::AnalysisExecutionError::Invalid(error.to_string()))?
        .into_iter()
        .zip(result.results())
        .zip(request.observations())
        .map(|((entry, numerical), observations)| {
            let recovery_request = numerical.native_field_request_from_result()?;
            let fields = numerical.recover_native_fields_from_result(&recovery_request)?;
            let sample = FrequencySamplePublicationSample {
                entry,
                recovery_request,
                fields: fields.manifest().map_err(|error| {
                    crate::execution::AnalysisExecutionError::Invalid(error.to_string())
                })?,
                observations: observations.clone(),
            };
            original_fields.push(fields);
            Ok(sample)
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let publication = result.publication_manifest(samples)?;
    Ok(AcousticFrequencyAssemblyOutput {
        result,
        publication,
        fields: original_fields,
    })
}
