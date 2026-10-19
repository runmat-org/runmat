//! Deterministic fan-in of original harmonic results; never invokes a solver.
use super::{AcousticFrequencyResult, AcousticFrequencyWorkload, AcousticHarmonicSolverResult};
use crate::execution::{
    AnalysisExecutionResult, FrequencySamplePublicationManifest, FrequencySamplePublicationSample,
};
use runmat_meshing_core::SolverMeshArtifact;

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
) -> AnalysisExecutionResult<AcousticFrequencyAssemblyOutput> {
    let result = AcousticFrequencyResult::assemble(workload, mesh, entries)?;
    let mut original_fields = Vec::with_capacity(result.results().len());
    let samples = result
        .workload()
        .plan()
        .entries()
        .map_err(|error| crate::execution::AnalysisExecutionError::Invalid(error.to_string()))?
        .into_iter()
        .zip(result.results())
        .map(|(entry, numerical)| {
            let recovery_request = numerical.native_field_request_from_result()?;
            let fields = numerical.recover_native_fields_from_result(&recovery_request)?;
            let sample = FrequencySamplePublicationSample {
                entry,
                recovery_request,
                fields: fields.manifest().map_err(|error| {
                    crate::execution::AnalysisExecutionError::Invalid(error.to_string())
                })?,
                observations: None,
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
