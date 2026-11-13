//! Producer-side state publication profiles borrow already validated numerical results.
use crate::{AnalysisBridgeError, AnalysisBridgeResult};
use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_analysis_fea::execution::{
    structural_state_publication_requests, SemanticStatePublicationProfileSource as Profile,
    SemanticStatePublicationSource as Source, SequenceIdentity, SolverResultIdentity,
    StatePublicationProfileEntry, StatePublicationProfiles, StepIdentity,
    StructuralLinearWorkloadResult, StructuralModalResponseSpectrumSolverResult,
    StructuralModalSolverResult, StructuralRandomResponseSolverResult,
};
use runmat_execution::Digest;

#[cfg(test)]
mod tests;

pub fn structural_state_profiles_output_facts(
    model: &AnalysisModel,
    step: &ProcedureStep,
    sources: &[Source],
    topology: Digest,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    crate::projected_output_resource_facts(
        runmat_analysis_fea::execution::structural_state_publication_profiles_encoding_profile(
            model, step, sources, topology,
        )?
        .encoded_bytes_upper_bound,
    )
}

pub(crate) fn validate_profiles(
    objects: &crate::PreparedAnalysisObjects,
    expected: StatePublicationProfiles,
) -> AnalysisBridgeResult<StatePublicationProfiles> {
    let actual = StatePublicationProfiles::from_canonical_bytes(&objects.payload())?;
    if objects.manifest.logical_identity != actual.identity() || actual != expected {
        return Err(AnalysisBridgeError::Identity(
            "structural publication profiles differ from source-bound canonical projection",
        ));
    }
    Ok(actual)
}

pub enum StructuralStateProfileResult<'a> {
    Linear(&'a StructuralLinearWorkloadResult),
    Modal(&'a StructuralModalSolverResult),
    ResponseSpectrum(&'a StructuralModalResponseSpectrumSolverResult),
    RandomResponse(&'a StructuralRandomResponseSolverResult),
}

/// The caller's normal solver-result validation must precede this projection. No result is
/// decoded or gathered solely for profiling, and no numerical arrays are cloned.
pub fn structural_state_publication_profiles(
    model: &AnalysisModel,
    step: &ProcedureStep,
    source_result: SolverResultIdentity,
    source_payload_digest: Digest,
    result: StructuralStateProfileResult<'_>,
) -> AnalysisBridgeResult<StatePublicationProfiles> {
    let (topology, sources): (_, &[Source]) = match &result {
        StructuralStateProfileResult::Linear(value) => (
            value.constraint_result.topology.discretization_digest(),
            &[
                Source::StructuralPrimaryFields,
                Source::StructuralLinearization,
            ],
        ),
        StructuralStateProfileResult::Modal(value) => {
            (value.topology_digest(), &[Source::ModalBasis])
        }
        StructuralStateProfileResult::ResponseSpectrum(value) => (
            value.assembly().scope().topology_digest(),
            &[Source::StructuralResponseEnvelope],
        ),
        StructuralStateProfileResult::RandomResponse(value) => (
            value.assembly().scope().topology_digest(),
            &[Source::StructuralRandomStatistics],
        ),
    };
    let requests =
        structural_state_publication_requests(model, step, sources, topology, source_result)?;
    let entries = requests
        .into_iter()
        .map(|request| {
            let source = match (&result, request.source) {
                (StructuralStateProfileResult::Linear(value), Source::StructuralPrimaryFields) => {
                    Profile::StructuralPrimaryFields(&value.run.fields)
                }
                (StructuralStateProfileResult::Linear(value), Source::StructuralLinearization) => {
                    Profile::StructuralLinearization(value.linearization.as_ref().ok_or(
                        AnalysisBridgeError::Identity(
                            "declared linearization is absent from result",
                        ),
                    )?)
                }
                (StructuralStateProfileResult::Modal(value), Source::ModalBasis) => {
                    Profile::ModalBasis(value.basis())
                }
                (
                    StructuralStateProfileResult::ResponseSpectrum(value),
                    Source::StructuralResponseEnvelope,
                ) => Profile::StructuralResponseEnvelope(value),
                (
                    StructuralStateProfileResult::RandomResponse(value),
                    Source::StructuralRandomStatistics,
                ) => Profile::StructuralRandomStatistics(value),
                _ => {
                    return Err(AnalysisBridgeError::Identity(
                        "publication source differs from producer",
                    ))
                }
            };
            let profile = request.encoding_profile(source)?;
            Ok(StatePublicationProfileEntry::new(request, profile)?)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    Ok(StatePublicationProfiles::new(
        StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&model.sequence)?,
            step,
        )?,
        source_result,
        topology,
        source_payload_digest,
        entries,
    )?)
}
