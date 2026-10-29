//! Detached profiles authenticate the retained acoustic response result, without rerunning physics.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::execution::{
    AcousticResponseSolverResult, AcousticResponseWorkload, AnalysisNumericalObjectKind,
    StatePublicationProfiles,
};

pub fn acoustic_response_state_publication_profiles(
    workload: &AcousticResponseWorkload,
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StatePublicationProfiles> {
    let result = decode_acoustic_response_source(objects)?;
    if result.source_workload() != workload.identity()? {
        return Err(AnalysisBridgeError::Identity(
            "acoustic response profile workload differs",
        ));
    }
    Ok(result.state_publication_profiles(objects.manifest.payload_digest)?)
}

pub(crate) fn decode_acoustic_response_source(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<AcousticResponseSolverResult> {
    if objects.manifest.kind != AnalysisNumericalObjectKind::SolverResult {
        return Err(AnalysisBridgeError::Identity(
            "acoustic response source must be a solver result",
        ));
    }
    let result = AcousticResponseSolverResult::from_canonical_bytes(&objects.payload())?;
    if result.source_result_identity()?.digest() != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "acoustic response source logical identity differs",
        ));
    }
    Ok(result)
}
