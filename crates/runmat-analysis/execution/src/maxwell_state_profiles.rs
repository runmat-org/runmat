//! Original retained complex source profiles, with no numerical re-execution.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, MaxwellHarmonicSolverResult, MaxwellHarmonicWorkload,
    StatePublicationProfiles,
};
use runmat_execution::Digest;
pub fn maxwell_state_publication_profiles(
    workload: &MaxwellHarmonicWorkload,
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StatePublicationProfiles> {
    let result = decode_maxwell_source(objects)?;
    if result.source_workload() != workload.identity()? {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell profile workload differs",
        ));
    }
    Ok(result.state_publication_profiles(objects.manifest.payload_digest)?)
}
pub(crate) fn decode_maxwell_source(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<MaxwellHarmonicSolverResult> {
    if objects.manifest.kind != AnalysisNumericalObjectKind::SolverResult {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell state source must be original solver result",
        ));
    }
    let bytes = objects.payload();
    if Digest::sha256(&bytes) != objects.manifest.payload_digest {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell source payload digest differs",
        ));
    }
    let result = MaxwellHarmonicSolverResult::from_canonical_bytes(&bytes)?;
    if result.identity()?.digest() != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell source logical identity differs",
        ));
    }
    Ok(result)
}
