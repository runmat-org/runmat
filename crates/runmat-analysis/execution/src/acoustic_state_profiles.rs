//! Source-bound acoustic publication metadata; numerical fields stay in the solver result.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::execution::{
    AcousticHarmonicSolverResult, AcousticHarmonicWorkload, AnalysisNumericalObjectKind,
    StatePublicationProfiles,
};

pub fn acoustic_state_publication_profiles(
    workload: &AcousticHarmonicWorkload,
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StatePublicationProfiles> {
    let result = decode_acoustic_source(objects)?;
    if result.source_workload() != workload.identity()? {
        return Err(AnalysisBridgeError::Identity(
            "acoustic profile workload differs",
        ));
    }
    Ok(result.state_publication_profiles(objects.manifest.payload_digest)?)
}

pub(crate) fn decode_acoustic_source(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<AcousticHarmonicSolverResult> {
    if objects.manifest.kind != AnalysisNumericalObjectKind::SolverResult {
        return Err(AnalysisBridgeError::Identity(
            "acoustic source must be a solver result",
        ));
    }
    let result = AcousticHarmonicSolverResult::from_canonical_bytes(&objects.payload())?;
    if result.source_result_identity()?.digest() != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "acoustic source logical identity differs",
        ));
    }
    Ok(result)
}
