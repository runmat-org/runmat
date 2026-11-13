//! Detached profiles authenticate the retained magnetic result, without rerunning physics.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, MagnetostaticSolverResult, MagnetostaticWorkload,
    StatePublicationProfiles,
};

pub fn magnetic_state_publication_profiles(
    workload: &MagnetostaticWorkload,
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StatePublicationProfiles> {
    let result = decode_magnetic_source(objects)?;
    if result.source_workload() != workload.identity()? {
        return Err(AnalysisBridgeError::Identity(
            "magnetic profile workload differs",
        ));
    }
    Ok(result.state_publication_profiles(objects.manifest.payload_digest)?)
}

pub(crate) fn decode_magnetic_source(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<MagnetostaticSolverResult> {
    if objects.manifest.kind != AnalysisNumericalObjectKind::SolverResult {
        return Err(AnalysisBridgeError::Identity(
            "magnetic source must be a solver result",
        ));
    }
    let result = MagnetostaticSolverResult::from_canonical_bytes(&objects.payload())?;
    if result.source_result_identity()?.digest() != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "magnetic source logical identity differs",
        ));
    }
    Ok(result)
}
