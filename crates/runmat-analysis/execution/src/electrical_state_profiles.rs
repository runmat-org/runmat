//! Detached profiles authenticate the retained electrical result, without rerunning physics.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, ElectricalStaticSolverResult, ElectricalStaticWorkload,
    StatePublicationProfiles,
};

pub fn electrical_state_publication_profiles(
    workload: &ElectricalStaticWorkload,
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StatePublicationProfiles> {
    let result = decode_electrical_source(objects)?;
    if result.source_workload() != workload.identity()? {
        return Err(AnalysisBridgeError::Identity(
            "electrical profile workload differs",
        ));
    }
    Ok(result.state_publication_profiles(objects.manifest.payload_digest)?)
}

pub(crate) fn decode_electrical_source(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<ElectricalStaticSolverResult> {
    if objects.manifest.kind != AnalysisNumericalObjectKind::SolverResult {
        return Err(AnalysisBridgeError::Identity(
            "electrical source must be a solver result",
        ));
    }
    let result = ElectricalStaticSolverResult::from_canonical_bytes(&objects.payload())?;
    if result.source_result_identity()?.digest() != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "electrical source logical identity differs",
        ));
    }
    Ok(result)
}
