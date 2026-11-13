//! Shared structural-result identity check for downstream consumers of completed solves.

use crate::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, SolverResultIdentity,
    StructuralLinearWorkloadResult,
};
use runmat_execution::Digest;

pub(crate) fn decode_structural_result_source(
    host: &AnalysisHostWorkload,
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<(SolverResultIdentity, StructuralLinearWorkloadResult)> {
    host.validate()?;
    if host.workload.kind != AnalysisWorkKind::Solve
        || objects.manifest.kind != AnalysisNumericalObjectKind::SolverResult
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural result source requires a structural solve and solver result".into(),
        ));
    }
    let payload = objects.payload();
    if Digest::sha256(&payload) != objects.manifest.payload_digest {
        return Err(AnalysisBridgeError::Identity(
            "structural source payload differs from its manifest",
        ));
    }
    let result = StructuralLinearWorkloadResult::from_canonical_bytes(&payload)?;
    let identity = result.result_identity(host.workload.identity)?;
    result.validate_thermal_source(host.workload.step, identity)?;
    if objects.manifest.logical_identity != identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "structural result identity differs from its source workload and content",
        ));
    }
    if let Some(source) = result
        .linearization
        .as_ref()
        .and_then(|value| value.thermal_source())
    {
        let transition = source.transition();
        let mut states = host
            .workload
            .inputs
            .iter()
            .filter(|input| input.kind == AnalysisNumericalObjectKind::SemanticState);
        let state = states.next().ok_or(AnalysisBridgeError::Identity(
            "thermal structural source has no admitted semantic state",
        ))?;
        if states.next().is_some() || state.logical_identity != transition.source_state.digest() {
            return Err(AnalysisBridgeError::Identity(
                "thermal structural source differs from its admitted step or state",
            ));
        }
    }
    Ok((identity, result))
}
