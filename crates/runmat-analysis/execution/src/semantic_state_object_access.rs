use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, CanonicalSemanticState};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ObjectNamespace;

use crate::{
    prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects,
};

pub fn prepare_semantic_state_objects(
    state: &CanonicalSemanticState,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    state.validate()?;
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::SemanticState,
        state.identity.digest(),
        &state.canonical_bytes()?,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn decode_prepared_semantic_state(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<CanonicalSemanticState> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::SemanticState {
        return Err(AnalysisBridgeError::Invalid(
            "expected a semantic state object".into(),
        ));
    }
    let state = CanonicalSemanticState::from_canonical_bytes(&prepared.payload())?;
    if prepared.manifest.logical_identity != state.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "semantic state object has the wrong logical identity",
        ));
    }
    Ok(state)
}
