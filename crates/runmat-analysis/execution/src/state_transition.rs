use runmat_analysis_core::{AnalysisModel, ProcedureStepId, StateUse};
use runmat_analysis_fea::execution::{CanonicalSemanticState, CanonicalStateTransition};
use runmat_execution::Digest;

use crate::AnalysisBridgeResult;

/// Resolves one authored upstream state input into the immutable transition consumed by a
/// numerical procedure. Placement and task attempts cannot enter this contract.
pub fn bind_upstream_state_transition(
    model: &AnalysisModel,
    consumer_step_id: &ProcedureStepId,
    state: &CanonicalSemanticState,
    expected_use: StateUse,
    target_topology_digest: Digest,
) -> AnalysisBridgeResult<CanonicalStateTransition> {
    CanonicalStateTransition::bind_upstream(
        model,
        consumer_step_id,
        state,
        expected_use,
        target_topology_digest,
    )
    .map_err(Into::into)
}
