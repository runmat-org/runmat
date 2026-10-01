//! Borrowed original Maxwell state wire projection, without field copies or solve.
use super::*;
use crate::execution::MaxwellHarmonicSolverResult;
#[derive(Serialize)]
struct MaxwellProjection<'a> {
    result: &'a MaxwellHarmonicSolverResult,
}
pub(super) fn profile(
    request: &SemanticStatePublicationRequest,
    result: &MaxwellHarmonicSolverResult,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    result.validate_publication_request(request)?;
    profile_state_payload(
        request,
        &Tagged {
            kind: "maxwell_harmonic",
            key: "state",
            value: &MaxwellProjection { result },
        },
    )
}
