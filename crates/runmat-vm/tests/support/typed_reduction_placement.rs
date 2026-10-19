//! Reduction-specific selection inventory layered on the shared feedback owner.
#[path = "placement_feedback.rs"]
pub(crate) mod feedback;
pub(crate) type CapturingPlacement = feedback::FeedbackPlacement<CapturedReductionSelection>;

use runmat_execution::{
    CandidatePreparationState, ExecutionCandidateKind, PlacementPlanRequest, PlacementSignature,
};

pub(crate) struct CapturedReductionSelection {
    pub(crate) request: PlacementPlanRequest,
    pub(crate) signature: PlacementSignature,
    pub(crate) candidate: String,
    pub(crate) kind: ExecutionCandidateKind,
    pub(crate) preparation: CandidatePreparationState,
    pub(crate) device_id: u32,
}

impl feedback::SelectionCapture for CapturedReductionSelection {
    fn accepts(operation: &str) -> bool {
        operation == "fusion.reduction"
    }
    fn capture(
        request: PlacementPlanRequest,
        signature: PlacementSignature,
        candidate: String,
        kind: ExecutionCandidateKind,
        preparation: CandidatePreparationState,
        device_id: u32,
    ) -> Self {
        Self {
            request,
            signature,
            candidate,
            kind,
            preparation,
            device_id,
        }
    }
}
