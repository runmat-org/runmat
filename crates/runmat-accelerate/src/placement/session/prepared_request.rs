//! A validated graph with only feedback-controlled candidate adjustments.

use std::collections::BTreeMap;

use runmat_execution::{Digest, ExecutionCandidateKind, PlacementPlanRequest};

use super::{apply_feedback_evidence, FeedbackCandidateKey, PlacementFeedbackSeries};

pub(in crate::placement) struct PreparedPlacementRequest {
    request: PlacementPlanRequest,
}

impl PreparedPlacementRequest {
    pub(in crate::placement) fn new(request: PlacementPlanRequest) -> Result<Self, &'static str> {
        request.validate()?;
        Ok(Self { request })
    }

    pub(in crate::placement) fn request(&self) -> &PlacementPlanRequest {
        &self.request
    }

    pub(super) fn apply_feedback(
        &mut self,
        signature_key: Digest,
        tick: u64,
        feedback: &BTreeMap<
            (Digest, FeedbackCandidateKey, ExecutionCandidateKind),
            PlacementFeedbackSeries,
        >,
    ) {
        apply_feedback_evidence(&mut self.request, signature_key, tick, feedback);
    }
}
