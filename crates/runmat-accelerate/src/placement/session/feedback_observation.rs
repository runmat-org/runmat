//! Revocable observation for a trusted host route whose execution is admitted
//! outside placement. This grants no execution/resource authority. It reuses
//! the existing profile/selection epoch and never creates a second profile.
use super::{PlacementPlanError, PlacementSession};
use runmat_execution::PlacementSignature;
use runmat_runtime::context::RuntimeWholeWorkObservationPermit;
use std::sync::Arc;

impl PlacementSession {
    pub fn feedback_observation_permit(
        &self,
        signature: &PlacementSignature,
    ) -> Result<Arc<dyn RuntimeWholeWorkObservationPermit>, PlacementPlanError> {
        signature.validate().map_err(|message| PlacementPlanError {
            code: "RunMat:Placement:InvalidFeedback",
            message: message.into(),
        })?;
        self.observation_permits
            .register(signature)
            .map_err(|error| PlacementPlanError {
                code: "RunMat:Placement:InvalidFeedback",
                message: error.to_string(),
            })
    }
}
