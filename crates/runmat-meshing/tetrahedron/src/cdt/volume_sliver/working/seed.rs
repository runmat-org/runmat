//! Shape mutation evidence wraps a shared revision-bound edit seed.
use super::*;
use crate::cdt::volume_state::VolumeSeed;

pub(in crate::cdt::volume_sliver) struct WorkingSeed {
    state: VolumeSeed,
    pub(in crate::cdt::volume_sliver) mutations: Vec<DelaunayVolumeSliverMutation>,
}
impl std::ops::Deref for WorkingSeed {
    type Target = VolumeSeed;
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}
pub(in crate::cdt::volume_sliver) trait CaptureSeed {
    fn capture(&self, mutations: Vec<DelaunayVolumeSliverMutation>) -> WorkingSeed;
}
impl CaptureSeed for WorkingTrial<'_, '_> {
    fn capture(&self, mutations: Vec<DelaunayVolumeSliverMutation>) -> WorkingSeed {
        WorkingSeed {
            state: self.capture_edits(),
            mutations,
        }
    }
}

impl WorkingSeed {
    pub(in crate::cdt::volume_sliver) fn retain_better(
        best: &mut Option<Self>,
        candidate: Self,
        policy: SliverSearchPolicy,
    ) {
        if best.as_ref().is_none_or(|current| {
            candidate.compare_policy(current, policy) == std::cmp::Ordering::Less
        }) {
            *best = Some(candidate);
        }
    }

    pub(in crate::cdt::volume_sliver) fn compare_policy(
        &self,
        other: &Self,
        policy: SliverSearchPolicy,
    ) -> std::cmp::Ordering {
        if matches!(
            policy.refinement,
            admission::CandidateRefinementPolicy::AllowForCoupledConvergence
        ) {
            self.state
                .compare_joint_spectrum(&other.state)
                .then_with(|| self.state.compare_spectrum(&other.state, policy.objective))
        } else {
            self.state.compare_spectrum(&other.state, policy.objective)
        }
    }
}
