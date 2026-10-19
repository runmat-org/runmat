//! VM entrypoints retain their grants while lending one execution authority.
//! A pre-effect
//! decline never carries completion, timing or host-readback rights.

use runmat_runtime::context::{VmFusionCompletionRight, VmFusionDeclineRight};

pub(super) enum VmFusionExecutionRight<'a> {
    Completion(&'a mut Option<VmFusionCompletionRight>),
    Decline(&'a VmFusionDeclineRight),
}

impl VmFusionExecutionRight<'_> {
    pub(super) fn as_selection(&self) -> Option<VmFusionSelectionRight<'_>> {
        match self {
            Self::Completion(right) => right.as_ref().map(VmFusionSelectionRight::Completion),
            Self::Decline(right) => Some(VmFusionSelectionRight::Decline(right)),
        }
    }

    pub(super) fn into_completion(self) -> Option<VmFusionCompletionRight> {
        match self {
            Self::Completion(right) => right.take(),
            Self::Decline(_) => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum VmFusionSelectionRight<'a> {
    Completion(&'a VmFusionCompletionRight),
    Decline(&'a VmFusionDeclineRight),
}

impl<'a> VmFusionSelectionRight<'a> {
    pub(super) fn completion(self) -> Option<&'a VmFusionCompletionRight> {
        match self {
            Self::Completion(right) => Some(right),
            Self::Decline(_) => None,
        }
    }

    pub(super) fn decline(self) -> Option<&'a VmFusionDeclineRight> {
        match self {
            Self::Completion(_) => None,
            Self::Decline(right) => Some(right),
        }
    }

    pub(super) fn certified_decline(
        self,
        request: &super::FusionExecutionRequest<'_>,
    ) -> Option<&'a VmFusionDeclineRight> {
        let right = self.decline()?;
        let runtime = request.runtime.as_ref()?;
        (right.is_current(runtime)
            && right.matches_group(
                request.plan.group.id,
                request.plan.group.span.start,
                request.plan.group.span.end,
            ))
        .then_some(right)
    }
}
