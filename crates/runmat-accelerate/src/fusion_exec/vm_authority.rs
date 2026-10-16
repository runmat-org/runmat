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
}
