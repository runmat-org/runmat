//! Typed callers and presentation requests for one network executor.

use runmat_value::{ObjectInstance, Value};

use super::host::network::HostTrace;
use crate::builtins::deep_learning::autodiff::ResidentTapeAdmission;
use crate::builtins::deep_learning::model::{
    gpu::mixed::ResidentTrace, plan::axes::AxisPlan, state::StateUpdate,
};

pub(in crate::builtins::deep_learning) enum NetworkTrace {
    Host(HostTrace),
    Resident {
        trace: ResidentTrace,
        admission: ResidentTapeAdmission,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum NetworkCaller {
    Forward,
    Predict,
}

impl NetworkCaller {
    pub(in crate::builtins::deep_learning) const fn function(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Predict => "predict",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum NetworkOutputLayout {
    Preserve,
    ObservationMajor,
    /// Preserve the input axis order if the network preserves rank; otherwise
    /// present batch before the remaining axes for unformatted dlarray data.
    MatchInputRank,
}

impl NetworkOutputLayout {
    pub(super) const fn resolve(self, input_rank: usize, output_rank: usize) -> Self {
        match self {
            Self::MatchInputRank if input_rank == output_rank => Self::Preserve,
            Self::MatchInputRank => Self::ObservationMajor,
            concrete => concrete,
        }
    }
}

pub(in crate::builtins::deep_learning) struct NetworkExecutionRequest<'a> {
    pub runtime: Option<&'a crate::context::RuntimeContext>,
    pub network: &'a ObjectInstance,
    pub input: &'a Value,
    pub caller: NetworkCaller,
    pub output_layout: NetworkOutputLayout,
    /// Canonical invocation arity; absent only for direct embedding callers.
    pub requested_outputs: Option<usize>,
}

/// `trace` retains the executor's original physical output geometry. The
/// presentation transform applies to `data` and `output_axes`; tape recording
/// must retain both axis plans so its reverse pass inverts that presentation.
pub(in crate::builtins::deep_learning) struct NetworkExecutionOutcome {
    pub data: Value,
    pub output_axes: AxisPlan,
    pub state_updates: Vec<StateUpdate>,
    pub trace: Option<NetworkTrace>,
    /// Invocation authority retained only through public presentation.
    pub runtime: Option<crate::context::RuntimeContext>,
    /// Process-local observation, completed only after public presentation.
    pub placement: Option<super::placement::PendingWholeWorkObservation>,
}

#[cfg(test)]
mod tests {
    use super::NetworkOutputLayout;

    #[test]
    fn unformatted_output_layout_depends_on_rank_not_tape_state() {
        assert_eq!(
            NetworkOutputLayout::MatchInputRank.resolve(3, 3),
            NetworkOutputLayout::Preserve
        );
        assert_eq!(
            NetworkOutputLayout::MatchInputRank.resolve(3, 2),
            NetworkOutputLayout::ObservationMajor
        );
    }
}
