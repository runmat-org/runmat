//! Typed callers and presentation requests for one network executor.

use runmat_value::{ObjectInstance, Value};

use super::host::network::HostTrace;
use crate::builtins::deep_learning::model::{plan::axes::AxisPlan, state::StateUpdate};

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
}

pub(in crate::builtins::deep_learning) struct NetworkExecutionRequest<'a> {
    pub network: &'a ObjectInstance,
    pub input: &'a Value,
    pub caller: NetworkCaller,
    pub output_layout: NetworkOutputLayout,
}

/// `trace` retains the executor's original physical output geometry. The
/// presentation transform applies only to `data` and `output_axes`; callers
/// must not use a reordered prediction as a tape node without the inverse
/// transform. `forward` requests Preserve and therefore has matching axes.
pub(in crate::builtins::deep_learning) struct NetworkExecutionOutcome {
    pub data: Value,
    pub output_axes: AxisPlan,
    pub state_updates: Vec<StateUpdate>,
    pub trace: Option<HostTrace>,
}
