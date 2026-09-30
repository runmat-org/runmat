//! Executable deep-learning kernels. Public dispatch and network admission
//! remain with `model`; this subtree owns numerical execution only.

pub(in crate::builtins::deep_learning) mod host;

mod layout;
mod network;
mod placement;
mod request;

pub(in crate::builtins::deep_learning) use layout::present_host;
pub(in crate::builtins::deep_learning) use network::execute_network;
pub(in crate::builtins::deep_learning) use request::{
    NetworkCaller, NetworkExecutionOutcome, NetworkExecutionRequest, NetworkOutputLayout,
    NetworkTrace,
};
