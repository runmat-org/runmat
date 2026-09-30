mod executed;

pub(in crate::builtins::deep_learning) use executed::{
    annotate_outputs as annotate_network_outputs, record_execution as record_network_execution,
};
pub(super) use executed::{reverse as reverse_execution, NetworkOutput, NetworkTapeRecord};
