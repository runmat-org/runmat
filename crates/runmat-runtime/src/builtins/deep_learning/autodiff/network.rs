mod executed;
mod resident;

pub(in crate::builtins::deep_learning) use executed::{
    annotate_outputs as annotate_network_outputs, record_execution as record_network_execution,
};
pub(super) use executed::{reverse as reverse_execution, NetworkOutput, NetworkTapeRecord};
pub(in crate::builtins::deep_learning) use resident::{
    admit_parents as admit_resident_tape_parents, record as record_resident_network_execution,
    ResidentTapeAdmission,
};
pub(super) use resident::{
    plan_reverse as plan_resident_reverse, reverse as reverse_resident_execution,
    ResidentReversePlan, ResidentTapeRecord,
};
