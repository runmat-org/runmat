//! Exact VM input occurrences. Only the installed bytecode owner constructs
//! this transcript; a reusable compute plan or matching buffer is insufficient.
use crate::graph::{ValueId, VarBinding};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VmHostLoadUse {
    pub pc: usize,
    pub binding: VarBinding,
    pub value_id: ValueId,
}

#[derive(Clone, Debug)]
pub struct VmHostInputUses {
    pub entry: Vec<(ValueId, usize)>,
    pub loads: Vec<VmHostLoadUse>,
}
