//! Checked preparation geometry for the liveness planner's physical owners.

use super::{HostOperationLiveness, HostRealSlot};
use crate::graph::ValueId;
use runmat_runtime::context::VmFusionHostRealSlotSpec;

pub(super) fn preparation_bytes(
    value_ids_upper: usize,
    operations: usize,
    max_rank: usize,
) -> Option<u64> {
    fn vector<T>(count: usize) -> Option<u64> {
        count
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
    }
    let mut total = 0_u64;
    let mut add = |bytes: u64| {
        total = total.checked_add(bytes)?;
        Some(())
    };
    // Alias IDs plus parent, producer, pin, remaining-use and occupied-slot
    // tables coexist with the completed slot/operation plan at return.
    add(vector::<ValueId>(value_ids_upper)?)?;
    add(vector::<usize>(value_ids_upper)?)?;
    add(vector::<Option<usize>>(value_ids_upper)?)?;
    add(vector::<bool>(value_ids_upper)?)?;
    add(vector::<usize>(value_ids_upper)?)?;
    add(vector::<Option<usize>>(value_ids_upper)?)?;
    add(vector::<HostRealSlot>(operations)?)?;
    add(vector::<bool>(operations)?)?;
    add(vector::<HostOperationLiveness>(operations)?)?;
    // The borrowed Runtime slot projection coexists with the retained plan.
    add(vector::<VmFusionHostRealSlotSpec<'static>>(operations)?)?;
    // Every retained slot owns one shape backing. While coloring an operation,
    // one additional candidate shape can coexist before reuse is decided.
    add(vector::<usize>(
        operations
            .checked_add(usize::from(operations != 0))?
            .checked_mul(max_rank)?,
    )?)?;
    Some(total)
}
