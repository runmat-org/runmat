//! One prepared reconciliation of a delegate's named workspace snapshot.
//!
//! The input is a set of named bindings, not an assignment event log. Sort it
//! in place, reject duplicate names, then reserve every frame collection once
//! before changing any visible slot. The single phase token excludes reentry
//! throughout preparation and the allocation-free commit.

use super::frame::{with_frame, with_frame_mut_unlocked, WorkspaceFrameId};
use super::frame_assignment::{add, copied_map, copied_set, failed, vector_growth};
use super::frame_metadata::{FrameMap, FrameMutationToken, FrameSet, FrameVec};
use super::{SlotLifecycle, WorkspaceState};
use crate::interpreter::vars::InterpreterSlot;
use runmat_runtime::context::host_allocation::{
    frame_hash_table_retained_bytes, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::WorkspaceName;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

fn unavailable() -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:Workspace:FrameUnavailable",
        "workspace frame is closed or a frame mutation is active",
    )
}

fn invalid_snapshot() -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:Workspace:DuplicateName",
        "delegate workspace snapshot contains a duplicate name",
    )
}

fn allocation(error: HostAllocationError) -> RuntimeError {
    super::frame_allocation_error(error)
}

fn target(current: usize, extra: usize) -> Result<usize, HostAllocationError> {
    current
        .checked_add(extra)
        .ok_or(HostAllocationError::Overflow)
}

fn bytes<T>(count: usize) -> Result<u64, HostAllocationError> {
    vector_growth::<T>(count, 0).map(|quote| quote.unwrap_or(0))
}

fn table<K: Eq + std::hash::Hash, V>(
    target: usize,
    capacity: usize,
) -> Result<Option<u64>, HostAllocationError> {
    (target > capacity)
        .then(|| frame_hash_table_retained_bytes::<K, V>(target))
        .transpose()
}

mod geometry;
use geometry::{Growth, Replacements};

pub(super) struct PreparedFrameReconcile {
    frame: WorkspaceFrameId,
    token: FrameMutationToken,
    context: Option<RuntimeContext>,
    receipt: Option<HostAllocationReceipt>,
    values: Vec<(String, Value)>,
    incoming: FrameVec<Option<(WorkspaceName, usize)>>,
    removed: FrameVec<(WorkspaceName, usize)>,
    retired_names: Option<FrameVec<WorkspaceName>>,
    retired_values: Option<FrameVec<InterpreterSlot>>,
    replacements: Replacements,
}

mod execution;

#[cfg(test)]
pub(crate) fn reconcile_workspace_frame(
    frame: WorkspaceFrameId,
    values: Vec<(String, Value)>,
) -> Result<(), String> {
    let context = runmat_runtime::context::legacy::active();
    reconcile_workspace_frame_with_context(frame, context.as_ref(), values)
        .map_err(|error| error.to_string())
}

pub(crate) fn reconcile_workspace_frame_with_context(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    mut values: Vec<(String, Value)>,
) -> Result<(), RuntimeError> {
    values.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    if values.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(invalid_snapshot());
    }
    let prepared = prepare(frame, context, values)?;
    prepared.checkpoint()?;
    prepared.commit();
    Ok(())
}

mod preparation;
use preparation::prepare;
