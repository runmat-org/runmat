//! Admitted mutable backing for one VM workspace frame.
//!
//! Names have one canonical exact owner shared by the frame's indexes and
//! lifecycle. The collection allocator keeps each physical table/vector block
//! claimed after the issuing operation retires.

use allocator_api2::vec::Vec as AllocatedVec;
use hashbrown::{HashMap as AllocatedMap, HashSet as AllocatedSet};
use runmat_execution::resource::ExactShared;
use runmat_runtime::context::host_allocation::{
    frame_hash_table_retained_bytes, FrameCollectionGate, HostAllocationError, HostAllocationPlan,
    HostAllocationReceipt, OperationCollectionAllocator,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::WorkspaceName;
use runmat_value::ExactHostAllocationClaim;
use std::cell::Cell;
use std::collections::{hash_map::RandomState, HashMap, HashSet};

use super::{SlotLifecycle, WorkspaceFrameId, WorkspaceOwner, WorkspaceState, WorkspaceStateGuard};
use crate::interpreter::vars::InterpreterVars;

pub type FrameMap<K, V> = AllocatedMap<K, V, RandomState, OperationCollectionAllocator>;
pub type FrameSet<K> = AllocatedSet<K, RandomState, OperationCollectionAllocator>;
pub type FrameVec<T> = AllocatedVec<T, OperationCollectionAllocator>;

pub(super) fn empty_map<K: Eq + std::hash::Hash, V>(
    allocator: OperationCollectionAllocator,
) -> FrameMap<K, V> {
    FrameMap::with_hasher_in(RandomState::new(), allocator)
}

pub(super) fn empty_set<K: Eq + std::hash::Hash>(
    allocator: OperationCollectionAllocator,
) -> FrameSet<K> {
    FrameSet::with_hasher_in(RandomState::new(), allocator)
}

fn add(total: &mut u64, bytes: u64) -> Result<(), HostAllocationError> {
    *total = total
        .checked_add(bytes)
        .ok_or(HostAllocationError::Overflow)?;
    Ok(())
}

pub(super) fn initial_plan(
    names: &HashMap<String, usize>,
    assigned: &HashSet<String>,
    global_aliases: &runmat_runtime::workspace::bindings::FrameGlobalBindings,
    _value_count: usize,
    captures_completion: bool,
) -> Result<HostAllocationPlan, HostAllocationError> {
    let mut retained = FrameCollectionGate::retained_bytes()?;
    add(&mut retained, global_aliases.ingress_retained_bytes()?)?;
    add(&mut retained, FrameMutationPhase::retained_bytes()?)?;
    add(
        &mut retained,
        frame_hash_table_retained_bytes::<WorkspaceName, usize>(names.len())?,
    )?;
    add(
        &mut retained,
        frame_hash_table_retained_bytes::<WorkspaceName, ()>(assigned.len())?,
    )?;
    add(
        &mut retained,
        frame_hash_table_retained_bytes::<usize, SlotLifecycle>(names.len())?,
    )?;
    add(
        &mut retained,
        InterpreterVars::shared_control_retained_bytes()?,
    )?;
    if captures_completion {
        add(
            &mut retained,
            super::completion::CompletionHandle::retained_bytes()?,
        )?;
    }
    for name in names.keys() {
        let physical = u64::try_from(
            WorkspaceName::requested_bytes(name).map_err(HostAllocationError::Numeric)?,
        )
        .map_err(|_| HostAllocationError::Overflow)?;
        add(
            &mut retained,
            HostAllocationPlan::retained_exact_output_bytes(physical, 1)?,
        )?;
    }
    HostAllocationPlan::new(0, retained)
}

pub(super) fn build_initial(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    values: &mut InterpreterVars,
    context: Option<&RuntimeContext>,
    receipt: Option<&HostAllocationReceipt>,
) -> Result<WorkspaceState, HostAllocationError> {
    let gate = match (context, receipt) {
        (Some(context), Some(receipt)) => FrameCollectionGate::prepare(context, receipt)?,
        _ => FrameCollectionGate::unbounded(),
    };
    let phase = match (context, receipt) {
        (Some(context), Some(receipt)) => FrameMutationPhase::prepare(context, receipt)?,
        _ => FrameMutationPhase::unbounded(),
    };
    let allocator = gate.allocator();
    let _installed = match (context, receipt) {
        (Some(context), Some(receipt)) => gate.install(context, receipt)?,
        _ => gate.install_unbounded()?,
    };
    let mut frame_names = empty_map(allocator.clone());
    let mut frame_assigned = empty_set(allocator.clone());
    let mut lifecycle = empty_map(allocator.clone());
    frame_names.try_reserve(names.len()).map_err(|_| {
        HostAllocationError::Numeric(
            runmat_value::ExactHostNumericAllocationError::AllocationFailed,
        )
    })?;
    frame_assigned.try_reserve(assigned.len()).map_err(|_| {
        HostAllocationError::Numeric(
            runmat_value::ExactHostNumericAllocationError::AllocationFailed,
        )
    })?;
    lifecycle.try_reserve(names.len()).map_err(|_| {
        HostAllocationError::Numeric(
            runmat_value::ExactHostNumericAllocationError::AllocationFailed,
        )
    })?;
    let frame_values = values.promote_shared(context, receipt)?;
    for (name, slot) in names {
        let canonical = match (context, receipt) {
            (Some(context), Some(receipt)) => receipt
                .with_output_allocation(context, |permit| WorkspaceName::prepare(&name, permit))?,
            _ => WorkspaceName::unbounded(&name),
        };
        let is_assigned = assigned.contains(&name);
        if is_assigned {
            frame_assigned.insert(canonical.clone());
        }
        lifecycle.insert(
            slot,
            if is_assigned {
                SlotLifecycle::Assigned(canonical.clone())
            } else {
                SlotLifecycle::Unassigned(canonical.clone())
            },
        );
        frame_names.insert(canonical, slot);
    }
    if let (Some(context), Some(receipt)) = (context, receipt) {
        receipt.checkpoint(context)?;
    }
    Ok(WorkspaceState {
        names: frame_names,
        assigned: frame_assigned,
        assigned_names_this_execution: empty_set(allocator.clone()),
        assigned_ids_this_execution: empty_set(allocator.clone()),
        removed_slots_this_execution: empty_map(allocator.clone()),
        removed_ids_this_execution: empty_set(allocator.clone()),
        removed_names_this_execution: empty_set(allocator.clone()),
        slot_lifecycle: lifecycle,
        values: frame_values,
        pending_var_reconciliation: empty_set(allocator),
        collection_gate: gate,
        mutation_phase: phase,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MutationPhase {
    Idle,
    Prepared,
    Publishing,
}

#[derive(Debug)]
struct MutationPhasePayload {
    phase: Cell<MutationPhase>,
    close_when_idle: Cell<bool>,
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug)]
pub(super) struct FrameMutationPhase(ExactShared<MutationPhasePayload>);

impl FrameMutationPhase {
    pub(super) fn retained_bytes() -> Result<u64, HostAllocationError> {
        HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(ExactShared::<MutationPhasePayload>::requested_bytes())
                .map_err(|_| HostAllocationError::Overflow)?,
            1,
        )
    }

    pub(super) fn prepare(
        context: &RuntimeContext,
        receipt: &HostAllocationReceipt,
    ) -> Result<Self, HostAllocationError> {
        receipt.with_output_allocation(context, |permit| {
            let claim = permit
                .reserve_exact_claim(ExactShared::<MutationPhasePayload>::requested_bytes())?;
            ExactShared::try_new(MutationPhasePayload {
                phase: Cell::new(MutationPhase::Idle),
                close_when_idle: Cell::new(false),
                _claim: Some(claim),
            })
            .map(Self)
            .map_err(|_| runmat_value::ExactHostNumericAllocationError::AllocationFailed)
        })
    }

    pub(super) fn unbounded() -> Self {
        Self(ExactShared::new(MutationPhasePayload {
            phase: Cell::new(MutationPhase::Idle),
            close_when_idle: Cell::new(false),
            _claim: None,
        }))
    }

    pub(super) fn is_busy(&self) -> bool {
        self.0.get().phase.get() != MutationPhase::Idle
    }

    pub(super) fn defer_close(&self) {
        self.0.get().close_when_idle.set(true);
    }

    pub(super) fn acquire(
        &self,
        owner: WorkspaceOwner,
        frame: WorkspaceFrameId,
    ) -> Option<FrameMutationToken> {
        let phase = &self.0.get().phase;
        if phase.get() != MutationPhase::Idle {
            return None;
        }
        phase.set(MutationPhase::Prepared);
        Some(FrameMutationToken {
            phase: self.clone(),
            owner,
            frame,
        })
    }
}

pub(super) struct FrameMutationToken {
    phase: FrameMutationPhase,
    owner: WorkspaceOwner,
    frame: WorkspaceFrameId,
}

impl FrameMutationToken {
    pub(super) fn is_prepared(&self) -> bool {
        self.phase.0.get().phase.get() == MutationPhase::Prepared
    }

    pub(super) fn begin_publication(&self) -> bool {
        let phase = &self.phase.0.get().phase;
        if phase.get() != MutationPhase::Prepared {
            return false;
        }
        phase.set(MutationPhase::Publishing);
        true
    }

    pub(super) fn is_publishing(&self) -> bool {
        self.phase.0.get().phase.get() == MutationPhase::Publishing
    }
}

impl Drop for FrameMutationToken {
    fn drop(&mut self) {
        let payload = self.phase.0.get();
        payload.phase.set(MutationPhase::Idle);
        if payload.close_when_idle.replace(false) {
            drop(WorkspaceStateGuard {
                owner: self.owner,
                frame: self.frame,
            });
        }
    }
}
