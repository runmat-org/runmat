//! Allocation-aware queue storage for the invocation continuation driver.

use super::{CallId, QueuedCall, ReadyEntry};
use allocator_api2::vec::Vec as AllocatedVec;
use runmat_runtime::context::host_allocation::{
    collection_block_retained_bytes, AdmittedCollectionAllocator, HostAllocationError,
    HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_value::{ExactHostAllocationPermit, ExactHostNumericAllocationError};
use std::alloc::Layout;
use std::collections::{BinaryHeap, HashMap};

type ManagedTasks = AllocatedVec<ManagedTaskSlot, AdmittedCollectionAllocator>;
type ManagedReady = AllocatedVec<ReadyEntry, AdmittedCollectionAllocator>;

pub(super) enum TaskStorage {
    Unbounded(HashMap<CallId, QueuedCall>),
    Managed {
        slots: ManagedTasks,
        free_head: Option<usize>,
    },
}

pub(super) struct ManagedTaskSlot {
    generation: u32,
    task: Option<QueuedCall>,
    next_free: Option<usize>,
}

pub(super) enum ReadyStorage {
    Unbounded(BinaryHeap<ReadyEntry>),
    Managed(ManagedReady),
}

pub(super) struct PreparedQueueGrowth {
    receipt: HostAllocationReceipt,
    tasks: Option<ManagedTasks>,
    ready: Option<ManagedReady>,
}

pub(super) struct QueueRetirement {
    _tasks: Option<ManagedTasks>,
    _ready: Option<ManagedReady>,
}

pub(super) struct QueueGrowthGeometry {
    pub(super) task_capacity: usize,
    pub(super) task_index: Option<usize>,
    pub(super) ready_capacity: usize,
    pub(super) ready_len: usize,
}

impl TaskStorage {
    pub(super) fn unbounded() -> Self {
        Self::Unbounded(HashMap::new())
    }

    pub(super) fn managed(allocator: AdmittedCollectionAllocator) -> Self {
        Self::Managed {
            slots: AllocatedVec::new_in(allocator),
            free_head: None,
        }
    }

    pub(super) fn is_managed(&self) -> bool {
        matches!(self, Self::Managed { .. })
    }

    pub(super) fn capacity(&self) -> usize {
        match self {
            Self::Unbounded(tasks) => tasks.capacity(),
            Self::Managed { slots, .. } => slots.capacity(),
        }
    }

    pub(super) fn next_id(&self, unbounded: u64) -> Result<CallId, HostAllocationError> {
        match self {
            Self::Unbounded(_) => Ok(CallId(unbounded)),
            Self::Managed { slots, free_head } => {
                let index = free_head.unwrap_or(slots.len());
                let generation = slots.get(index).map_or(0, |slot| slot.generation);
                CallId::managed(index, generation)
            }
        }
    }

    pub(super) fn required_index(&self) -> usize {
        match self {
            Self::Unbounded(_) => 0,
            Self::Managed { slots, free_head } => free_head.unwrap_or(slots.len()),
        }
    }

    pub(super) fn insert(&mut self, id: CallId, task: QueuedCall) {
        match self {
            Self::Unbounded(tasks) => {
                tasks.insert(id, task);
            }
            Self::Managed { slots, free_head } => {
                let (index, generation) = id.managed_parts().expect("managed call id");
                if slots.len() == index {
                    slots.push(ManagedTaskSlot {
                        generation,
                        task: Some(task),
                        next_free: None,
                    });
                    return;
                }
                let slot = &mut slots[index];
                debug_assert_eq!(slot.generation, generation);
                debug_assert!(slot.task.is_none());
                // A task is temporarily removed from its slot while it is
                // polled. Reinserting that task must not disturb the free
                // list; only a newly registered task consumes its head.
                if *free_head == Some(index) {
                    *free_head = slot.next_free.take();
                }
                slot.task = Some(task);
            }
        }
    }

    pub(super) fn take(&mut self, id: &CallId) -> Option<QueuedCall> {
        match self {
            Self::Unbounded(tasks) => tasks.remove(id),
            Self::Managed { slots, .. } => {
                let (index, generation) = id.managed_parts()?;
                let slot = slots.get_mut(index)?;
                (slot.generation == generation)
                    .then_some(())
                    .and_then(|()| slot.task.take())
            }
        }
    }

    pub(super) fn retire(&mut self, id: &CallId) -> Option<QueuedCall> {
        let task = self.take(id)?;
        self.release(*id);
        Some(task)
    }

    pub(super) fn release(&mut self, id: CallId) {
        let Self::Managed { slots, free_head } = self else {
            return;
        };
        let Some((index, generation)) = id.managed_parts() else {
            return;
        };
        let Some(slot) = slots.get_mut(index) else {
            return;
        };
        if slot.generation != generation || slot.task.is_some() {
            return;
        }
        let Some(next_generation) = slot.generation.checked_add(1) else {
            return;
        };
        slot.generation = next_generation;
        slot.next_free = *free_head;
        *free_head = Some(index);
    }

    pub(super) fn get(&self, id: &CallId) -> Option<&QueuedCall> {
        match self {
            Self::Unbounded(tasks) => tasks.get(id),
            Self::Managed { slots, .. } => {
                let (index, generation) = id.managed_parts()?;
                let slot = slots.get(index)?;
                (slot.generation == generation)
                    .then_some(slot.task.as_ref())
                    .flatten()
            }
        }
    }

    pub(super) fn ids(&self) -> impl Iterator<Item = CallId> + '_ {
        match self {
            Self::Unbounded(tasks) => TaskIds::Unbounded(tasks.keys().copied()),
            Self::Managed { slots, .. } => TaskIds::Managed(slots.iter().enumerate()),
        }
    }

    fn replace_backing(&mut self, mut replacement: ManagedTasks) -> Option<ManagedTasks> {
        let Self::Managed { slots, .. } = self else {
            return None;
        };
        debug_assert!(replacement.capacity() >= slots.len());
        replacement.extend(slots.drain(..));
        Some(std::mem::replace(slots, replacement))
    }
}

enum TaskIds<'a> {
    Unbounded(std::iter::Copied<std::collections::hash_map::Keys<'a, CallId, QueuedCall>>),
    Managed(std::iter::Enumerate<std::slice::Iter<'a, ManagedTaskSlot>>),
}

impl Iterator for TaskIds<'_> {
    type Item = CallId;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Unbounded(ids) => ids.next(),
            Self::Managed(entries) => entries.find_map(|(index, slot)| {
                slot.task
                    .as_ref()
                    .and_then(|_| CallId::managed(index, slot.generation).ok())
            }),
        }
    }
}

impl ReadyStorage {
    pub(super) fn unbounded() -> Self {
        Self::Unbounded(BinaryHeap::new())
    }

    pub(super) fn managed(allocator: AdmittedCollectionAllocator) -> Self {
        Self::Managed(AllocatedVec::new_in(allocator))
    }

    pub(super) fn capacity(&self) -> usize {
        match self {
            Self::Unbounded(ready) => ready.capacity(),
            Self::Managed(ready) => ready.capacity(),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Unbounded(ready) => ready.len(),
            Self::Managed(ready) => ready.len(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn push(&mut self, entry: ReadyEntry) {
        match self {
            Self::Unbounded(ready) => ready.push(entry),
            Self::Managed(ready) => {
                debug_assert!(ready.len() < ready.capacity());
                ready.push(entry);
                let mut child = ready.len() - 1;
                while child > 0 {
                    let parent = (child - 1) / 2;
                    if ready[parent] >= ready[child] {
                        break;
                    }
                    ready.swap(parent, child);
                    child = parent;
                }
            }
        }
    }

    pub(super) fn pop(&mut self) -> Option<ReadyEntry> {
        match self {
            Self::Unbounded(ready) => ready.pop(),
            Self::Managed(ready) => {
                let last = ready.pop()?;
                if ready.is_empty() {
                    return Some(last);
                }
                let result = std::mem::replace(&mut ready[0], last);
                let mut parent = 0;
                loop {
                    let left = parent * 2 + 1;
                    if left >= ready.len() {
                        break;
                    }
                    let right = left + 1;
                    let child = if right < ready.len() && ready[right] > ready[left] {
                        right
                    } else {
                        left
                    };
                    if ready[parent] >= ready[child] {
                        break;
                    }
                    ready.swap(parent, child);
                    parent = child;
                }
                Some(result)
            }
        }
    }

    pub(super) fn remove(&mut self, id: CallId) {
        match self {
            Self::Unbounded(ready) => ready.retain(|entry| entry.0.get().id != id),
            Self::Managed(ready) => {
                ready.retain(|entry| entry.0.get().id != id);
                // Restore the max heap in place after the rare cancellation
                // removal. Ordinary enqueue/pop remain O(log n).
                for parent in (0..ready.len() / 2).rev() {
                    sift_down(ready, parent);
                }
            }
        }
    }

    fn replace_backing(&mut self, mut replacement: ManagedReady) -> Option<ManagedReady> {
        let Self::Managed(ready) = self else {
            return None;
        };
        debug_assert!(replacement.capacity() >= ready.len());
        replacement.extend(ready.drain(..));
        Some(std::mem::replace(ready, replacement))
    }
}

fn sift_down(ready: &mut ManagedReady, mut parent: usize) {
    loop {
        let left = parent * 2 + 1;
        if left >= ready.len() {
            break;
        }
        let right = left + 1;
        let child = if right < ready.len() && ready[right] > ready[left] {
            right
        } else {
            left
        };
        if ready[parent] >= ready[child] {
            break;
        }
        ready.swap(parent, child);
        parent = child;
    }
}

impl PreparedQueueGrowth {
    pub(super) fn prepare(
        context: &RuntimeContext,
        geometry: QueueGrowthGeometry,
        scratch_bytes: u64,
        control_bytes: u64,
        control_claims: usize,
    ) -> Result<Self, HostAllocationError> {
        let task_capacity = geometry
            .task_index
            .map(|index| growth_capacity(geometry.task_capacity, index + 1))
            .transpose()?
            .flatten();
        let active_capacity = task_capacity.unwrap_or(geometry.task_capacity);
        let ready_required = geometry
            .ready_len
            .checked_add(1)
            .and_then(|ready| {
                active_capacity
                    .checked_add(1)
                    .map(|active| ready.max(active))
            })
            .ok_or(HostAllocationError::Overflow)?;
        let ready_capacity = growth_capacity(geometry.ready_capacity, ready_required)?;
        let mut retained = 0_u64;
        if let Some(capacity) = task_capacity {
            retained = retained
                .checked_add(collection_bytes::<ManagedTaskSlot>(capacity)?)
                .ok_or(HostAllocationError::Overflow)?;
        }
        if let Some(capacity) = ready_capacity {
            retained = retained
                .checked_add(collection_bytes::<ReadyEntry>(capacity)?)
                .ok_or(HostAllocationError::Overflow)?;
        }
        retained = retained
            .checked_add(HostAllocationPlan::retained_exact_output_bytes(
                control_bytes,
                control_claims,
            )?)
            .ok_or(HostAllocationError::Overflow)?;
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(scratch_bytes, retained)?,
        )?
        .ok_or(HostAllocationError::MissingManagedOwner)?;
        let allocator = receipt.collection_allocator(context)?;
        let tasks = task_capacity
            .map(|capacity| allocated_tasks(capacity, allocator.clone()))
            .transpose()?;
        let ready = ready_capacity
            .map(|capacity| allocated_ready(capacity, allocator))
            .transpose()?;
        receipt.checkpoint(context)?;
        Ok(Self {
            receipt,
            tasks,
            ready,
        })
    }

    pub(super) fn with_output_allocation<T>(
        &self,
        context: &RuntimeContext,
        construct: impl FnOnce(
            &dyn ExactHostAllocationPermit,
        ) -> Result<T, ExactHostNumericAllocationError>,
    ) -> Result<T, HostAllocationError> {
        self.receipt.with_output_allocation(context, construct)
    }

    pub(super) fn with_output_construction<T, E>(
        &self,
        context: &RuntimeContext,
        construct: impl FnOnce(&dyn ExactHostAllocationPermit) -> Result<T, E>,
    ) -> Result<Result<T, E>, HostAllocationError> {
        self.receipt.with_output_construction(context, construct)
    }

    pub(super) fn checkpoint(&self, context: &RuntimeContext) -> Result<(), HostAllocationError> {
        self.receipt.checkpoint(context)?;
        Ok(())
    }

    pub(super) fn install(
        &mut self,
        tasks: &mut TaskStorage,
        ready: &mut ReadyStorage,
    ) -> QueueRetirement {
        let old_tasks = self
            .tasks
            .take()
            .and_then(|next| tasks.replace_backing(next));
        let old_ready = self
            .ready
            .take()
            .and_then(|next| ready.replace_backing(next));
        QueueRetirement {
            _tasks: old_tasks,
            _ready: old_ready,
        }
    }

    pub(super) fn retained_scratch(
        &self,
    ) -> std::sync::Arc<runmat_runtime::context::host_effect::ManagedHostEffectStageReservation>
    {
        self.receipt.retain_scratch_for_cpu_work()
    }
}

fn allocated_tasks(
    capacity: usize,
    allocator: AdmittedCollectionAllocator,
) -> Result<ManagedTasks, HostAllocationError> {
    let mut tasks = AllocatedVec::new_in(allocator);
    tasks
        .try_reserve_exact(capacity)
        .map_err(|_| HostAllocationError::Overflow)?;
    Ok(tasks)
}

fn allocated_ready(
    capacity: usize,
    allocator: AdmittedCollectionAllocator,
) -> Result<ManagedReady, HostAllocationError> {
    let mut ready = AllocatedVec::new_in(allocator);
    ready
        .try_reserve_exact(capacity)
        .map_err(|_| HostAllocationError::Overflow)?;
    Ok(ready)
}

fn growth_capacity(current: usize, required: usize) -> Result<Option<usize>, HostAllocationError> {
    if required <= current {
        return Ok(None);
    }
    let doubled = current
        .max(1)
        .checked_mul(2)
        .ok_or(HostAllocationError::Overflow)?;
    Ok(Some(doubled.max(required)))
}

fn collection_bytes<T>(capacity: usize) -> Result<u64, HostAllocationError> {
    let layout = Layout::array::<T>(capacity).map_err(|_| HostAllocationError::Overflow)?;
    collection_block_retained_bytes(layout)
}
