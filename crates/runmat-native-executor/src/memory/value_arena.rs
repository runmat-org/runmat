#[cfg(not(target_arch = "wasm32"))]
use std::ptr::NonNull;

#[cfg(not(target_arch = "wasm32"))]
use runmat_gc::{GcHandle, GcRoot, RootId, Trace, Tracer};
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::{RuntimeContext, RuntimeContextLocalId};
use runmat_runtime::native::NativeValueRef;
use runmat_value::{
    exact_capacity_vec, ExactHostAllocationClaim, ExactHostNumericAllocationError, Value,
};
use std::alloc::{alloc, Layout};

use runmat_runtime::call::CapturedValueLease;
use runmat_runtime::workspace::WorkspaceSlotReadLease;

use crate::{NativeExecutorError, NativeExecutorResult};

#[derive(Debug)]
struct Entry {
    generation: u64,
    references: usize,
    marked: bool,
    // usize::MAX is outside every valid Entry-array layout.
    next_free: usize,
    value: Option<EntryValue>,
}

/// A workspace read retains its original immutable value and root owner.
/// Native stores replace references; they never mutate a borrowed generation.
#[derive(Debug)]
enum EntryValue {
    Owned(Value),
    Retained(WorkspaceSlotReadLease),
    Capture(CapturedValueLease),
    /// A non-owning view of one element of an invocation-owned `varargin`.
    /// The parent's arena reference is retained until this entry is released.
    CellElement {
        parent: NativeValueRef,
        index: usize,
    },
}

impl EntryValue {
    /// A cadence signal, not a new allocation quote or ownership charge.
    /// Unknown heap forests force the next boundary without walking payloads.
    fn retirement_bytes(&self) -> usize {
        let value = match self {
            Self::Owned(value) => value,
            Self::Retained(lease) => lease.value(),
            Self::Capture(lease) => lease.value(),
            Self::CellElement { .. } => return 0,
        };
        match value {
            Value::Int(_) | Value::Num(_) | Value::Complex(_, _) | Value::Bool(_) => 0,
            Value::Tensor(tensor) => tensor
                .host_buffer()
                .checked_byte_len()
                .unwrap_or(usize::MAX),
            _ => usize::MAX,
        }
    }

    fn value(&self) -> &Value {
        match self {
            Self::Owned(value) => value,
            Self::Retained(lease) => lease.value(),
            Self::Capture(lease) => lease.value(),
            Self::CellElement { .. } => unreachable!("cell element resolves through arena"),
        }
    }
}

#[derive(Debug, Default)]
struct ArenaEntries {
    values: Vec<Entry>,
    // Physical entry backing retires before its original retained claim.
    values_claim: Option<ExactHostAllocationClaim>,
}

/// Invocation-owned opaque value table.
///
/// Generated code sees only generation-checked `NativeValueRef` tokens. The
/// current Rust `Value` representation never crosses the native ABI.
#[derive(Debug)]
pub struct ValueArena {
    // The root scanner retains a pointer to this allocation. Boxing the table
    // keeps that pointer stable when the arena itself moves into HostState.
    entries: Box<ArenaEntries>,
    // The header claim outlives the boxed header's physical deallocation.
    _entries_claim: Option<ExactHostAllocationClaim>,
    context_identity: Option<RuntimeContextLocalId>,
    free: Option<usize>,
    retirement_debt: usize,
    payload_byte_debt: usize,
    retirement_threshold: usize,
    #[cfg(not(target_arch = "wasm32"))]
    root_id: Option<RootId>,
    // Registration owns the physical scanner Box until unregister in Drop.
    #[cfg(not(target_arch = "wasm32"))]
    _root_claim: Option<ExactHostAllocationClaim>,
}

impl ValueArena {
    pub fn new() -> NativeExecutorResult<Self> {
        Self::prepare(None)
    }

    pub(crate) fn new_in(runtime: &RuntimeContext) -> NativeExecutorResult<Self> {
        Self::prepare(Some(runtime))
    }

    fn prepare(runtime: Option<&RuntimeContext>) -> NativeExecutorResult<Self> {
        let (entries, entries_claim) =
            allocate_metadata(runtime, Layout::new::<ArenaEntries>(), || {
                // SAFETY: allocate the actual Box layout after admission, then
                // initialize exactly once before transferring it to Box.
                let pointer =
                    unsafe { alloc(Layout::new::<ArenaEntries>()) }.cast::<ArenaEntries>();
                if pointer.is_null() {
                    return Err(ExactHostNumericAllocationError::AllocationFailed);
                }
                unsafe {
                    pointer.write(ArenaEntries::default());
                    Ok(Box::from_raw(pointer))
                }
            })?;
        let arena = Self {
            entries,
            _entries_claim: entries_claim,
            context_identity: runtime.map(RuntimeContext::local_identity),
            free: None,
            retirement_debt: 0,
            payload_byte_debt: 0,
            retirement_threshold: 64,
            #[cfg(not(target_arch = "wasm32"))]
            root_id: None,
            #[cfg(not(target_arch = "wasm32"))]
            _root_claim: None,
        };
        #[cfg(target_arch = "wasm32")]
        {
            Ok(arena)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut arena = arena;
            arena.activate_gc_roots(runtime)?;
            Ok(arena)
        }
    }

    /// Register every live value in this invocation as a GC root source.
    #[cfg(not(target_arch = "wasm32"))]
    fn activate_gc_roots(&mut self, runtime: Option<&RuntimeContext>) -> NativeExecutorResult<()> {
        if self.root_id.is_some() {
            return Ok(());
        }
        // SAFETY: `entries` is boxed and therefore remains at this address even
        // if `ValueArena` moves. `Drop` unregisters the scanner before the box
        // is released. Collection is deferred when another thread owns roots,
        // and native execution never scans concurrently with arena mutation on
        // the owning thread.
        let root = unsafe { ArenaRoot::new(NonNull::from(self.entries.as_mut())) };
        let (root, claim) = allocate_metadata(runtime, Layout::new::<ArenaRoot>(), || {
            // SAFETY: this nonzero exact root layout is initialized once,
            // and GC registration takes the resulting Box's ownership.
            let pointer = unsafe { alloc(Layout::new::<ArenaRoot>()) }.cast::<ArenaRoot>();
            if pointer.is_null() {
                return Err(ExactHostNumericAllocationError::AllocationFailed);
            }
            unsafe {
                pointer.write(root);
                Ok(Box::from_raw(pointer))
            }
        })?;
        self.root_id = Some(runmat_gc::gc_register_value_root(root).map_err(|error| {
            NativeExecutorError::Host(format!("register native GC roots: {error}"))
        })?);
        self._root_claim = claim;
        Ok(())
    }

    pub fn insert(
        &mut self,
        value: Value,
        runtime: Option<&RuntimeContext>,
    ) -> NativeExecutorResult<NativeValueRef> {
        self.insert_entry(EntryValue::Owned(value), runtime)
    }

    /// Retain an executor input without materializing another language value.
    pub(crate) fn insert_retained_lease(
        &mut self,
        value: impl Into<WorkspaceSlotReadLease>,
        runtime: Option<&RuntimeContext>,
    ) -> NativeExecutorResult<NativeValueRef> {
        self.insert_entry(EntryValue::Retained(value.into()), runtime)
    }

    pub(crate) fn insert_capture_lease(
        &mut self,
        lease: CapturedValueLease,
        runtime: Option<&RuntimeContext>,
    ) -> NativeExecutorResult<NativeValueRef> {
        self.insert_entry(EntryValue::Capture(lease), runtime)
    }

    pub(crate) fn insert_cell_element(
        &mut self,
        parent: NativeValueRef,
        index: usize,
        runtime: Option<&RuntimeContext>,
    ) -> NativeExecutorResult<NativeValueRef> {
        if matches!(
            self.entry(parent)?.value.as_ref(),
            Some(EntryValue::CellElement { .. })
        ) {
            return Err(NativeExecutorError::StaleValue);
        }
        let Value::Cell(cell) = self.get(parent)? else {
            return Err(NativeExecutorError::StaleValue);
        };
        if cell.data().get(index).is_none() {
            return Err(NativeExecutorError::StaleValue);
        }
        // Admission may fail; never strand a parent retain on that path.
        self.prepare_insert(runtime)?;
        self.retain(parent)?;
        Ok(self.publish_entry(EntryValue::CellElement { parent, index }))
    }

    /// Refresh a local's immutable generation without adding an arena entry
    /// for an unchanged binding. Older entries remain valid for SSA operands.
    pub(crate) fn reuse_or_insert_retained_lease(
        &mut self,
        previous: NativeValueRef,
        value: impl Into<WorkspaceSlotReadLease>,
        runtime: Option<&RuntimeContext>,
    ) -> NativeExecutorResult<NativeValueRef> {
        let value = value.into();
        if !previous.is_null() && std::ptr::eq(self.get(previous)?, value.value()) {
            return Ok(previous);
        }
        self.insert_retained_lease(value, runtime)
    }

    fn insert_entry(
        &mut self,
        value: EntryValue,
        runtime: Option<&RuntimeContext>,
    ) -> NativeExecutorResult<NativeValueRef> {
        self.prepare_insert(runtime)?;
        Ok(self.publish_entry(value))
    }

    fn prepare_insert(&mut self, runtime: Option<&RuntimeContext>) -> NativeExecutorResult<()> {
        if self.context_identity != runtime.map(RuntimeContext::local_identity) {
            return Err(metadata_error(runtime, HostAllocationError::StaleAuthority));
        }
        if runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(metadata_error(runtime, HostAllocationError::Cancelled));
        }
        if self.free.is_some() || self.entries.values.len() < self.entries.values.capacity() {
            return Ok(());
        }
        let count = self
            .entries
            .values
            .len()
            .checked_add(1)
            .ok_or_else(|| metadata_error(runtime, HostAllocationError::Overflow))?;
        // This is the actual selected growth geometry, not a payload estimate:
        // request and charge precisely this many Entry slots before allocation.
        let capacity = self
            .entries
            .values
            .capacity()
            .checked_mul(2)
            .unwrap_or(count)
            .max(count)
            .max(4);
        let layout = Layout::array::<Entry>(capacity)
            .map_err(|_| metadata_error(runtime, HostAllocationError::Overflow))?;
        let (mut values, claim) =
            allocate_metadata(runtime, layout, || exact_capacity_vec(capacity))?;
        // Move entries once, preserving their generation and every owned edge.
        // Both original and replacement physical blocks are charged at overlap.
        values.append(&mut self.entries.values);
        let old_values = std::mem::replace(&mut self.entries.values, values);
        let old_claim = std::mem::replace(&mut self.entries.values_claim, claim);
        drop(old_values);
        drop(old_claim);
        Ok(())
    }

    /// Called only after capacity admission. Release never allocates: dead
    /// slots link through their own metadata instead of a second Vec backing.
    fn publish_entry(&mut self, value: EntryValue) -> NativeValueRef {
        self.payload_byte_debt = self
            .payload_byte_debt
            .saturating_add(value.retirement_bytes());
        let index = match self.free {
            Some(index) => {
                let next = self.entries.values[index].next_free;
                self.free = (next != usize::MAX).then_some(next);
                self.entries.values[index].next_free = usize::MAX;
                index
            }
            None => {
                let index = self.entries.values.len();
                self.entries.values.push(Entry {
                    generation: 1,
                    references: 0,
                    marked: false,
                    next_free: usize::MAX,
                    value: None,
                });
                index
            }
        };
        let entry = &mut self.entries.values[index];
        entry.references = 1;
        entry.marked = false;
        entry.value = Some(value);
        self.retirement_debt = self.retirement_debt.saturating_add(1);
        NativeValueRef {
            handle: index as u64 + 1,
            generation: entry.generation,
        }
    }

    pub fn get(&self, reference: NativeValueRef) -> NativeExecutorResult<&Value> {
        let entry = self.entry(reference)?;
        match entry
            .value
            .as_ref()
            .ok_or(NativeExecutorError::StaleValue)?
        {
            EntryValue::CellElement { parent, index } => {
                let Value::Cell(cell) = self.get(*parent)? else {
                    return Err(NativeExecutorError::StaleValue);
                };
                cell.data()
                    .get(*index)
                    .ok_or(NativeExecutorError::StaleValue)
            }
            value => Ok(value.value()),
        }
    }

    pub fn retain(&mut self, reference: NativeValueRef) -> NativeExecutorResult<()> {
        if reference.is_null() {
            return Ok(());
        }
        let entry = self.entry_mut(reference)?;
        entry.references = entry
            .references
            .checked_add(1)
            .ok_or(NativeExecutorError::StaleValue)?;
        Ok(())
    }

    /// Only the executor's settled ownership boundary may retire baseline
    /// arena residents. Public insert/retain/release semantics are unchanged.
    pub(crate) fn has_retirement_debt(&self) -> bool {
        self.retirement_debt != 0
    }

    pub(crate) fn retirement_due(&self) -> bool {
        self.retirement_debt >= self.retirement_threshold || self.payload_byte_debt >= 256 * 1024
    }

    pub(crate) fn begin_retirement(&mut self) {
        for entry in &mut self.entries.values {
            entry.marked = false;
        }
    }

    pub(crate) fn mark_resident(&mut self, reference: NativeValueRef) -> NativeExecutorResult<()> {
        if !reference.is_null() {
            self.entry_mut(reference)?.marked = true;
        }
        Ok(())
    }

    pub(crate) fn finish_retirement(&mut self) -> NativeExecutorResult<()> {
        // Retire dead children before considering parents: a parent may have
        // released its initial residence and be owned solely by that edge.
        // Surviving children (marked or explicitly retained) protect ancestors.
        // Nested CellElements are forbidden by insert_cell_element.
        for index in 0..self.entries.values.len() {
            let entry = &self.entries.values[index];
            let Some(EntryValue::CellElement { parent, .. }) = &entry.value else {
                continue;
            };
            if entry.marked || entry.references > 1 {
                let parent = *parent;
                self.mark_resident(parent)?;
            } else {
                let reference = NativeValueRef {
                    handle: index as u64 + 1,
                    generation: entry.generation,
                };
                self.release(reference)?;
            }
        }
        for index in 0..self.entries.values.len() {
            let entry = &self.entries.values[index];
            // Locals/SSA aliases borrow one arena residence; explicit retains
            // (including CellElement parent dependencies) must remain intact.
            if entry.value.is_some() && !entry.marked && entry.references == 1 {
                let reference = NativeValueRef {
                    handle: index as u64 + 1,
                    generation: entry.generation,
                };
                self.release(reference)?;
            }
        }
        let survivors = self
            .entries
            .values
            .iter()
            .filter(|entry| entry.value.is_some())
            .count();
        self.retirement_debt = 0;
        self.payload_byte_debt = 0;
        // Amortize scans against historical slot highwater as well as current
        // residents. Half-highwater permits reuse without forcing fresh slots
        // each cycle, unlike a whole-highwater insertion threshold.
        self.retirement_threshold = survivors
            .saturating_add(64)
            .max(self.entries.values.len() / 2);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn retirement_stats(&self) -> (usize, usize, usize) {
        (
            self.entries
                .values
                .iter()
                .filter(|entry| entry.value.is_some())
                .count(),
            self.entries.values.len(),
            self.entries.values.capacity(),
        )
    }

    pub fn release(&mut self, reference: NativeValueRef) -> NativeExecutorResult<()> {
        if reference.is_null() {
            return Ok(());
        }
        let index = checked_index(reference)?;
        let entry = self
            .entries
            .values
            .get_mut(index)
            .ok_or(NativeExecutorError::StaleValue)?;
        if entry.generation != reference.generation
            || entry.value.is_none()
            || entry.references == 0
        {
            return Err(NativeExecutorError::StaleValue);
        }
        entry.references -= 1;
        // The last explicit pin may disappear without another insertion.
        // Revisit this potentially obsolete residence at the next lawful
        // boundary, using the existing debt rather than another owner tracker.
        if entry.references == 1 {
            self.retirement_debt = self.retirement_debt.max(1);
        }
        let retired = if entry.references == 0 {
            let retired = entry.value.take();
            // A wrapped generation could make a prehistoric reference valid
            // again. Retire the slot permanently instead of permitting ABA.
            if let Some(next_generation) = entry.generation.checked_add(1) {
                entry.generation = next_generation;
                entry.next_free = self.free.unwrap_or(usize::MAX);
                self.free = Some(index);
            }
            retired
        } else {
            None
        };
        if let Some(EntryValue::CellElement { parent, .. }) = &retired {
            self.release(*parent)?;
        }
        drop(retired);
        Ok(())
    }

    fn entry(&self, reference: NativeValueRef) -> NativeExecutorResult<&Entry> {
        if reference.is_null() {
            return Err(NativeExecutorError::StaleValue);
        }
        let entry = self
            .entries
            .values
            .get(checked_index(reference)?)
            .ok_or(NativeExecutorError::StaleValue)?;
        if entry.generation != reference.generation || entry.value.is_none() {
            return Err(NativeExecutorError::StaleValue);
        }
        Ok(entry)
    }

    fn entry_mut(&mut self, reference: NativeValueRef) -> NativeExecutorResult<&mut Entry> {
        if reference.is_null() {
            return Err(NativeExecutorError::StaleValue);
        }
        let entry = self
            .entries
            .values
            .get_mut(checked_index(reference)?)
            .ok_or(NativeExecutorError::StaleValue)?;
        if entry.generation != reference.generation || entry.value.is_none() {
            return Err(NativeExecutorError::StaleValue);
        }
        Ok(entry)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for ValueArena {
    fn drop(&mut self) {
        if let Some(root_id) = self.root_id.take() {
            let _ = runmat_gc::gc_unregister_root(root_id);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct ArenaRoot {
    entries: NonNull<ArenaEntries>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ArenaRoot {
    /// # Safety
    ///
    /// `entries` must remain allocated until this root is unregistered, and it
    /// must not be scanned concurrently with mutation.
    unsafe fn new(entries: NonNull<ArenaEntries>) -> Self {
        Self { entries }
    }

    fn entries(&self) -> &[Entry] {
        // SAFETY: upheld by `new` and `ValueArena::activate_gc_roots`.
        unsafe { &self.entries.as_ref().values }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GcRoot for ArenaRoot {
    fn scan(&self) -> Vec<GcHandle> {
        struct HandleCollector {
            handles: Vec<GcHandle>,
        }

        impl Tracer for HandleCollector {
            fn mark(&mut self, handle: GcHandle) {
                self.handles.push(handle);
            }
        }

        let mut collector = HandleCollector {
            handles: Vec::new(),
        };
        for value in self
            .entries()
            .iter()
            .filter_map(|entry| entry.value.as_ref())
        {
            if !matches!(value, EntryValue::CellElement { .. }) {
                value.value().trace(&mut collector);
            }
        }
        collector.handles
    }

    fn description(&self) -> String {
        "native_value_arena".to_owned()
    }

    fn estimated_size(&self) -> usize {
        std::mem::size_of_val(self.entries())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl runmat_gc::ValueRoot for ArenaRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        // Same pinned arena lifetime and exclusive scan as the GC projection.
        for value in self
            .entries()
            .iter()
            .filter_map(|entry| entry.value.as_ref())
        {
            if !matches!(value, EntryValue::CellElement { .. }) {
                visitor(value.value())?;
            }
        }
        Some(())
    }
}

/// RAII protection for values while native invocation state is being built.
/// The root scanner owns the cloned values; this guard owns only registration.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct ScopedValueRoots {
    root_id: Option<RootId>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ScopedValueRoots {
    pub(crate) fn register(values: Vec<Value>, description: &str) -> NativeExecutorResult<Self> {
        if values.is_empty() {
            return Ok(Self { root_id: None });
        }
        let root = runmat_gc::GlobalRoot::new(values, description.to_string());
        let root_id = runmat_gc::gc_register_value_root(Box::new(root)).map_err(|error| {
            NativeExecutorError::Host(format!("register native GC roots: {error}"))
        })?;
        Ok(Self {
            root_id: Some(root_id),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for ScopedValueRoots {
    fn drop(&mut self) {
        if let Some(root_id) = self.root_id.take() {
            let _ = runmat_gc::gc_unregister_root(root_id);
        }
    }
}

fn metadata_error(
    runtime: Option<&RuntimeContext>,
    cause: HostAllocationError,
) -> NativeExecutorError {
    NativeExecutorError::from(host_allocation_runtime_error(
        runtime,
        "RunMat:Native:ArenaAdmission",
        "native arena metadata admission failed",
        cause,
    ))
}

/// One canonical original-owner claim, from the selected operation's issuer.
/// The caller retains it until its exact physical block has been deallocated.
fn allocate_metadata<T>(
    runtime: Option<&RuntimeContext>,
    layout: Layout,
    allocate: impl FnOnce() -> Result<T, ExactHostNumericAllocationError>,
) -> NativeExecutorResult<(T, Option<ExactHostAllocationClaim>)> {
    let error = |cause| metadata_error(runtime, cause);
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(layout.size()).map_err(|_| error(HostAllocationError::Overflow))?,
        1,
    )
    .map_err(error)?;
    let plan = HostAllocationPlan::new(0, retained).map_err(error)?;
    let receipt = HostAllocationReceipt::admit(runtime, plan).map_err(error)?;
    match (runtime, receipt) {
        (Some(runtime), Some(receipt)) => receipt
            .with_output_allocation(runtime, |permit| {
                let claim = permit.reserve_exact_claim(layout.size())?;
                Ok((allocate()?, Some(claim)))
            })
            .map_err(error),
        (_, None) => allocate()
            .map(|value| (value, None))
            .map_err(|cause| error(HostAllocationError::Numeric(cause))),
        (None, Some(_)) => Err(error(HostAllocationError::StaleAuthority)),
    }
}

fn checked_index(reference: NativeValueRef) -> NativeExecutorResult<usize> {
    let handle = reference
        .handle
        .checked_sub(1)
        .ok_or(NativeExecutorError::StaleValue)?;
    usize::try_from(handle).map_err(|_| NativeExecutorError::StaleValue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::workspace::WorkspaceValueReadLease;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn managed_metadata_denial_preserves_generations_and_parent_edges() {
        use runmat_runtime::context::{
            host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
            RuntimeParallelService, RuntimeServicePorts,
        };
        use std::rc::Rc;

        struct Parallel(ManagedHostEffectLedger);
        impl RuntimeParallelService for Parallel {
            fn supports(&self, _: ParallelCapability) -> bool {
                false
            }
            fn placement_resources(&self) -> RuntimeParallelResources {
                let snapshot = self.0.snapshot();
                RuntimeParallelResources {
                    memory_available_bytes: Some(snapshot.free_bytes()),
                    epoch: snapshot.epoch,
                    ..Default::default()
                }
            }
            fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
                Some(self.0.clone())
            }
        }

        runmat_gc::gc_test_context(|| {
            let ledger = ManagedHostEffectLedger::new(16 * 1024);
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            );
            let guard = runtime.enter();
            let mut arena = ValueArena::new_in(&runtime).unwrap();
            assert!(
                ledger.snapshot().live_claimed_bytes >= std::mem::size_of::<ArenaEntries>() as u64
            );
            // This fixture tests metadata admission, not the Cell payload's
            // separate producer accounting or whole-invocation certification.
            let parent = arena
                .insert(
                    Value::Cell(runmat_value::CellArray::new(vec![Value::Num(9.0)], 1, 1).unwrap()),
                    Some(&runtime),
                )
                .unwrap();
            let mut last = parent;
            loop {
                assert!(
                    arena.entries.values.len() < 1024,
                    "bounded metadata denial was not reached"
                );
                match arena.insert(Value::Num(1.0), Some(&runtime)) {
                    Ok(reference) => last = reference,
                    Err(_) => break,
                }
            }
            let before = arena.retirement_stats();
            assert_eq!(before.1, before.2);
            let claimed = ledger.snapshot().live_claimed_bytes;
            assert!(
                claimed
                    >= (std::mem::size_of::<ArenaEntries>()
                        + before.2 * std::mem::size_of::<Entry>()) as u64
            );
            let references = arena.entry(parent).unwrap().references;
            assert!(arena
                .insert_cell_element(parent, 0, Some(&runtime))
                .is_err());
            assert_eq!(arena.entry(parent).unwrap().references, references);
            assert_eq!(arena.retirement_stats(), before);
            assert_eq!(ledger.snapshot().live_claimed_bytes, claimed);
            assert_eq!(arena.get(last).unwrap(), &Value::Num(1.0));

            // Release links the existing free slot without allocating. Reuse
            // keeps its original backing claim and advances only its generation.
            arena.release(last).unwrap();
            let replacement = arena.insert(Value::Num(2.0), Some(&runtime)).unwrap();
            assert_eq!(replacement.handle, last.handle);
            assert_ne!(replacement.generation, last.generation);
            assert!(arena.get(last).is_err());
            assert_eq!(ledger.snapshot().live_claimed_bytes, claimed);
            assert!(arena.insert(Value::Num(3.0), None).is_err());
            drop(arena);
            drop(guard);
            drop(runtime);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        });
    }

    #[test]
    fn retirement_preserves_marked_aliases_and_explicit_pins() {
        let mut arena = ValueArena::new().unwrap();
        let alias = arena.insert(Value::Num(1.0), None).unwrap();
        let pinned = arena.insert(Value::Num(2.0), None).unwrap();
        let dead = arena.insert(Value::Num(3.0), None).unwrap();
        arena.retain(pinned).unwrap();
        arena.begin_retirement();
        arena.mark_resident(alias).unwrap();
        arena.mark_resident(alias).unwrap();
        arena.finish_retirement().unwrap();
        assert_eq!(arena.get(alias).unwrap(), &Value::Num(1.0));
        assert_eq!(arena.get(pinned).unwrap(), &Value::Num(2.0));
        assert!(matches!(
            arena.get(dead),
            Err(NativeExecutorError::StaleValue)
        ));
        let reused = arena.insert(Value::Num(4.0), None).unwrap();
        assert_eq!(dead.handle, reused.handle);
        assert_ne!(dead.generation, reused.generation);
        // Marking a stale owner is an error, not a silent alias of a reused slot.
        assert!(matches!(
            arena.mark_resident(dead),
            Err(NativeExecutorError::StaleValue)
        ));
        arena.begin_retirement();
        arena.mark_resident(alias).unwrap();
        arena.finish_retirement().unwrap();
        assert!(!arena.has_retirement_debt());
        arena.release(pinned).unwrap();
        assert!(arena.has_retirement_debt());
        arena.begin_retirement();
        arena.mark_resident(alias).unwrap();
        arena.finish_retirement().unwrap();
        assert!(matches!(
            arena.get(pinned),
            Err(NativeExecutorError::StaleValue)
        ));
    }

    #[test]
    fn retirement_runs_only_after_amortized_allocation_debt() {
        let mut arena = ValueArena::new().unwrap();
        assert!(!arena.has_retirement_debt());
        for _ in 0..63 {
            arena.insert(Value::Num(1.0), None).unwrap();
        }
        assert!(!arena.retirement_due());
        assert!(arena.has_retirement_debt());
        arena.insert(Value::Num(2.0), None).unwrap();
        assert!(arena.retirement_due());
        arena.begin_retirement();
        arena.finish_retirement().unwrap();
        assert_eq!(arena.retirement_stats().0, 0);
        assert!(!arena.has_retirement_debt());
        assert!(!arena.retirement_due());
        for _ in 0..64 {
            arena.insert(Value::Num(3.0), None).unwrap();
        }
        assert!(arena.retirement_due());
        assert_eq!(arena.retirement_stats().1, 64);
    }

    #[test]
    fn retirement_preserves_child_owned_parent_after_original_release() {
        let mut arena = ValueArena::new().unwrap();
        let parent = arena
            .insert(
                Value::Cell(runmat_value::CellArray::new(vec![Value::Num(9.0)], 1, 1).unwrap()),
                None,
            )
            .unwrap();
        let child = arena.insert_cell_element(parent, 0, None).unwrap();
        arena.release(parent).unwrap();
        arena.begin_retirement();
        arena.mark_resident(child).unwrap();
        arena.finish_retirement().unwrap();
        assert_eq!(arena.get(child).unwrap(), &Value::Num(9.0));
        arena.begin_retirement();
        arena.finish_retirement().unwrap();
        assert!(matches!(
            arena.get(child),
            Err(NativeExecutorError::StaleValue)
        ));
        assert!(matches!(
            arena.get(parent),
            Err(NativeExecutorError::StaleValue)
        ));
    }

    #[test]
    fn retirement_cadence_bounds_shallow_payload_debt_without_traversal() {
        let mut arena = ValueArena::new().unwrap();
        let small =
            Value::Tensor(runmat_value::Tensor::new(vec![1.0; 1024], vec![1024, 1]).unwrap());
        for _ in 0..31 {
            arena.insert(small.clone(), None).unwrap();
        }
        assert!(!arena.retirement_due());
        arena.insert(small, None).unwrap();
        assert!(arena.retirement_due());
        arena.begin_retirement();
        arena.finish_retirement().unwrap();
        assert!(!arena.retirement_due());
        let large = arena
            .insert(
                Value::Tensor(runmat_value::Tensor::new(vec![1.0; 32768], vec![32768, 1]).unwrap()),
                None,
            )
            .unwrap();
        assert!(
            arena.retirement_due(),
            "one large payload forces next boundary"
        );
        arena.begin_retirement();
        arena.mark_resident(large).unwrap();
        arena.finish_retirement().unwrap();
        assert!(!arena.retirement_due());
        arena
            .insert(Value::String("opaque heap backing".into()), None)
            .unwrap();
        assert!(arena.retirement_due(), "unknown backing is conservative");
    }

    fn workspace_lease(value: Value) -> WorkspaceValueReadLease {
        use runmat_runtime::workspace::{
            BaseWorkspaceOwner, LogicalWorkspaceGeneration, WorkspaceDelta,
        };

        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let owner = BaseWorkspaceOwner::new();
        let mut delta = WorkspaceDelta::new();
        delta.assign("value", value).unwrap();
        owner.apply(delta).unwrap();
        LogicalWorkspaceGeneration::capture(&runtime, &owner, &runtime.session_variable_authority())
            .unwrap()
            .read_lease("value")
            .unwrap()
    }

    #[test]
    fn references_are_generation_checked_and_reuse_slots_without_aliasing() {
        let mut arena = ValueArena::new().unwrap();
        let first = arena.insert(Value::Num(1.0), None).unwrap();
        arena.retain(first).unwrap();
        arena.release(first).unwrap();
        assert_eq!(arena.get(first).unwrap(), &Value::Num(1.0));
        arena.release(first).unwrap();
        assert!(matches!(
            arena.get(first),
            Err(NativeExecutorError::StaleValue)
        ));

        let second = arena.insert(Value::Num(2.0), None).unwrap();
        assert_eq!(second.handle, first.handle);
        assert_ne!(second.generation, first.generation);
        assert_eq!(arena.get(second).unwrap(), &Value::Num(2.0));

        let lease = workspace_lease(Value::String("immutable input".into()));
        let original = lease.value() as *const Value;
        let leased = arena.insert_retained_lease(lease.clone(), None).unwrap();
        assert!(std::ptr::eq(arena.get(leased).unwrap(), original));
        assert_eq!(
            arena
                .reuse_or_insert_retained_lease(leased, lease, None)
                .unwrap(),
            leased
        );
        arena.retain(leased).unwrap();
        arena.release(leased).unwrap();
        assert!(std::ptr::eq(arena.get(leased).unwrap(), original));
        arena.release(leased).unwrap();
        assert!(matches!(
            arena.get(leased),
            Err(NativeExecutorError::StaleValue)
        ));

        let cell = arena
            .insert(
                Value::Cell(runmat_value::CellArray::new(vec![Value::Num(9.0)], 1, 1).unwrap()),
                None,
            )
            .unwrap();
        let original_element = match arena.get(cell).unwrap() {
            Value::Cell(cell) => &cell.data()[0] as *const Value,
            _ => unreachable!(),
        };
        let element = arena.insert_cell_element(cell, 0, None).unwrap();
        arena.release(cell).unwrap();
        assert!(std::ptr::eq(arena.get(element).unwrap(), original_element));
        arena.release(element).unwrap();
        assert!(matches!(
            arena.get(cell),
            Err(NativeExecutorError::StaleValue)
        ));
    }

    #[test]
    fn null_is_a_noop_for_lifetime_but_never_materializes_a_value() {
        let mut arena = ValueArena::new().unwrap();
        arena.retain(NativeValueRef::NULL).unwrap();
        arena.release(NativeValueRef::NULL).unwrap();
        assert!(matches!(
            arena.get(NativeValueRef::NULL),
            Err(NativeExecutorError::StaleValue)
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn live_arena_values_are_visible_until_their_reference_is_released() {
        runmat_gc::gc_test_context(|| {
            let mut arena = ValueArena::new().unwrap();
            let reference = arena
                .insert(Value::String("retained-by-native".into()), None)
                .unwrap();
            let visible = || {
                let mut found = false;
                runmat_gc::gc_visit_retained_values(&[], &mut |value| {
                    found |= value == &Value::String("retained-by-native".into());
                    Some(())
                })
                .unwrap();
                found
            };
            assert!(visible());
            arena.release(reference).unwrap();
            assert!(!visible());
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn live_arena_values_are_traced_as_gc_roots() {
        if runmat_gc::test_support::run_isolated_gc_test(concat!(
            module_path!(),
            "::live_arena_values_are_traced_as_gc_roots"
        )) {
            return;
        }
        runmat_gc::gc_test_context(|| {
            let rooted = runmat_gc::gc_allocate_rooted(Value::String("live".to_string().into()))
                .expect("allocate rooted test value");
            let handle = rooted.handle();
            let mut arena = ValueArena::new().expect("activate arena roots");
            arena
                .insert(
                    Value::HandleObject(runmat_value::HandleRef {
                        class_name: "test".into(),
                        target: handle,
                        valid: true,
                    }),
                    None,
                )
                .unwrap();
            arena
                .insert_retained_lease(
                    workspace_lease(Value::HandleObject(runmat_value::HandleRef {
                        class_name: "test".into(),
                        target: handle,
                        valid: true,
                    })),
                    None,
                )
                .unwrap();
            rooted.unroot().expect("transfer ownership to arena root");

            runmat_gc::gc_collect_major().expect("collect with live arena root");
            let value = runmat_gc::gc_clone_value(&handle).expect("rooted value remains live");
            assert_eq!(value, Value::String("live".to_string().into()));

            drop(arena);
            runmat_gc::gc_collect_major().expect("collect after arena teardown");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}
