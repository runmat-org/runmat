#[cfg(not(target_arch = "wasm32"))]
use std::ptr::NonNull;

#[cfg(not(target_arch = "wasm32"))]
use runmat_gc::{GcHandle, GcRoot, RootId, Trace, Tracer};
use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

use runmat_runtime::call::CapturedValueLease;
use runmat_runtime::workspace::WorkspaceSlotReadLease;

use crate::{NativeExecutorError, NativeExecutorResult};

#[derive(Debug)]
struct Entry {
    generation: u64,
    references: usize,
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
    free: Vec<usize>,
    #[cfg(not(target_arch = "wasm32"))]
    root_id: Option<RootId>,
}

impl ValueArena {
    pub fn new() -> NativeExecutorResult<Self> {
        let arena = Self {
            entries: Box::default(),
            free: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            root_id: None,
        };
        #[cfg(target_arch = "wasm32")]
        {
            Ok(arena)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut arena = arena;
            arena.activate_gc_roots()?;
            Ok(arena)
        }
    }

    /// Register every live value in this invocation as a GC root source.
    #[cfg(not(target_arch = "wasm32"))]
    fn activate_gc_roots(&mut self) -> NativeExecutorResult<()> {
        if self.root_id.is_some() {
            return Ok(());
        }
        // SAFETY: `entries` is boxed and therefore remains at this address even
        // if `ValueArena` moves. `Drop` unregisters the scanner before the box
        // is released. Collection is deferred when another thread owns roots,
        // and native execution never scans concurrently with arena mutation on
        // the owning thread.
        let root = unsafe {
            ArenaRoot::new(
                NonNull::from(self.entries.as_mut()),
                "native_value_arena".to_string(),
            )
        };
        self.root_id = Some(runmat_gc::gc_register_value_root(Box::new(root)).map_err(
            |error| NativeExecutorError::Host(format!("register native GC roots: {error}")),
        )?);
        Ok(())
    }

    pub fn insert(&mut self, value: Value) -> NativeValueRef {
        self.insert_entry(EntryValue::Owned(value))
    }

    /// Retain an executor input without materializing another language value.
    pub(crate) fn insert_retained_lease(
        &mut self,
        value: impl Into<WorkspaceSlotReadLease>,
    ) -> NativeValueRef {
        self.insert_entry(EntryValue::Retained(value.into()))
    }

    pub(crate) fn insert_capture_lease(&mut self, lease: CapturedValueLease) -> NativeValueRef {
        self.insert_entry(EntryValue::Capture(lease))
    }

    pub(crate) fn insert_cell_element(
        &mut self,
        parent: NativeValueRef,
        index: usize,
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
        if cell.data.get(index).is_none() {
            return Err(NativeExecutorError::StaleValue);
        }
        self.retain(parent)?;
        Ok(self.insert_entry(EntryValue::CellElement { parent, index }))
    }

    /// Refresh a local's immutable generation without adding an arena entry
    /// for an unchanged binding. Older entries remain valid for SSA operands.
    pub(crate) fn reuse_or_insert_retained_lease(
        &mut self,
        previous: NativeValueRef,
        value: impl Into<WorkspaceSlotReadLease>,
    ) -> NativeExecutorResult<NativeValueRef> {
        let value = value.into();
        if !previous.is_null() && std::ptr::eq(self.get(previous)?, value.value()) {
            return Ok(previous);
        }
        Ok(self.insert_retained_lease(value))
    }

    fn insert_entry(&mut self, value: EntryValue) -> NativeValueRef {
        let index = self.free.pop().unwrap_or_else(|| {
            self.entries.values.push(Entry {
                generation: 1,
                references: 0,
                value: None,
            });
            self.entries.values.len() - 1
        });
        let entry = &mut self.entries.values[index];
        entry.references = 1;
        entry.value = Some(value);
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
                cell.data.get(*index).ok_or(NativeExecutorError::StaleValue)
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
        let retired = if entry.references == 0 {
            let retired = entry.value.take();
            // A wrapped generation could make a prehistoric reference valid
            // again. Retire the slot permanently instead of permitting ABA.
            if let Some(next_generation) = entry.generation.checked_add(1) {
                entry.generation = next_generation;
                self.free.push(index);
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
    description: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl ArenaRoot {
    /// # Safety
    ///
    /// `entries` must remain allocated until this root is unregistered, and it
    /// must not be scanned concurrently with mutation.
    unsafe fn new(entries: NonNull<ArenaEntries>, description: String) -> Self {
        Self {
            entries,
            description,
        }
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
        self.description.clone()
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
        let first = arena.insert(Value::Num(1.0));
        arena.retain(first).unwrap();
        arena.release(first).unwrap();
        assert_eq!(arena.get(first).unwrap(), &Value::Num(1.0));
        arena.release(first).unwrap();
        assert!(matches!(
            arena.get(first),
            Err(NativeExecutorError::StaleValue)
        ));

        let second = arena.insert(Value::Num(2.0));
        assert_eq!(second.handle, first.handle);
        assert_ne!(second.generation, first.generation);
        assert_eq!(arena.get(second).unwrap(), &Value::Num(2.0));

        let lease = workspace_lease(Value::String("immutable input".into()));
        let original = lease.value() as *const Value;
        let leased = arena.insert_retained_lease(lease.clone());
        assert!(std::ptr::eq(arena.get(leased).unwrap(), original));
        assert_eq!(
            arena.reuse_or_insert_retained_lease(leased, lease).unwrap(),
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

        let cell = arena.insert(Value::Cell(
            runmat_value::CellArray::new(vec![Value::Num(9.0)], 1, 1).unwrap(),
        ));
        let original_element = match arena.get(cell).unwrap() {
            Value::Cell(cell) => &cell.data[0] as *const Value,
            _ => unreachable!(),
        };
        let element = arena.insert_cell_element(cell, 0).unwrap();
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
            let reference = arena.insert(Value::String("retained-by-native".into()));
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
        if crate::test_support::run_isolated_gc_test(concat!(
            module_path!(),
            "::live_arena_values_are_traced_as_gc_roots"
        )) {
            return;
        }
        runmat_gc::gc_test_context(|| {
            let rooted = runmat_gc::gc_allocate_rooted(Value::String("live".to_string()))
                .expect("allocate rooted test value");
            let handle = rooted.handle();
            let mut arena = ValueArena::new().expect("activate arena roots");
            arena.insert(Value::HandleObject(runmat_value::HandleRef {
                class_name: "test".into(),
                target: handle,
                valid: true,
            }));
            arena.insert_retained_lease(workspace_lease(Value::HandleObject(
                runmat_value::HandleRef {
                    class_name: "test".into(),
                    target: handle,
                    valid: true,
                },
            )));
            rooted.unroot().expect("transfer ownership to arena root");

            runmat_gc::gc_collect_major().expect("collect with live arena root");
            let value = runmat_gc::gc_clone_value(&handle).expect("rooted value remains live");
            assert_eq!(value, Value::String("live".to_string()));

            drop(arena);
            runmat_gc::gc_collect_major().expect("collect after arena teardown");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}
