use crate::runtime::workspace::FrameVec;
use runmat_execution::resource::ExactShared;
use runmat_runtime::call::CapturedValueLease;
use runmat_runtime::context::host_allocation::{
    collection_block_retained_bytes, FrameCollectionGate, HostAllocationError, HostAllocationPlan,
    HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::WorkspaceSlotReadLease;
use runmat_value::{ExactHostAllocationClaim, Value};
use std::alloc::Layout;
use std::cell::{Ref, RefCell};
use std::ops::Deref;

#[cfg(test)]
mod extent_tests;
mod workspace_input;

#[derive(Clone, Debug)]
pub(crate) enum InterpreterSlot {
    Owned(Value),
    Retained(WorkspaceSlotReadLease),
    Captured(CapturedValueLease),
}

impl InterpreterSlot {
    pub(crate) fn retained(value: WorkspaceSlotReadLease) -> Self {
        Self::Retained(value)
    }

    fn value(&self) -> &Value {
        match self {
            Self::Owned(value) => value,
            Self::Retained(value) => value.value(),
            Self::Captured(value) => value.value(),
        }
    }

    fn replace(&mut self, value: Value) -> Self {
        std::mem::replace(self, Self::Owned(value))
    }

    fn replace_lexical(
        &mut self,
        value: runmat_runtime::call::lexical::LexicalBindingReadLease,
    ) -> Self {
        std::mem::replace(self, Self::Retained(value.into()))
    }

    fn retained_lease(&self) -> Option<WorkspaceSlotReadLease> {
        match self {
            Self::Owned(_) => None,
            Self::Retained(value) => Some(value.clone()),
            Self::Captured(_) => None,
        }
    }
}

#[derive(Debug)]
enum InterpreterVarsStorage {
    Owned(Vec<Value>),
    Retained(FrameVec<InterpreterSlot>),
}

/// Empty backing displaced while an admitted slot vector grows. Keeping the
/// allocation owner here lets callers retire its block after releasing any
/// workspace or shared-control borrow.
pub(crate) struct InterpreterVarsRetirement {
    _storage: InterpreterVarsStorage,
}

impl InterpreterVarsStorage {
    fn len(&self) -> usize {
        match self {
            Self::Owned(values) => values.len(),
            Self::Retained(values) => values.len(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Owned(values) => values.capacity(),
            Self::Retained(values) => values.capacity(),
        }
    }

    fn value(&self, slot: usize) -> Option<&Value> {
        match self {
            Self::Owned(values) => values.get(slot),
            Self::Retained(values) => values.get(slot).map(InterpreterSlot::value),
        }
    }

    fn value_slot(&self, slot: usize) -> Option<&InterpreterSlot> {
        match self {
            Self::Owned(_) => None,
            Self::Retained(values) => values.get(slot),
        }
    }
}

#[derive(Debug)]
struct SharedInterpreterVars {
    storage: RefCell<InterpreterVarsStorage>,
    _claim: Option<ExactHostAllocationClaim>,
}

/// The single active slot owner for an interpreter frame. Ordinary calls keep
/// their original `Vec<Value>` allocation. Workspace admission promotes that
/// allocation into one exact shared control, so interpreter execution and
/// workspace metadata never maintain duplicate payload vectors.
#[derive(Debug)]
pub struct InterpreterVars {
    storage: InterpreterVarsOwner,
}

#[derive(Debug)]
enum InterpreterVarsOwner {
    Direct(InterpreterVarsStorage),
    Shared(ExactShared<SharedInterpreterVars>),
}

pub struct InterpreterValueRef<'a>(InterpreterValueRefInner<'a>);

enum InterpreterValueRefInner<'a> {
    Direct(&'a Value),
    Shared(Ref<'a, Value>),
}

/// One stable read of an interpreter frame's physical slot owner.
///
/// Workspace completion keeps this guard for the complete borrowed-source
/// transaction. Shared frames therefore never expose a reference past their
/// `RefCell` borrow, while direct frames retain their allocation-free path.
pub(crate) struct InterpreterVarsRead<'a>(InterpreterVarsReadInner<'a>);

enum InterpreterVarsReadInner<'a> {
    Direct(&'a InterpreterVarsStorage),
    Shared(Ref<'a, InterpreterVarsStorage>),
}

impl InterpreterVarsRead<'_> {
    pub(crate) fn value(&self, slot: usize) -> Option<&Value> {
        match &self.0 {
            InterpreterVarsReadInner::Direct(storage) => storage.value(slot),
            InterpreterVarsReadInner::Shared(storage) => storage.value(slot),
        }
    }

    pub(crate) fn retained_lease(&self, slot: usize) -> Option<WorkspaceSlotReadLease> {
        match &self.0 {
            InterpreterVarsReadInner::Direct(storage) => storage
                .value_slot(slot)
                .and_then(InterpreterSlot::retained_lease),
            InterpreterVarsReadInner::Shared(storage) => storage
                .value_slot(slot)
                .and_then(InterpreterSlot::retained_lease),
        }
    }
}

impl Deref for InterpreterValueRef<'_> {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        match &self.0 {
            InterpreterValueRefInner::Direct(value) => value,
            InterpreterValueRefInner::Shared(value) => value,
        }
    }
}

impl<'a> InterpreterValueRef<'a> {
    /// Wrap a value whose owner already outlives the read. This lets scoped
    /// consumers handle static and shared-frame inputs uniformly without
    /// erasing a shared storage guard into a naked reference.
    pub(crate) fn borrowed(value: &'a Value) -> Self {
        Self(InterpreterValueRefInner::Direct(value))
    }
}

impl InterpreterVars {
    pub fn from_owned(values: Vec<Value>) -> Self {
        Self {
            storage: InterpreterVarsOwner::Direct(InterpreterVarsStorage::Owned(values)),
        }
    }

    pub(crate) fn from_retained(values: FrameVec<InterpreterSlot>) -> Self {
        Self {
            storage: InterpreterVarsOwner::Direct(InterpreterVarsStorage::Retained(values)),
        }
    }

    /// Prepare the exact slot backing used when a function frame retains
    /// immutable closure captures. Ordinary functions keep their original
    /// `Vec<Value>` allocation and never pay for this owner.
    pub(crate) fn prepare_captured_frame(
        context: &RuntimeContext,
        len: usize,
    ) -> Result<Self, HostAllocationError> {
        let layout =
            Layout::array::<InterpreterSlot>(len).map_err(|_| HostAllocationError::Overflow)?;
        let retained = FrameCollectionGate::retained_bytes()?
            .checked_add(collection_block_retained_bytes(layout)?)
            .ok_or(HostAllocationError::Overflow)?;
        let plan = HostAllocationPlan::new(0, retained)?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)?;
        let gate = match &receipt {
            Some(receipt) => FrameCollectionGate::prepare(context, receipt)?,
            None => FrameCollectionGate::unbounded(),
        };
        let installed = match &receipt {
            Some(receipt) => gate.install(context, receipt)?,
            None => gate.install_unbounded()?,
        };
        let mut values = FrameVec::new_in(gate.allocator());
        values.try_reserve_exact(len).map_err(|_| {
            HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            )
        })?;
        values.resize_with(len, || InterpreterSlot::Owned(Value::Num(0.0)));
        if let Some(receipt) = &receipt {
            receipt.checkpoint(context)?;
        }
        drop(installed);
        Ok(Self::from_retained(values))
    }

    /// Move an owned frame into an exactly admitted retained-slot
    /// backing large enough for compiler-private locals. The caller keeps the
    /// source Values rooted while this transfer runs; no payload is cloned.
    pub fn prepare_extent(
        mut self,
        context: &RuntimeContext,
        extent: usize,
    ) -> Result<Self, runmat_runtime::RuntimeError> {
        HostAllocationReceipt::policy(Some(context)).map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
        })?;
        if extent <= self.len() {
            return Ok(self);
        }

        let layout = Layout::array::<InterpreterSlot>(extent)
            .map_err(|_| HostAllocationError::Overflow)
            .map_err(|error| {
                crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
            })?;
        let retained = FrameCollectionGate::retained_bytes()
            .and_then(|gate| {
                gate.checked_add(collection_block_retained_bytes(layout)?)
                    .ok_or(HostAllocationError::Overflow)
            })
            .map_err(|error| {
                crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
            })?;
        let plan = HostAllocationPlan::new(0, retained).map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
        })?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan).map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
        })?;
        let gate = match &receipt {
            Some(receipt) => FrameCollectionGate::prepare(context, receipt),
            None => Ok(FrameCollectionGate::unbounded()),
        }
        .map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
        })?;
        let installed = match &receipt {
            Some(receipt) => gate.install(context, receipt),
            None => gate.install_unbounded(),
        }
        .map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
        })?;
        let mut retained_values = FrameVec::new_in(gate.allocator());
        retained_values.try_reserve_exact(extent).map_err(|_| {
            crate::runtime::workspace::frame_allocation_error_in_context(
                Some(context),
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        })?;
        if let Some(receipt) = &receipt {
            receipt.checkpoint(context).map_err(|error| {
                crate::runtime::workspace::frame_allocation_error_in_context(Some(context), error)
            })?;
        }
        // Install under the same shared control, when present. The existing
        // transfer owner moves slots and defers displaced block retirement
        // until its shared-storage borrow has ended; no Value is cloned.
        let retired = self.install_replacement(retained_values);
        self.resize_owned(extent, Value::Num(0.0));
        drop(installed);
        drop(retired);
        Ok(self)
    }

    pub(crate) fn shared_control_retained_bytes() -> Result<u64, HostAllocationError> {
        let physical = u64::try_from(ExactShared::<SharedInterpreterVars>::requested_bytes())
            .map_err(|_| HostAllocationError::Overflow)?;
        runmat_runtime::context::host_allocation::HostAllocationPlan::retained_exact_output_bytes(
            physical, 1,
        )
    }

    /// Move this frame's active storage into one exact shared control and
    /// return the workspace's handle to that same control.
    pub(crate) fn promote_shared(
        &mut self,
        context: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
    ) -> Result<Self, HostAllocationError> {
        if let InterpreterVarsOwner::Shared(shared) = &self.storage {
            return Ok(Self {
                storage: InterpreterVarsOwner::Shared(shared.clone()),
            });
        }

        let direct = match std::mem::replace(
            &mut self.storage,
            InterpreterVarsOwner::Direct(InterpreterVarsStorage::Owned(Vec::new())),
        ) {
            InterpreterVarsOwner::Direct(storage) => storage,
            InterpreterVarsOwner::Shared(_) => unreachable!(),
        };
        let shared = match (context, receipt) {
            (Some(context), Some(receipt)) => receipt.with_output_allocation(context, |permit| {
                let claim = permit
                    .reserve_exact_claim(ExactShared::<SharedInterpreterVars>::requested_bytes())?;
                ExactShared::try_new(SharedInterpreterVars {
                    storage: RefCell::new(InterpreterVarsStorage::Owned(Vec::new())),
                    _claim: Some(claim),
                })
                .map_err(|_| runmat_value::ExactHostNumericAllocationError::AllocationFailed)
            }),
            _ => Ok(ExactShared::new(SharedInterpreterVars {
                storage: RefCell::new(InterpreterVarsStorage::Owned(Vec::new())),
                _claim: None,
            })),
        };
        let shared = match shared {
            Ok(shared) => shared,
            Err(error) => {
                self.storage = InterpreterVarsOwner::Direct(direct);
                return Err(error);
            }
        };
        *shared.get().storage.borrow_mut() = direct;
        self.storage = InterpreterVarsOwner::Shared(shared.clone());
        Ok(Self {
            storage: InterpreterVarsOwner::Shared(shared),
        })
    }

    pub fn len(&self) -> usize {
        match &self.storage {
            InterpreterVarsOwner::Direct(storage) => storage.len(),
            InterpreterVarsOwner::Shared(shared) => shared.get().storage.borrow().len(),
        }
    }

    pub fn capacity(&self) -> usize {
        match &self.storage {
            InterpreterVarsOwner::Direct(storage) => storage.capacity(),
            InterpreterVarsOwner::Shared(shared) => shared.get().storage.borrow().capacity(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, slot: usize) -> Option<InterpreterValueRef<'_>> {
        match &self.storage {
            InterpreterVarsOwner::Direct(storage) => storage
                .value(slot)
                .map(|value| InterpreterValueRef(InterpreterValueRefInner::Direct(value))),
            InterpreterVarsOwner::Shared(shared) => {
                let storage = shared.get().storage.borrow();
                if slot >= storage.len() {
                    return None;
                }
                Some(InterpreterValueRef(InterpreterValueRefInner::Shared(
                    Ref::map(storage, |storage| {
                        storage
                            .value(slot)
                            .expect("interpreter slot was checked within bounds")
                    }),
                )))
            }
        }
    }

    pub(crate) fn read(&self) -> InterpreterVarsRead<'_> {
        match &self.storage {
            InterpreterVarsOwner::Direct(storage) => {
                InterpreterVarsRead(InterpreterVarsReadInner::Direct(storage))
            }
            InterpreterVarsOwner::Shared(shared) => InterpreterVarsRead(
                InterpreterVarsReadInner::Shared(shared.get().storage.borrow()),
            ),
        }
    }

    pub(crate) fn for_each(&self, mut visitor: impl FnMut(&Value)) {
        match &self.storage {
            InterpreterVarsOwner::Direct(storage) => {
                for slot in 0..storage.len() {
                    visitor(storage.value(slot).expect("slot is within frame extent"));
                }
            }
            InterpreterVarsOwner::Shared(shared) => {
                let storage = shared.get().storage.borrow();
                for slot in 0..storage.len() {
                    visitor(storage.value(slot).expect("slot is within frame extent"));
                }
            }
        }
    }

    pub(crate) fn iter(&self) -> InterpreterValues<'_> {
        InterpreterValues {
            vars: self,
            next: 0,
            len: self.len(),
        }
    }

    pub(crate) fn clone_values(&self) -> Vec<Value> {
        let mut values = Vec::with_capacity(self.len());
        self.for_each(|value| values.push(value.clone()));
        values
    }

    pub fn push_owned(&mut self, value: Value) {
        self.with_storage_mut(|storage| match storage {
            InterpreterVarsStorage::Owned(values) => values.push(value),
            InterpreterVarsStorage::Retained(values) => values.push(InterpreterSlot::Owned(value)),
        });
    }

    pub fn resize_owned(&mut self, len: usize, value: Value) {
        if len < self.len() {
            self.truncate(len);
            return;
        }
        self.with_storage_mut(|storage| match storage {
            InterpreterVarsStorage::Owned(values) => values.resize(len, value),
            InterpreterVarsStorage::Retained(values) => {
                values.resize_with(len, || InterpreterSlot::Owned(value.clone()))
            }
        });
    }

    pub(crate) fn replace_owned(&mut self, slot: usize, value: Value) -> Option<InterpreterSlot> {
        self.with_storage_mut(|storage| match storage {
            InterpreterVarsStorage::Owned(values) => values
                .get_mut(slot)
                .map(|current| InterpreterSlot::Owned(std::mem::replace(current, value))),
            InterpreterVarsStorage::Retained(values) => {
                values.get_mut(slot).map(|current| current.replace(value))
            }
        })
    }

    /// Move one compiler-private, uniquely consumed temporary out of its
    /// frame slot without publishing a workspace mutation. The bytecode proof
    /// authorizing this operation is checked by the dispatch caller. A
    /// retained workspace or captured binding is never converted into an
    /// owned value by cloning.
    pub(crate) fn take_private_owned(&mut self, slot: usize) -> Option<Value> {
        self.with_storage_mut(|storage| match storage {
            InterpreterVarsStorage::Owned(values) => values
                .get_mut(slot)
                .map(|value| std::mem::replace(value, Value::Num(0.0))),
            InterpreterVarsStorage::Retained(values) => {
                let Some(InterpreterSlot::Owned(value)) = values.get_mut(slot) else {
                    return None;
                };
                Some(std::mem::replace(value, Value::Num(0.0)))
            }
        })
    }

    pub(crate) fn replace_retained(
        &mut self,
        slot: usize,
        value: WorkspaceSlotReadLease,
    ) -> Option<InterpreterSlot> {
        self.with_storage_mut(|storage| match storage {
            InterpreterVarsStorage::Owned(_) => {
                panic!("retained workspace write requires a prepared retained-slot replacement")
            }
            InterpreterVarsStorage::Retained(values) => values
                .get_mut(slot)
                .map(|current| std::mem::replace(current, InterpreterSlot::Retained(value))),
        })
    }

    pub(crate) fn replace_captured(
        &mut self,
        slot: usize,
        value: CapturedValueLease,
    ) -> Option<InterpreterSlot> {
        self.with_storage_mut(|storage| match storage {
            InterpreterVarsStorage::Owned(_) => {
                panic!("captured function input requires a prepared retained-slot frame")
            }
            InterpreterVarsStorage::Retained(values) => values
                .get_mut(slot)
                .map(|current| std::mem::replace(current, InterpreterSlot::Captured(value))),
        })
    }

    pub(crate) fn uses_owned_storage(&self) -> bool {
        match &self.storage {
            InterpreterVarsOwner::Direct(InterpreterVarsStorage::Owned(_)) => true,
            InterpreterVarsOwner::Direct(InterpreterVarsStorage::Retained(_)) => false,
            InterpreterVarsOwner::Shared(shared) => matches!(
                &*shared.get().storage.borrow(),
                InterpreterVarsStorage::Owned(_)
            ),
        }
    }

    pub(crate) fn retained_lease(&self, slot: usize) -> Option<WorkspaceSlotReadLease> {
        match &self.storage {
            InterpreterVarsOwner::Direct(InterpreterVarsStorage::Owned(_)) => None,
            InterpreterVarsOwner::Direct(InterpreterVarsStorage::Retained(values)) => {
                values.get(slot)?.retained_lease()
            }
            InterpreterVarsOwner::Shared(shared) => shared
                .get()
                .storage
                .borrow()
                .value_slot(slot)
                .and_then(InterpreterSlot::retained_lease),
        }
    }

    /// Retain the exact shared slot control without copying any Value. Frame
    /// completion uses this to hand the authoritative child generation back
    /// to the suspended parent interpreter.
    pub(crate) fn shared_clone_if_distinct(&self, current: &Self) -> Option<Self> {
        match (&self.storage, &current.storage) {
            (InterpreterVarsOwner::Shared(next), InterpreterVarsOwner::Shared(active))
                if next.ptr_eq(active) =>
            {
                None
            }
            (InterpreterVarsOwner::Shared(shared), _) => Some(Self {
                storage: InterpreterVarsOwner::Shared(shared.clone()),
            }),
            (InterpreterVarsOwner::Direct(_), _) => None,
        }
    }

    pub(crate) fn shares_storage_with(&self, other: &Self) -> bool {
        matches!(
            (&self.storage, &other.storage),
            (InterpreterVarsOwner::Shared(left), InterpreterVarsOwner::Shared(right))
                if left.ptr_eq(right)
        )
    }

    pub(crate) fn has_shared_control(&self) -> bool {
        matches!(&self.storage, InterpreterVarsOwner::Shared(_))
    }

    /// Replace this interpreter's slot control with an already-admitted
    /// authoritative control. The displaced generation is returned so its
    /// Values and leases can retire after all workspace borrows are gone.
    pub(crate) fn adopt_shared(&mut self, source: Self) -> Option<Self> {
        let InterpreterVarsOwner::Shared(next) = source.storage else {
            return None;
        };
        if matches!(
            &self.storage,
            InterpreterVarsOwner::Shared(current) if current.ptr_eq(&next)
        ) {
            return None;
        }
        let retired = std::mem::replace(&mut self.storage, InterpreterVarsOwner::Shared(next));
        Some(Self { storage: retired })
    }

    pub(crate) fn replace_lexical(
        &mut self,
        slot: usize,
        value: runmat_runtime::call::lexical::LexicalBindingReadLease,
    ) -> Option<InterpreterSlot> {
        self.with_storage_mut(|storage| match storage {
            InterpreterVarsStorage::Owned(values) => values.get_mut(slot).map(|current| {
                InterpreterSlot::Owned(std::mem::replace(current, value.value().clone()))
            }),
            InterpreterVarsStorage::Retained(values) => values
                .get_mut(slot)
                .map(|current| current.replace_lexical(value)),
        })
    }

    pub(crate) fn install_replacement(
        &mut self,
        mut replacement: FrameVec<InterpreterSlot>,
    ) -> InterpreterVarsRetirement {
        self.with_storage_mut(|storage| {
            let previous = std::mem::replace(storage, InterpreterVarsStorage::Owned(Vec::new()));
            let retired = match previous {
                InterpreterVarsStorage::Owned(mut values) => {
                    replacement.extend(values.drain(..).map(InterpreterSlot::Owned));
                    InterpreterVarsStorage::Owned(values)
                }
                InterpreterVarsStorage::Retained(mut values) => {
                    replacement.extend(values.drain(..));
                    InterpreterVarsStorage::Retained(values)
                }
            };
            *storage = InterpreterVarsStorage::Retained(replacement);
            InterpreterVarsRetirement { _storage: retired }
        })
    }

    pub fn truncate(&mut self, len: usize) {
        while self.len() > len {
            let retired = self.with_storage_mut(|storage| match storage {
                InterpreterVarsStorage::Owned(values) => values.pop().map(InterpreterSlot::Owned),
                InterpreterVarsStorage::Retained(values) => values.pop(),
            });
            // Value/lease destruction may retire GC or foreign resources and
            // therefore must occur after releasing the shared frame borrow.
            drop(retired);
        }
    }

    pub fn into_owned(self) -> Result<Vec<Value>, Self> {
        match self.storage {
            InterpreterVarsOwner::Direct(InterpreterVarsStorage::Owned(values)) => Ok(values),
            InterpreterVarsOwner::Direct(InterpreterVarsStorage::Retained(values)) => Err(Self {
                storage: InterpreterVarsOwner::Direct(InterpreterVarsStorage::Retained(values)),
            }),
            InterpreterVarsOwner::Shared(shared) => {
                // The workspace completion may retain the same immutable
                // terminal control. Until public result Values can carry a
                // SharedValueGeneration directly, this detached result ABI is
                // the explicit remaining all-class materialization boundary.
                let storage = shared.get().storage.borrow();
                Ok((0..storage.len())
                    .map(|slot| {
                        storage
                            .value(slot)
                            .expect("terminal slot lies within frame extent")
                            .clone()
                    })
                    .collect())
            }
        }
    }

    fn with_storage_mut<R>(
        &mut self,
        operation: impl FnOnce(&mut InterpreterVarsStorage) -> R,
    ) -> R {
        match &mut self.storage {
            InterpreterVarsOwner::Direct(storage) => operation(storage),
            InterpreterVarsOwner::Shared(shared) => {
                operation(&mut shared.get().storage.borrow_mut())
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn clone_values_for_test(&self) -> Vec<Value> {
        self.clone_values()
    }
}

impl From<Vec<Value>> for InterpreterVars {
    fn from(values: Vec<Value>) -> Self {
        Self::from_owned(values)
    }
}

impl PartialEq<Vec<Value>> for InterpreterVars {
    fn eq(&self, other: &Vec<Value>) -> bool {
        self.len() == other.len()
            && (0..other.len())
                .all(|slot| self.get(slot).is_some_and(|value| *value == other[slot]))
    }
}

impl runmat_gc::Trace for InterpreterVars {
    fn trace(&self, tracer: &mut dyn runmat_gc::Tracer) {
        self.for_each(|value| runmat_gc::Trace::trace(value, tracer));
    }
}

pub(crate) struct InterpreterValues<'a> {
    vars: &'a InterpreterVars,
    next: usize,
    len: usize,
}

impl<'a> Iterator for InterpreterValues<'a> {
    type Item = InterpreterValueRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.len {
            return None;
        }
        let slot = self.next;
        self.next += 1;
        self.vars.get(slot)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for InterpreterValues<'_> {}
