//! Release consumed VM values only after complete alias visibility is available.
use super::dispatch::SequenceState;
use runmat_accelerate_api::{handle_identity, AccelProvider, GpuHandleIdentity, GpuTensorHandle};
use runmat_gc::RootId;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};
use std::ptr::NonNull;

mod arguments;

struct Allocation {
    handle: GpuTensorHandle,
    owner: &'static dyn AccelProvider,
}

#[derive(Default)]
struct AllocationCollector(HashMap<GpuHandleIdentity, GpuTensorHandle>);
impl AllocationCollector {
    fn visit(&mut self, value: &Value) -> Result<(), String> {
        runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
            self.0
                .entry(handle_identity(handle))
                .or_insert_with(|| handle.clone());
        })
        .map_err(|error| error.to_string())
    }
    fn finish(self) -> Result<Vec<Allocation>, String> {
        self.0
            .into_values()
            .map(|handle| {
                let owner = runmat_accelerate_api::provider_for_handle(&handle)
                    .ok_or_else(|| "resident value owner is unavailable".to_string())?;
                Ok(Allocation { handle, owner })
            })
            .collect()
    }
}

fn allocations(values: &[Value]) -> Result<Vec<Allocation>, String> {
    let mut collector = AllocationCollector::default();
    for value in values {
        collector.visit(value)?;
    }
    collector.finish()
}

fn retain_value(value: &Value, keep: &mut HashSet<GpuHandleIdentity>) -> Option<()> {
    runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
        keep.insert(handle_identity(handle));
    })
    .ok()
}

fn retained(
    excluded: &[RootId],
    extra: &HashSet<GpuHandleIdentity>,
) -> Result<HashSet<GpuHandleIdentity>, String> {
    let mut keep = extra.clone();
    let mut visit = |value: &Value| retain_value(value, &mut keep);
    runmat_gc::gc_visit_retained_values(excluded, &mut visit)
        .ok_or_else(|| "registered value roots are unavailable".to_string())?;
    runmat_runtime::workspace::session::visit_roots(&mut visit)
        .ok_or_else(|| "session retained values are unavailable".to_string())?;
    runmat_runtime::class_registry::visit_static_property_retained_values(&mut visit)
        .ok_or_else(|| "class retained values are unavailable".to_string())?;
    if let Some(runtime) = runmat_runtime::context::legacy::active() {
        if let Some(calls) = runtime.service_ports().call() {
            calls
                .visit_retained_values(&mut visit)
                .ok_or_else(|| "call service retained values are unavailable".to_string())?;
        }
        if let Some(foreign) = runtime.service_ports().foreign() {
            foreign
                .visit_retained_values(&mut visit)
                .ok_or_else(|| "foreign service retained values are unavailable".to_string())?;
        }
        if let Some(distributed) = runtime.service_ports().distributed() {
            distributed
                .visit_retained_values(&mut visit)
                .ok_or_else(|| "distributed service retained values are unavailable".to_string())?;
        }
        runtime
            .execution()
            .visit_retained_values(&mut visit)
            .ok_or_else(|| "execution retained values are unavailable".to_string())?;
        if let Some(objects) = runtime.service_ports().object() {
            objects
                .visit_retained_values(&mut visit)
                .ok_or_else(|| "object service retained values are unavailable".to_string())?;
        }
        if let Some(workspace) = runtime.service_ports().workspace() {
            workspace
                .visit_retained_values(&mut visit)
                .ok_or_else(|| "workspace retained values are unavailable".to_string())?;
        }
    }
    Ok(keep)
}

fn release(owned: Vec<Allocation>, excluded: &[RootId], extra: &HashSet<GpuHandleIdentity>) {
    if owned.is_empty() {
        return;
    }
    let keep = match retained(excluded, extra) {
        Ok(keep) => keep,
        Err(reason) => {
            log::debug!(
                "VM resident cleanup skipped because alias visibility is unavailable: {reason}"
            );
            return;
        }
    };
    for allocation in owned {
        if !keep.contains(&handle_identity(&allocation.handle)) {
            match allocation.owner.free(&allocation.handle) {
                Ok(()) => runmat_accelerate_api::clear_handle_metadata(&allocation.handle),
                Err(error) => log::warn!("VM resident release failed: {error}"),
            }
        }
    }
}

/// Declared before a suspended dispatch future; therefore that future is
/// destroyed before this guard scans the surviving frame and published roots.
struct Registration(RootId);
impl Drop for Registration {
    fn drop(&mut self) {
        let _ = runmat_gc::gc_unregister_root(self.0);
    }
}

/// Public `interpret_with_vars` borrows the caller's inputs. Its error/drop
/// path leaves them untouched; only successful workspace publication retires
/// inputs that the replacement no longer owns.
pub(super) struct BorrowedInputs(ConsumedValues);
impl BorrowedInputs {
    pub fn new(values: &[Value]) -> Result<Self, String> {
        let allocations = allocations(values)?;
        let root = arguments::ArgumentRoot::new(
            values,
            allocations
                .iter()
                .map(|allocation| allocation.handle.clone()),
        );
        let root = if root.is_empty() {
            None
        } else {
            Some(Registration(
                runmat_gc::gc_register_value_root(Box::new(root))
                    .map_err(|error| error.to_string())?,
            ))
        };
        Ok(Self(ConsumedValues {
            allocations: Some(allocations),
            root,
            keep: Some(HashSet::new()),
        }))
    }

    pub fn complete(mut self, outputs: &[Value]) {
        self.0.protect(outputs);
        if let (Some(values), Some(keep)) = (self.0.allocations.take(), &self.0.keep) {
            // Publication is complete. The old compact root is no longer a
            // live caller owner; all other registered owners still participate.
            self.0.root.take();
            release(values, &[], keep);
        }
    }
}
impl Drop for BorrowedInputs {
    fn drop(&mut self) {
        // Abandonment does not consume the caller-owned input workspace.
        self.0.allocations = None;
    }
}

pub(crate) struct ConsumedValues {
    allocations: Option<Vec<Allocation>>,
    root: Option<Registration>,
    keep: Option<HashSet<GpuHandleIdentity>>,
}
impl ConsumedValues {
    pub fn new(values: &[Value]) -> Self {
        match allocations(values) {
            Ok(allocations) if allocations.is_empty() => Self {
                allocations: None,
                root: None,
                keep: Some(HashSet::new()),
            },
            Ok(allocations) => {
                // Suspended callees own the moved arguments. Publish that
                // lifetime through the existing root owner so another nested
                // or interleaved VM frame cannot retire the same alias.
                let root =
                    runmat_gc::gc_register_value_root(Box::new(arguments::ArgumentRoot::new(
                        values,
                        allocations
                            .iter()
                            .map(|allocation| allocation.handle.clone()),
                    )));
                match root {
                    Ok(root) => Self {
                        allocations: Some(allocations),
                        root: Some(Registration(root)),
                        keep: Some(HashSet::new()),
                    },
                    Err(reason) => {
                        log::warn!("VM consumed-value registration unavailable: {reason}");
                        Self {
                            allocations: None,
                            root: None,
                            keep: Some(HashSet::new()),
                        }
                    }
                }
            }
            Err(reason) => {
                log::warn!("VM consumed-value ownership unavailable: {reason}");
                Self {
                    allocations: None,
                    root: None,
                    keep: Some(HashSet::new()),
                }
            }
        }
    }
    pub fn protect(&mut self, values: &[Value]) {
        if self.allocations.is_none() {
            return;
        }
        for value in values {
            if let Some(keep) = &mut self.keep {
                if retain_value(value, keep).is_none() {
                    self.keep = None;
                    log::warn!("VM protected output aliases are unavailable");
                    break;
                }
            }
        }
    }
    pub fn finish(mut self, outputs: &[Value]) {
        self.protect(outputs);
    }
}
impl Drop for ConsumedValues {
    fn drop(&mut self) {
        if let (Some(values), Some(keep)) = (self.allocations.take(), &self.keep) {
            let excluded = self.root.iter().map(|root| root.0).collect::<Vec<_>>();
            release(values, &excluded, keep);
        }
    }
}

/// Same pinned-frame lifetime as its GC registrations. The inner instruction
/// future is dropped first; this guard precedes all frame values in drop order.
pub(super) struct FrameValues {
    stack: NonNull<Vec<Value>>,
    vars: NonNull<Vec<Value>>,
    locals: NonNull<Vec<Value>>,
    sequence: NonNull<SequenceState>,
    registrations: Vec<RootId>,
    active: bool,
}
impl FrameValues {
    /// # Safety
    /// All referenced frame owners must remain at these addresses and may not
    /// be mutated concurrently with this guard's terminal cleanup.
    pub unsafe fn new(
        stack: &Vec<Value>,
        vars: &Vec<Value>,
        locals: &Vec<Value>,
        sequence: &SequenceState,
        registrations: Vec<RootId>,
    ) -> Self {
        Self {
            stack: NonNull::from(stack),
            vars: NonNull::from(vars),
            locals: NonNull::from(locals),
            sequence: NonNull::from(sequence),
            registrations,
            active: true,
        }
    }
    fn finish(&mut self, outputs: &[Value]) {
        if !self.active {
            return;
        }
        self.active = false;
        // SAFETY: construction requires all frame owners to outlive this guard,
        // and the caller has already dropped the suspended instruction future.
        let mut collector = AllocationCollector::default();
        let mut result = Ok(());
        let mut visit = |value: &Value| {
            if result.is_ok() {
                result = collector.visit(value);
            }
        };
        for value in unsafe { self.stack.as_ref() }
            .iter()
            .chain(unsafe { self.vars.as_ref() })
            .chain(unsafe { self.locals.as_ref() })
        {
            visit(value);
        }
        unsafe { self.sequence.as_ref() }.visit_retained_values(&mut visit);
        let collected = result.and_then(|()| collector.finish());
        match collected {
            Ok(owned) if owned.is_empty() => {}
            Ok(owned) => {
                let mut keep = HashSet::new();
                if outputs
                    .iter()
                    .all(|value| retain_value(value, &mut keep).is_some())
                {
                    release(owned, &self.registrations, &keep);
                } else {
                    log::warn!("VM completed-output aliases are unavailable");
                }
            }
            Err(reason) => log::warn!("VM frame ownership unavailable: {reason}"),
        }
    }
    pub fn complete(&mut self, outputs: &[Value]) {
        self.finish(outputs);
    }
}
impl Drop for FrameValues {
    fn drop(&mut self) {
        self.finish(&[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::context::{
        RuntimeCallRequest, RuntimeCallService, RuntimeContext, RuntimeServiceFuture,
        RuntimeServicePorts,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn pure_host_arguments_do_not_register_roots_or_scan_global_owners() {
        let values = vec![Value::Tensor(
            runmat_value::Tensor::new(vec![2.0; 65_536], vec![256, 256]).unwrap(),
        )];
        let guard = ConsumedValues::new(&values);
        assert!(guard.allocations.is_none());
        assert!(guard.root.is_none());
        drop(guard);
    }

    struct CapturingCall(RefCell<Vec<Value>>);
    impl RuntimeCallService for CapturingCall {
        fn visit_retained_values(
            &self,
            visitor: &mut dyn FnMut(&Value) -> Option<()>,
        ) -> Option<()> {
            for value in self.0.try_borrow().ok()?.iter() {
                visitor(value)?;
            }
            Some(())
        }
        fn resolve(&self, _: &str) -> Option<usize> {
            None
        }
        fn invoke(
            &self,
            _: RuntimeCallRequest,
        ) -> RuntimeServiceFuture<Result<runmat_value::ValueSequence, runmat_runtime::RuntimeError>>
        {
            unreachable!("root projection does not invoke callbacks")
        }
    }
    struct OpaqueCall;
    impl RuntimeCallService for OpaqueCall {
        fn resolve(&self, _: &str) -> Option<usize> {
            None
        }
        fn invoke(
            &self,
            _: RuntimeCallRequest,
        ) -> RuntimeServiceFuture<Result<runmat_value::ValueSequence, runmat_runtime::RuntimeError>>
        {
            unreachable!("root projection does not invoke callbacks")
        }
    }

    #[test]
    fn injected_callback_nested_alias_is_retained_and_borrow_failure_is_not_empty() {
        runmat_gc::gc_test_context(|| {
            let handle = GpuTensorHandle::new(vec![1, 1], 9123, 42);
            let nested = Value::Cell(
                runmat_value::CellArray::new_with_shape(
                    vec![Value::GpuTensor(handle.clone())],
                    vec![1, 1],
                )
                .unwrap(),
            );
            let calls = Rc::new(CapturingCall(RefCell::new(vec![nested])));
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(RuntimeServicePorts::default().with_call(calls.clone()));
            futures::executor::block_on(runtime.scope(async {
                assert!(retained(&[], &HashSet::new())
                    .unwrap()
                    .contains(&handle_identity(&handle)));
                let _borrow = calls.0.borrow_mut();
                assert!(retained(&[], &HashSet::new())
                    .unwrap_err()
                    .contains("call service"));
            }));
        });
    }

    #[test]
    fn opaque_callback_service_prevents_partial_alias_proof() {
        runmat_gc::gc_test_context(|| {
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(RuntimeServicePorts::default().with_call(Rc::new(OpaqueCall)));
            futures::executor::block_on(runtime.scope(async {
                assert!(retained(&[], &HashSet::new())
                    .unwrap_err()
                    .contains("call service"));
            }));
        });
    }
}
