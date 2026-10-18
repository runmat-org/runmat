use runmat_gc::{GcHandle, PreparedRootPlan, Trace, Tracer};
use runmat_value::{Value, ValueSequence};

use super::ExecutionServiceError;

#[path = "completion/roots.rs"]
mod roots;
use roots::RetainedValueRoots;
pub(crate) use roots::{
    RootedValueAdmissionError, RootedValueChildren, RootedValueHandles, RootedValueMember,
};

/// A completed executor-owned value whose reachable managed handles remain GC
/// roots until the completion record is discarded or transferred to a sink.
#[derive(Clone, Debug)]
pub struct RootedResult<T: Trace> {
    value: T,
    _roots: RootedValueHandles,
}

pub type RootedValueSequence = RootedResult<ValueSequence>;
pub type RootedValue = RootedResult<Value>;

/// Owned values whose reachable managed handles remain roots while an async
/// operation retains the batch. This is the canonical input counterpart to
/// [`RootedValueSequence`].
#[derive(Clone, Debug)]
pub struct RootedValues {
    values: Vec<Value>,
    _roots: RetainedValueRoots,
}

/// A root-only lease for values already owned by a host result carrier. It
/// traces managed identities without cloning tensor or device payloads.
#[derive(Clone, Debug)]
pub struct RootedValueLease {
    _roots: RootedValueHandles,
}

impl RootedValueLease {
    /// Root one private GC activation without tracing a language Value or
    /// allocating traversal scratch. Used by a prepared lexical transaction
    /// while its source environment may otherwise retire before commit.
    pub(crate) fn for_handle_in_context(
        handle: GcHandle,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, RootedValueAdmissionError> {
        use crate::context::host_allocation::{
            HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
        };

        struct Handle(GcHandle);
        impl Trace for Handle {
            fn trace(&self, tracer: &mut dyn Tracer) {
                tracer.mark(self.0);
            }
        }

        let source = Handle(handle);
        let plan = PreparedRootPlan::for_traced(&source).map_err(|error| {
            RootedValueAdmissionError::Gc(runmat_gc::GcError::RootRegistrationFailed(
                error.to_string(),
            ))
        })?;
        let quote = plan.quote();
        let raw = u64::try_from(quote.raw_retained_bytes)
            .map_err(|_| RootedValueAdmissionError::Host(HostAllocationError::Overflow))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(raw, quote.retained_claims)?;
        if let Some(receipt) =
            HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, retained)?)?
        {
            let permit = receipt.output_permit(runtime)?;
            let roots = plan.prepare(&source, &permit).map_err(|error| {
                RootedValueAdmissionError::Gc(runmat_gc::GcError::RootRegistrationFailed(
                    error.to_string(),
                ))
            })?;
            receipt.checkpoint(runtime)?;
            Ok(Self {
                _roots: RootedValueHandles::from_prepared(roots),
            })
        } else {
            let roots = plan.prepare_unbounded(&source).map_err(|error| {
                RootedValueAdmissionError::Gc(runmat_gc::GcError::RootRegistrationFailed(
                    error.to_string(),
                ))
            })?;
            Ok(Self {
                _roots: RootedValueHandles::from_prepared(roots),
            })
        }
    }
    pub fn for_traced(value: &impl Trace) -> Result<Self, ExecutionServiceError> {
        let roots = RootedValueHandles::for_traced(value).map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "could not root pending host result: {error}"
            ))
        })?;
        Ok(Self { _roots: roots })
    }

    pub fn for_values<'a>(
        values: impl Iterator<Item = &'a Value>,
    ) -> Result<Self, ExecutionServiceError> {
        let roots = RootedValueHandles::for_values(values).map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "could not root pending host result: {error}"
            ))
        })?;
        Ok(Self { _roots: roots })
    }

    /// Retain an owned request's supplied arguments before its future can be
    /// queued unpolled. The original Vec remains with its descriptor; this
    /// admits only the independent GC root lease, not a second Value copy.
    pub(crate) fn for_values_in_context(
        values: &[Value],
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, crate::RuntimeError> {
        let _roots =
            RootedValueHandles::for_values_slice_in_context(values, runtime).map_err(|source| {
                match source {
                    RootedValueAdmissionError::Host(source) => {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            Some(runtime),
                            "RunMat:Call:Arguments",
                            "call argument root admission failed",
                            source,
                        )
                    }
                    RootedValueAdmissionError::Gc(source) => {
                        crate::runtime_error::semantic_error_with_source(
                            "RunMat:Call:Arguments",
                            "call argument root admission failed",
                            source,
                        )
                    }
                }
            })?;
        Ok(Self { _roots })
    }

    /// Root one already-protected Value under the current operation's exact
    /// allocation policy. The returned lease can be shared into a retained
    /// publication owner without registering its handles a second time.
    pub(crate) fn for_value_in_context(
        value: &Value,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, RootedValueAdmissionError> {
        RootedValueHandles::for_value_in_context(value, runtime).map(|_roots| Self { _roots })
    }

    pub(crate) fn empty() -> Self {
        Self {
            _roots: RootedValueHandles::empty(),
        }
    }

    pub(crate) fn shared_handles(&self) -> RootedValueHandles {
        self._roots.clone()
    }

    pub fn for_sequence(value: &ValueSequence) -> Result<Self, ExecutionServiceError> {
        Self::for_traced(value)
    }
}

/// Incremental owner for values decoded by a potentially allocating codec.
/// Every value is rooted before decoding or allocating the next member.
#[derive(Debug, Default)]
pub struct RootedValuesBuilder {
    values: Vec<Value>,
    member_roots: RootedValueChildren,
}

impl RootedValuesBuilder {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            member_roots: RootedValueChildren::new(),
        }
    }

    pub fn push(&mut self, value: Value) -> Result<(), ExecutionServiceError> {
        let roots = RootedValueHandles::for_traced(&value).map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "could not root retained future input: {error}"
            ))
        })?;
        self.values.push(value);
        self.member_roots.push(RetainedValueRoots::direct(roots));
        Ok(())
    }

    pub(crate) fn push_rooted(&mut self, member: RootedValueMember) {
        let (value, roots) = member.into_parts();
        self.values.push(value);
        self.member_roots.push(roots);
    }

    /// Transfer a completed value's existing roots into this batch without
    /// retiring them or registering a duplicate set of roots.
    pub fn push_owned(&mut self, value: RootedValue) {
        let RootedResult { value, _roots } = value;
        self.values.push(value);
        self.member_roots.push(RetainedValueRoots::direct(_roots));
    }

    pub fn finish(self) -> Result<RootedValues, ExecutionServiceError> {
        Ok(RootedValues {
            values: self.values,
            _roots: self.member_roots.finish(),
        })
    }
}

impl RootedValues {
    /// Take ownership of an already-allocated argument vector and admit its
    /// external GC roots under the current operation without copying Values.
    /// This does not certify arbitrary preexisting payload/vector backing.
    pub fn in_context(
        values: Vec<Value>,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, crate::RuntimeError> {
        let roots = RootedValueHandles::for_values_slice_in_context(&values, runtime).map_err(
            |source| match source {
                RootedValueAdmissionError::Host(source) => {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        Some(runtime),
                        "RunMat:Call:Arguments",
                        "function argument root admission failed",
                        source,
                    )
                }
                RootedValueAdmissionError::Gc(source) => {
                    crate::runtime_error::semantic_error_with_source(
                        "RunMat:Call:Arguments",
                        "function argument root admission failed",
                        source,
                    )
                }
            },
        )?;
        Ok(Self {
            values,
            _roots: RetainedValueRoots::direct(roots),
        })
    }

    pub fn new(values: Vec<Value>) -> Result<Self, ExecutionServiceError> {
        let roots = RootedValueHandles::for_values(values.iter()).map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "could not root retained future inputs: {error}"
            ))
        })?;
        Ok(Self {
            values,
            _roots: RetainedValueRoots::direct(roots),
        })
    }

    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Publish retained values into a registered sink before their temporary
    /// root lease can notify retirement observers.
    pub fn with_owned_values<R>(self, publish: impl FnOnce(Vec<Value>) -> R) -> R {
        let Self { values, _roots } = self;
        let result = publish(values);
        drop(_roots);
        result
    }

    /// Keep the input lease through an effectful consumer. The returned value
    /// must itself be rooted before this future completes if it can contain
    /// managed outputs; this method transfers only the argument ownership.
    pub async fn with_owned_values_async<R, F>(self, consume: impl FnOnce(Vec<Value>) -> F) -> R
    where
        F: std::future::Future<Output = R>,
    {
        let Self { values, _roots } = self;
        let result = consume(values).await;
        drop(_roots);
        result
    }
}

impl std::ops::Deref for RootedValues {
    type Target = [Value];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl PartialEq<Vec<Value>> for RootedValues {
    fn eq(&self, other: &Vec<Value>) -> bool {
        &self.values == other
    }
}

impl<T: Trace> RootedResult<T> {
    pub fn new(value: T) -> Result<Self, ExecutionServiceError> {
        let roots = RootedValueHandles::for_traced(&value).map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "could not root completed future output: {error}"
            ))
        })?;
        Ok(Self {
            value,
            _roots: roots,
        })
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    /// Transfer an owned value and its existing root registration separately
    /// into an iterative effectful transformation. The raw value must remain
    /// alongside this lease until a successor has been rooted or published.
    pub(crate) fn into_value_and_lease(self) -> (T, RootedValueLease) {
        let Self { value, _roots } = self;
        (value, RootedValueLease { _roots })
    }

    /// Map an owned result while retaining its old managed roots. The new
    /// result is rooted before old roots can retire or notify observers.
    pub fn try_map_owned<U: Trace, E>(
        self,
        map: impl FnOnce(T) -> Result<U, E>,
        root_error: impl FnOnce(ExecutionServiceError) -> E,
    ) -> Result<RootedResult<U>, E> {
        let Self { value, _roots } = self;
        let mapped = map(value)?;
        let rooted = RootedResult::new(mapped).map_err(root_error)?;
        drop(_roots);
        Ok(rooted)
    }

    /// Consume into a synchronous sink that publishes its own roots or fully
    /// encodes the value before this closure returns. Returning a raw managed
    /// value from the closure is not a valid ownership transfer.
    pub fn with_owned<R>(self, consume: impl FnOnce(T) -> R) -> R {
        let Self { value, _roots } = self;
        let result = consume(value);
        drop(_roots);
        result
    }

    /// Keep this result rooted while an effectful consumer awaits. The
    /// consumer must complete a terminal codec or publish a new rooted owner
    /// before its future resolves; returning raw managed output is invalid.
    pub async fn with_owned_async<R, F>(self, consume: impl FnOnce(T) -> F) -> R
    where
        F: std::future::Future<Output = R>,
    {
        let Self { value, _roots } = self;
        let result = consume(value).await;
        drop(_roots);
        result
    }
}

impl RootedResult<ValueSequence> {
    /// A zero-output result owns no managed values or root registrations.
    /// It is independent of ambient runtime context and needs no allocation.
    pub fn empty() -> Self {
        Self {
            value: ValueSequence::empty(),
            _roots: RootedValueHandles::empty(),
        }
    }

    pub fn with_owned_sequence<R>(self, consume: impl FnOnce(ValueSequence) -> R) -> R {
        self.with_owned(consume)
    }
}

impl<T: Trace> std::ops::Deref for RootedResult<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: Trace + PartialEq> PartialEq for RootedResult<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Trace + PartialEq> PartialEq<T> for RootedResult<T> {
    fn eq(&self, other: &T) -> bool {
        &self.value == other
    }
}

impl PartialEq<RootedValueSequence> for ValueSequence {
    fn eq(&self, other: &RootedValueSequence) -> bool {
        self == &other.value
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use runmat_value::{HandleRef, Value};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn unmanaged_values_need_no_shared_root_owner() {
        let values = [Value::Num(7.0), Value::Bool(false)];
        let roots = RootedValueHandles::for_values(values.iter()).expect("no managed handles");
        assert!(roots.is_empty());
        assert!(roots.clone().is_empty());
    }

    #[test]
    fn completed_sequence_roots_nested_handle_values_until_discarded() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("retained".into()))
                .expect("allocate rooted payload");
            let handle = initial.handle();
            let sequence = ValueSequence::comma_separated(vec![Value::HandleObject(HandleRef {
                class_name: "CompletionHandle".into(),
                target: handle,
                valid: true,
            })])
            .expect("valid completion sequence");
            let rooted = RootedValueSequence::new(sequence).expect("root completion");
            initial.unroot().expect("transfer root to completion");

            runmat_gc::gc_collect_major().expect("collect with live completion");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("completion remains rooted"),
                Value::String("retained".into())
            );

            drop(rooted);
            runmat_gc::gc_collect_major().expect("collect after completion discard");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn mapped_completion_registers_new_owner_before_old_owner_retires() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("mapped".into()))
                .expect("allocate rooted payload");
            let handle = initial.handle();
            let value = Value::HandleObject(HandleRef {
                class_name: "MappedCompletionHandle".into(),
                target: handle,
                valid: true,
            });
            let source = RootedValue::new(value).expect("root source completion");
            initial
                .unroot()
                .expect("transfer initial root to source completion");

            let destination = source
                .try_map_owned(
                    |value| {
                        runmat_gc::gc_collect_major().expect("collect during owner mapping");
                        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                        ValueSequence::single(value)
                            .map_err(crate::sequence::sequence_error_to_runtime)
                    },
                    |error| error.into_runtime_error(),
                )
                .expect("map while source owner is live");
            runmat_gc::gc_collect_major().expect("collect with destination owner");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("destination remains rooted"),
                Value::String("mapped".into())
            );

            drop(destination);
            runmat_gc::gc_collect_major().expect("collect after final owner retires");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn separated_value_and_lease_keep_managed_target_live_through_successor_rooting() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("leased".into()))
                .expect("allocate rooted payload");
            let handle = initial.handle();
            let source = RootedValue::new(Value::HandleObject(HandleRef {
                class_name: "LeasedCompletionHandle".into(),
                target: handle,
                valid: true,
            }))
            .expect("root source completion");
            initial.unroot().expect("transfer setup root");

            let (value, lease) = source.into_value_and_lease();
            runmat_gc::gc_collect_major().expect("collect with separated lease");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("lease retains target"),
                Value::String("leased".into())
            );
            let destination = RootedValue::new(value).expect("root successor");
            drop(lease);
            runmat_gc::gc_collect_major().expect("collect after source lease retires");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("successor retains target"),
                Value::String("leased".into())
            );
            drop(destination);
            runmat_gc::gc_collect_major().expect("collect after successor retires");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn mapped_completion_survives_collecting_root_retirement_observer() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("handoff".into()))
                .expect("allocate rooted payload");
            let handle = initial.handle();
            let source = RootedValue::new(Value::HandleObject(HandleRef {
                class_name: "MappedCompletionHandle".into(),
                target: handle,
                valid: true,
            }))
            .expect("root source completion");
            initial.unroot().expect("transfer root to source");

            let observed = Arc::new(AtomicBool::new(false));
            let observed_in_callback = Arc::clone(&observed);
            let identity = (handle.addr(), handle.epoch());
            let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
                let runmat_gc::RootRetirement::Explicit(handle) = event else {
                    return;
                };
                if (handle.addr(), handle.epoch()) != identity
                    || observed_in_callback.swap(true, Ordering::SeqCst)
                {
                    return;
                }
                runmat_gc::gc_collect_major().expect("collect during old-root retirement");
                assert_eq!(
                    runmat_gc::gc_clone_value(&handle).expect("destination root is live"),
                    Value::String("handoff".into())
                );
            }))
            .expect("install retirement observer");

            let destination = source
                .try_map_owned(
                    |value| {
                        ValueSequence::single(value)
                            .map_err(crate::sequence::sequence_error_to_runtime)
                    },
                    |error| error.into_runtime_error(),
                )
                .expect("map with overlapping roots");
            assert!(observed.load(Ordering::SeqCst));
            assert_eq!(destination.as_slice().len(), 1);
        });
    }

    #[test]
    fn retained_input_values_root_nested_handles_until_discarded() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("input".into()))
                .expect("allocate rooted payload");
            let handle = initial.handle();
            let values = RootedValues::new(vec![Value::HandleObject(HandleRef {
                class_name: "InputHandle".into(),
                target: handle,
                valid: true,
            })])
            .expect("root retained inputs");
            initial.unroot().expect("transfer root to retained inputs");

            runmat_gc::gc_collect_major().expect("collect with live inputs");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("input remains rooted"),
                Value::String("input".into())
            );

            drop(values);
            runmat_gc::gc_collect_major().expect("collect after retained input discard");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn incremental_inputs_survive_collection_between_decoded_members() {
        runmat_gc::gc_test_context(|| {
            let first = runmat_gc::gc_allocate_rooted(Value::String("first".into()))
                .expect("allocate first payload");
            let first_handle = first.handle();
            let mut builder = RootedValuesBuilder::with_capacity(2);
            builder
                .push(Value::HandleObject(HandleRef {
                    class_name: "DecodedHandle".into(),
                    target: first_handle,
                    valid: true,
                }))
                .expect("root first decoded member");
            first.unroot().expect("transfer first root");
            runmat_gc::gc_collect_major().expect("collect between decoded members");
            assert_eq!(
                runmat_gc::gc_clone_value(&first_handle).expect("first member remains rooted"),
                Value::String("first".into())
            );

            let second = runmat_gc::gc_allocate_rooted(Value::String("second".into()))
                .expect("allocate second payload");
            let second_handle = second.handle();
            builder
                .push(Value::HandleObject(HandleRef {
                    class_name: "DecodedHandle".into(),
                    target: second_handle,
                    valid: true,
                }))
                .expect("root second decoded member");
            second.unroot().expect("transfer second root");
            let values = builder.finish().expect("finish rooted batch");

            runmat_gc::gc_collect_major().expect("collect completed batch");
            assert!(runmat_gc::gc_clone_value(&first_handle).is_ok());
            assert!(runmat_gc::gc_clone_value(&second_handle).is_ok());
            drop(values);
            runmat_gc::gc_collect_major().expect("collect after batch drop");
            assert!(runmat_gc::gc_clone_value(&first_handle).is_err());
            assert!(runmat_gc::gc_clone_value(&second_handle).is_err());
        });
    }

    #[test]
    fn completed_member_roots_transfer_without_retrace_and_clone_until_last_owner() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("member".into()))
                .expect("allocate payload");
            let handle = initial.handle();
            let mut builder = RootedValuesBuilder::with_capacity(1);
            builder
                .push(Value::HandleObject(HandleRef {
                    class_name: "DecodedHandle".into(),
                    target: handle,
                    valid: true,
                }))
                .expect("root decoded member exactly once");
            initial.unroot().expect("transfer initial root");
            let values = builder.finish().expect("transfer member roots");
            let clone = values.clone();
            drop(values);
            runmat_gc::gc_collect_major().expect("collect with cloned completion");
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            drop(clone);
            runmat_gc::gc_collect_major().expect("collect after final completion owner");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn completed_value_transfers_into_batch_without_root_retirement() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("owned member".into()))
                .expect("allocate managed payload");
            let handle = initial.handle();
            let value = RootedValue::new(Value::HandleObject(HandleRef {
                class_name: "CompletedHandle".into(),
                target: handle,
                valid: true,
            }))
            .expect("root completed value");
            initial.unroot().expect("transfer allocation root");
            let identity = (handle.addr(), handle.epoch());
            let retirements = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let count = std::sync::Arc::clone(&retirements);
            let _observer =
                runmat_gc::RootRetirementObserver::register(std::sync::Arc::new(move |event| {
                    if let runmat_gc::RootRetirement::Explicit(handle) = event {
                        if (handle.addr(), handle.epoch()) == identity {
                            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        }
                    }
                }))
                .expect("observe exact root retirement");
            let mut builder = RootedValuesBuilder::with_capacity(1);
            builder.push_owned(value);
            let values = builder.finish().expect("finish owned batch");
            assert_eq!(retirements.load(std::sync::atomic::Ordering::SeqCst), 0);
            runmat_gc::gc_collect_major().expect("collect with transferred owner");
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            drop(values);
            assert_eq!(retirements.load(std::sync::atomic::Ordering::SeqCst), 1);
            runmat_gc::gc_collect_major().expect("collect after batch retirement");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn chunked_member_roots_clone_and_retire_without_flattening() {
        runmat_gc::gc_test_context(|| {
            const MEMBER_COUNT: usize = 129;
            let mut initial = Vec::new();
            let mut builder = RootedValuesBuilder::with_capacity(MEMBER_COUNT);
            for index in 0..MEMBER_COUNT {
                let rooted = runmat_gc::gc_allocate_rooted(Value::Num(index as f64))
                    .expect("allocate managed payload");
                let handle = rooted.handle();
                builder
                    .push(Value::HandleObject(HandleRef {
                        class_name: "ChunkedCompletionHandle".into(),
                        target: handle,
                        valid: true,
                    }))
                    .expect("root member");
                initial.push((rooted, handle));
            }
            for (rooted, _) in initial.drain(..) {
                rooted.unroot().expect("transfer initial root");
            }
            let values = builder.finish().expect("finish chunked roots");
            let retained = values.clone();
            drop(values);
            runmat_gc::gc_collect_major().expect("collect with shared chunk chain");
            for value in retained.values() {
                let Value::HandleObject(handle) = value else {
                    panic!("retained member should be a handle");
                };
                assert!(runmat_gc::gc_clone_value(&handle.target).is_ok());
            }
            let handles = retained
                .values()
                .iter()
                .map(|value| match value {
                    Value::HandleObject(handle) => handle.target,
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            drop(retained);
            runmat_gc::gc_collect_major().expect("collect after final chunk owner");
            assert!(handles
                .iter()
                .all(|handle| runmat_gc::gc_clone_value(handle).is_err()));
        });
    }
}
