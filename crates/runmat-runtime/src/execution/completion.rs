use runmat_gc::{GcHandle, PreparedRootPlan, Trace, Tracer};
use runmat_value::{LexicalEnvironment, OutputListStorage, Value, ValueSequence};

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
    values: OutputListStorage,
    _roots: RetainedValueRoots,
}

/// Consuming traversal of one rooted argument owner. The physical collection
/// (and any exact claim) drops before its root lease, including partial
/// iteration and early failure.
#[derive(Debug)]
pub struct RootedValuesIntoIter {
    values: runmat_value::OutputListIntoIter,
    _roots: RetainedValueRoots,
}

impl Iterator for RootedValuesIntoIter {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        self.values.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}

impl DoubleEndedIterator for RootedValuesIntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.values.next_back()
    }
}

impl ExactSizeIterator for RootedValuesIntoIter {}
impl std::iter::FusedIterator for RootedValuesIntoIter {}

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
        struct Handle(GcHandle);
        impl Trace for Handle {
            fn trace(&self, tracer: &mut dyn Tracer) {
                tracer.mark(self.0);
            }
        }
        let source = Handle(handle);
        Self::for_handle_source_in_context(&source, runtime)
    }

    /// Retain every activation in one lexical parent chain as a single exact
    /// root lease. This metadata-only trace marks handles directly; unlike
    /// language `Value::trace`, it cannot allocate an unbounded traversal
    /// workspace while operating under a managed admission policy.
    pub fn for_lexical_environment_in_context(
        environment: &LexicalEnvironment,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, crate::RuntimeError> {
        struct ActivationChain<'a>(&'a LexicalEnvironment);
        impl Trace for ActivationChain<'_> {
            fn trace(&self, tracer: &mut dyn Tracer) {
                self.0.for_each_handle(|handle| tracer.mark(handle));
            }
        }
        Self::for_handle_source_in_context(&ActivationChain(environment), runtime).map_err(
            |source| match source {
                RootedValueAdmissionError::Host(source) => {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        Some(runtime),
                        "RunMat:Lexical:RootLease",
                        "lexical environment root admission failed",
                        source,
                    )
                }
                RootedValueAdmissionError::Gc(source) => {
                    crate::runtime_error::semantic_error_with_source(
                        "RunMat:Lexical:RootLease",
                        "lexical environment root admission failed",
                        source,
                    )
                }
            },
        )
    }

    fn for_handle_source_in_context(
        source: &impl Trace,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, RootedValueAdmissionError> {
        use crate::context::host_allocation::{
            HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
        };

        let plan = PreparedRootPlan::for_traced(source).map_err(|error| {
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
            let roots = plan.prepare(source, &permit).map_err(|error| {
                RootedValueAdmissionError::Gc(runmat_gc::GcError::RootRegistrationFailed(
                    error.to_string(),
                ))
            })?;
            receipt.checkpoint(runtime)?;
            Ok(Self {
                _roots: RootedValueHandles::from_prepared(roots),
            })
        } else {
            let roots = plan.prepare_unbounded(source).map_err(|error| {
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

    /// Root several already-owned Value slices as one exact operation before
    /// transferring them to queued work. The root plan replays the same
    /// borrowed slices through one admitted trace workspace; no Value forest
    /// or reference vector is copied. Keep the returned lease with the queued
    /// owner until its inputs are independently rooted or retired.
    pub fn for_value_slices_in_context(
        slices: &[&[Value]],
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, crate::RuntimeError> {
        let _roots =
            RootedValueHandles::for_value_slices_in_context(slices, runtime).map_err(|source| {
                match source {
                    RootedValueAdmissionError::Host(source) => {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            Some(runtime),
                            "RunMat:Execution:RootLease",
                            "queued value root admission failed",
                            source,
                        )
                    }
                    RootedValueAdmissionError::Gc(source) => {
                        crate::runtime_error::semantic_error_with_source(
                            "RunMat:Execution:RootLease",
                            "queued value root admission failed",
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
            values: OutputListStorage::unbounded(self.values),
            _roots: self.member_roots.finish(),
        })
    }
}

impl RootedValues {
    /// Prepend one already-owned metadata value while preserving every input
    /// Value and the original exact collection claim until the replacement
    /// collection is fully admitted. No payload is cloned.
    pub fn prepend_in_context(
        self,
        prefix: Value,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, crate::RuntimeError> {
        use crate::context::host_allocation::{
            host_allocation_runtime_error, HostAllocationPlan, HostAllocationReceipt,
        };

        let width = self.len().checked_add(1).ok_or_else(|| {
            crate::runtime_error::semantic_error(
                "RunMat:Call:Arguments",
                "prefixed argument width overflows",
            )
        })?;
        let raw = OutputListStorage::requested_bytes(width).map_err(|source| {
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Call:Arguments",
                "prefixed argument layout overflow",
                source.into(),
            )
        })?;
        let raw = u64::try_from(raw).map_err(|_| {
            crate::runtime_error::semantic_error(
                "RunMat:Call:Arguments",
                "prefixed argument layout overflows",
            )
        })?;
        let retained =
            HostAllocationPlan::retained_exact_output_bytes(raw, usize::from(width != 0)).map_err(
                |source| {
                    host_allocation_runtime_error(
                        Some(runtime),
                        "RunMat:Call:Arguments",
                        "prefixed argument admission failed",
                        source,
                    )
                },
            )?;
        let receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(0, retained).map_err(|source| {
                host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Call:Arguments",
                    "prefixed argument admission failed",
                    source,
                )
            })?,
        )
        .map_err(|source| {
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Call:Arguments",
                "prefixed argument admission failed",
                source,
            )
        })?;

        if let Some(receipt) = receipt {
            let permit = receipt.output_permit(runtime).map_err(|source| {
                host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Call:Arguments",
                    "prefixed argument admission failed",
                    source,
                )
            })?;
            let mut prefix = Some(prefix);
            let mut source = self.try_into_values()?;
            let storage = OutputListStorage::try_initialize(width, &permit, |index| {
                if index == 0 {
                    Ok::<_, crate::RuntimeError>(prefix.take().expect("single prefix"))
                } else {
                    source.next().ok_or_else(|| {
                        crate::runtime_error::semantic_error(
                            "RunMat:Call:Arguments",
                            "prefixed argument owner ended early",
                        )
                    })
                }
            })
            .map_err(|error| match error {
                runmat_value::OutputListInitializationError::Allocation(source) => {
                    host_allocation_runtime_error(
                        Some(runtime),
                        "RunMat:Call:Arguments",
                        "prefixed argument allocation failed",
                        source.into(),
                    )
                }
                runmat_value::OutputListInitializationError::Element(source) => source,
            })?;
            receipt.checkpoint(runtime).map_err(|source| {
                host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Call:Arguments",
                    "prefixed argument checkpoint failed",
                    source,
                )
            })?;
            Self::from_output_storage_in_context(storage, runtime)
        } else {
            self.with_unbounded_values(|mut values| {
                values.insert(0, prefix);
                Self::in_context(values, runtime)
            })?
        }
    }

    /// Remove routing metadata from the front while moving the remaining
    /// Values into a newly admitted owner. Used only when a selected foreign
    /// probe proves a pre-marshalling miss and returns the original payload.
    pub fn remove_first_in_context(
        self,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, crate::RuntimeError> {
        use crate::context::host_allocation::{
            host_allocation_runtime_error, HostAllocationPlan, HostAllocationReceipt,
        };
        if self.is_empty() {
            return Err(crate::runtime_error::semantic_error(
                "ForeignDeclineIdentity",
                "qualified foreign decline lost its routing metadata",
            ));
        }
        let width = self.len() - 1;
        let raw = OutputListStorage::requested_bytes(width).map_err(|source| {
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Call:Arguments",
                "returned argument layout overflow",
                source.into(),
            )
        })?;
        let raw = u64::try_from(raw).map_err(|_| {
            crate::runtime_error::semantic_error(
                "RunMat:Call:Arguments",
                "returned argument layout overflows",
            )
        })?;
        let retained =
            HostAllocationPlan::retained_exact_output_bytes(raw, usize::from(width != 0)).map_err(
                |source| {
                    host_allocation_runtime_error(
                        Some(runtime),
                        "RunMat:Call:Arguments",
                        "returned argument admission failed",
                        source,
                    )
                },
            )?;
        let receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(0, retained).map_err(|source| {
                host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Call:Arguments",
                    "returned argument admission failed",
                    source,
                )
            })?,
        )
        .map_err(|source| {
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Call:Arguments",
                "returned argument admission failed",
                source,
            )
        })?;
        let mut source = self.try_into_values()?;
        let _routing_metadata = source.next().expect("nonempty rooted arguments");
        if let Some(receipt) = receipt {
            let permit = receipt.output_permit(runtime).map_err(|source| {
                host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Call:Arguments",
                    "returned argument admission failed",
                    source,
                )
            })?;
            let storage = OutputListStorage::try_initialize(width, &permit, |_| {
                source.next().ok_or_else(|| {
                    crate::runtime_error::semantic_error(
                        "RunMat:Call:Arguments",
                        "returned argument owner ended early",
                    )
                })
            })
            .map_err(|error| match error {
                runmat_value::OutputListInitializationError::Allocation(source) => {
                    host_allocation_runtime_error(
                        Some(runtime),
                        "RunMat:Call:Arguments",
                        "returned argument allocation failed",
                        source.into(),
                    )
                }
                runmat_value::OutputListInitializationError::Element(source) => source,
            })?;
            receipt.checkpoint(runtime).map_err(|source| {
                host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Call:Arguments",
                    "returned argument checkpoint failed",
                    source,
                )
            })?;
            Self::from_output_storage_in_context(storage, runtime)
        } else {
            Self::in_context(source.collect(), runtime)
        }
    }

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
            values: OutputListStorage::unbounded(values),
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
            values: OutputListStorage::unbounded(values),
            _roots: RetainedValueRoots::direct(roots),
        })
    }

    /// Retain the canonical output backing and its physical claim while
    /// registering the Values' independent managed roots.
    pub fn from_output_storage(values: OutputListStorage) -> Result<Self, ExecutionServiceError> {
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

    pub fn from_output_storage_in_context(
        values: OutputListStorage,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, crate::RuntimeError> {
        let roots = RootedValueHandles::for_values_slice_in_context(values.as_slice(), runtime)
            .map_err(|source| match source {
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
            })?;
        Ok(Self {
            values,
            _roots: RetainedValueRoots::direct(roots),
        })
    }

    pub fn values(&self) -> &[Value] {
        self.values.as_slice()
    }

    /// Move every Value out of a unique collection while preserving both its
    /// exact physical claim and GC roots through the last yielded member.
    /// Shared aliases are rejected instead of cloning nested payloads.
    pub fn try_into_values(self) -> Result<RootedValuesIntoIter, crate::RuntimeError> {
        let Self { values, _roots } = self;
        let values = values.try_into_owned().map_err(|_| {
            crate::sequence::sequence_error_to_runtime(
                runmat_value::ValueSequenceError::SharedOutputStorage,
            )
        })?;
        Ok(RootedValuesIntoIter {
            values: values.into_iter(),
            _roots,
        })
    }

    /// Transfer the original collection owner while the prior roots remain
    /// live through synchronous publication. Claimed backing is never turned
    /// into a bare Vec or reallocated here.
    pub fn with_owned_output_storage<R>(self, publish: impl FnOnce(OutputListStorage) -> R) -> R {
        let Self { values, _roots } = self;
        let result = publish(values);
        drop(_roots);
        result
    }

    /// Async form of the exact-owner handoff. The root lease remains in the
    /// suspended future through completion of the consumer.
    pub async fn with_owned_output_storage_async<R, F>(
        self,
        consume: impl FnOnce(OutputListStorage) -> F,
    ) -> R
    where
        F: std::future::Future<Output = R>,
    {
        let Self { values, _roots } = self;
        let result = consume(values).await;
        drop(_roots);
        result
    }

    /// Compatibility handoff for APIs that still require a bare Vec. Exact
    /// backing is refused rather than losing its physical claim. This is not
    /// an admission mechanism for a new contiguous collection.
    pub fn with_unbounded_values<R>(
        self,
        publish: impl FnOnce(Vec<Value>) -> R,
    ) -> Result<R, crate::RuntimeError> {
        let Self { values, _roots } = self;
        let values = values
            .try_into_owned()
            .map_err(|_| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?
            .try_into_unbounded_vec()
            .map_err(|_| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?;
        let result = publish(values);
        drop(_roots);
        Ok(result)
    }

    /// Async compatibility handoff with the same explicit finite refusal.
    pub async fn with_unbounded_values_async<R, F>(
        self,
        consume: impl FnOnce(Vec<Value>) -> F,
    ) -> Result<R, crate::RuntimeError>
    where
        F: std::future::Future<Output = R>,
    {
        let Self { values, _roots } = self;
        let values = values
            .try_into_owned()
            .map_err(|_| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?
            .try_into_unbounded_vec()
            .map_err(|_| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?;
        let result = consume(values).await;
        drop(_roots);
        Ok(result)
    }
}

impl std::ops::Deref for RootedValues {
    type Target = [Value];

    fn deref(&self) -> &Self::Target {
        self.values.as_slice()
    }
}

impl PartialEq<Vec<Value>> for RootedValues {
    fn eq(&self, other: &Vec<Value>) -> bool {
        self.values.as_slice() == other.as_slice()
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
    use runmat_value::{
        ExactHostAllocationPermit, ExactHostAllocationReservation, ExactHostNumericAllocationError,
        HandleRef, Value,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Debug)]
    struct OutputPermit(Arc<AtomicUsize>);
    #[derive(Debug)]
    struct OutputClaim(Arc<AtomicUsize>, usize);
    impl ExactHostAllocationReservation for OutputClaim {}
    impl Drop for OutputClaim {
        fn drop(&mut self) {
            self.0.fetch_sub(self.1, Ordering::SeqCst);
        }
    }
    impl ExactHostAllocationPermit for OutputPermit {
        fn reserve_exact(
            &self,
            bytes: usize,
        ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError>
        {
            self.0.fetch_add(bytes, Ordering::SeqCst);
            Ok(Arc::new(OutputClaim(Arc::clone(&self.0), bytes)))
        }
    }

    #[test]
    fn rooted_output_handoff_retains_exact_backing_without_vec_materialization() {
        let live = Arc::new(AtomicUsize::new(0));
        let storage = OutputListStorage::try_initialize(2, &OutputPermit(Arc::clone(&live)), |i| {
            Ok::<_, ()>(Value::Num(i as f64 + 1.0))
        })
        .expect("admitted output storage");
        let pointer = storage.as_slice().as_ptr();
        let rooted = RootedValues::from_output_storage(storage).expect("root output values");
        assert!(live.load(Ordering::SeqCst) > 0);
        let sequence = rooted.with_owned_output_storage(|storage| {
            assert_eq!(storage.as_slice().as_ptr(), pointer);
            RootedValueSequence::new(
                ValueSequence::from_output_storage(storage).expect("valid output sequence"),
            )
            .expect("root successor")
        });
        assert_eq!(sequence.as_slice().as_ptr(), pointer);
        assert!(live.load(Ordering::SeqCst) > 0);
        drop(sequence);
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }

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
