//! Incremental ownership of GC roots retained by decoded values.

use runmat_execution::resource::ExactShared;

#[path = "roots/children.rs"]
mod children;
use children::RootChunk;
pub(crate) use children::RootedValueChildren;

use runmat_gc::{GcError, PreparedRootLease, PreparedRootPlan, RootTraceFailure, Trace, Tracer};
use runmat_value::{ExactHostNumericAllocationError, Value, ValueTraceWorkspace};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTraceScratchPermit,
};
use crate::context::RuntimeContext;

#[derive(Debug)]
pub(crate) enum RootedValueAdmissionError {
    Host(HostAllocationError),
    Gc(GcError),
}

impl From<HostAllocationError> for RootedValueAdmissionError {
    fn from(error: HostAllocationError) -> Self {
        Self::Host(error)
    }
}

/// Managed handles reachable from owned Values while an asynchronous owner
/// retains them. This roots identities, not cloned payloads or device buffers.
#[derive(Clone, Debug)]
pub(crate) struct RootedValueHandles {
    _roots: PreparedRootLease,
}

/// One decoded value paired with the roots retained while constructing it.
///
/// `root` traces a standalone value. `aggregate` is a trusted crate-private
/// construction boundary: its caller must transfer every managed child edge
/// from the same aggregate into `children` and must not add an untracked edge.
#[derive(Debug)]
pub(crate) struct RootedValueMember {
    value: Value,
    roots: RetainedValueRoots,
}

impl RootedValueMember {
    pub(crate) fn value(&self) -> &Value {
        &self.value
    }

    /// Transfer an existing registration rather than retracing callback output.
    pub(crate) fn from_rooted(value: super::RootedValue) -> Self {
        let super::RootedResult { value, _roots } = value;
        Self {
            value,
            roots: RetainedValueRoots::direct(_roots),
        }
    }
    /// Root under the selected invocation without using the compatibility
    /// planner's unbounded traversal/reference storage.
    pub(crate) fn root_in_context(
        value: Value,
        runtime: &RuntimeContext,
    ) -> Result<Self, RootedValueAdmissionError> {
        let roots =
            RetainedValueRoots::direct(RootedValueHandles::for_value_in_context(&value, runtime)?);
        Ok(Self { value, roots })
    }

    pub(crate) fn root(value: Value) -> Result<Self, GcError> {
        let roots = RetainedValueRoots::direct(RootedValueHandles::for_traced(&value)?);
        Ok(Self { value, roots })
    }

    pub(crate) fn aggregate(
        value: Value,
        mut children: RootedValueChildren<'_>,
    ) -> Result<Self, HostAllocationError> {
        match children.try_finish() {
            Ok(roots) => Ok(Self { value, roots }),
            Err(error) => {
                // Root retirement can trigger collection. Retire the physical
                // aggregate before its only construction roots on failure.
                drop(value);
                drop(children);
                Err(error)
            }
        }
    }

    pub(crate) fn into_unrooted_value(self) -> Value {
        self.value
    }

    /// Promote a decoded member without retiring its construction roots first.
    pub(crate) fn into_rooted_value(
        self,
    ) -> Result<super::RootedValue, super::super::ExecutionServiceError> {
        let Self { value, roots } = self;
        let rooted = super::RootedValue::new(value)?;
        drop(roots);
        Ok(rooted)
    }

    pub(crate) fn into_child(
        self,
        children: &mut RootedValueChildren<'_>,
    ) -> Result<Value, HostAllocationError> {
        let Self { value, roots } = self;
        match children.try_push(roots) {
            Ok(()) => Ok(value),
            Err((roots, error)) => {
                drop(value);
                drop(roots);
                Err(error)
            }
        }
    }

    pub(super) fn into_parts(self) -> (Value, RetainedValueRoots) {
        (self.value, self.roots)
    }
}

#[derive(Clone, Debug)]
pub(super) enum RetainedValueRoots {
    Empty,
    Direct { _roots: RootedValueHandles },
    Aggregate { _root: ExactShared<RootChunk> },
}

impl RootedValueChildren<'_> {
    /// Retain an existing registration, never retrace its Value generation.
    pub(crate) fn try_retain_handles(
        &mut self,
        handles: RootedValueHandles,
    ) -> Result<(), HostAllocationError> {
        self.try_push(RetainedValueRoots::direct(handles))
            .map_err(|(_roots, error)| error)
    }

    pub(crate) fn try_publish_values(
        mut self,
        values: runmat_value::OutputListStorage,
    ) -> Result<super::RootedValues, HostAllocationError> {
        match self.try_finish() {
            Ok(_roots) => Ok(super::RootedValues { values, _roots }),
            Err(error) => {
                drop(values);
                Err(error)
            }
        }
    }
}

impl super::RootedValues {
    pub(crate) fn retain_existing_roots(
        &self,
        children: &mut RootedValueChildren<'_>,
    ) -> Result<(), HostAllocationError> {
        children
            .try_push(self._roots.clone())
            .map_err(|(_roots, error)| error)
    }
}

impl RetainedValueRoots {
    pub(super) fn direct(roots: RootedValueHandles) -> Self {
        if roots.is_empty() {
            Self::Empty
        } else {
            Self::Direct { _roots: roots }
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

impl RootedValueHandles {
    pub(crate) fn into_prepared_lease(self) -> PreparedRootLease {
        self._roots
    }
    pub(crate) fn from_prepared(_roots: PreparedRootLease) -> Self {
        Self { _roots }
    }
    pub(crate) fn empty() -> Self {
        Self::for_traced(&Value::Num(0.0)).expect("a scalar has no managed roots")
    }

    /// Root one Value through a dynamically claimed traversal workspace under
    /// the exact current operation. The two GC traversals reuse the same
    /// scratch block; retained registration is independently precharged.
    pub(crate) fn for_value_in_context(
        value: &Value,
        runtime: &RuntimeContext,
    ) -> Result<Self, RootedValueAdmissionError> {
        Self::for_values_slice_in_context(std::slice::from_ref(value), runtime)
    }

    /// Replay one stable owned slice through the canonical iterative Value
    /// tracer. No reference collection or second argument Value vector is
    /// allocated; every actual scratch growth uses the operation permit.
    pub(crate) fn for_values_slice_in_context(
        values: &[Value],
        runtime: &RuntimeContext,
    ) -> Result<Self, RootedValueAdmissionError> {
        Self::for_value_slices_in_context(&[values], runtime)
    }

    /// One admitted traversal and root lease over disjoint borrowed batches.
    /// The source slices remain owned by their caller; no Value references or
    /// payloads are collected into a second buffer.
    pub(crate) fn for_value_slices_in_context(
        slices: &[&[Value]],
        runtime: &RuntimeContext,
    ) -> Result<Self, RootedValueAdmissionError> {
        Self::for_values_iterator_in_context(
            slices.iter().flat_map(|values| values.iter()),
            runtime,
        )
    }

    /// Replay original borrowed owners without constructing a reference or
    /// payload collection. The iterator must enumerate the same stable Values
    /// during planning and registration.
    pub(crate) fn for_values_iterator_in_context<'a>(
        values: impl Iterator<Item = &'a Value> + Clone,
        runtime: &RuntimeContext,
    ) -> Result<Self, RootedValueAdmissionError> {
        Self::for_values_and_handles_iterator_in_context(values, std::iter::empty(), runtime)
    }

    /// Native model candidates retain their real direct GC edges before a
    /// claimed native/object header exists. The same canonical planner and
    /// trace workspace protect borrowed Values and handles as one root lease.
    pub(crate) fn for_values_and_handles_iterator_in_context<'a>(
        values: impl Iterator<Item = &'a Value> + Clone,
        handles: impl Iterator<Item = runmat_gc::GcHandle> + Clone,
        runtime: &RuntimeContext,
    ) -> Result<Self, RootedValueAdmissionError> {
        let Some(scratch) = HostAllocationReceipt::dynamic_trace_scratch(runtime)? else {
            return Self::for_traced(&ValuesAndHandlesIterator(values, handles))
                .map_err(RootedValueAdmissionError::Gc);
        };
        let mut workspace = ValueTraceWorkspace::admitted(&scratch);
        let mut trace = |tracer: &mut dyn Tracer| {
            for handle in handles.clone() {
                tracer.mark(handle);
            }
            for value in values.clone() {
                workspace.try_trace(value, tracer)?;
            }
            Ok(())
        };
        let plan = PreparedRootPlan::try_for_traced(&mut trace)
            .map_err(|error| trace_admission_error(error, &scratch))?;
        scratch.checkpoint(runtime)?;
        let quote = plan.quote();
        let raw = u64::try_from(quote.raw_retained_bytes)
            .map_err(|_| RootedValueAdmissionError::Host(HostAllocationError::Overflow))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(raw, quote.retained_claims)?;
        if retained == 0 {
            // The authenticated plan has no root lease or registry growth.
            // Re-trace under the same admitted scratch issuer without minting
            // a zero-byte receipt or relaxing the finite allocation policy.
            let roots = plan
                .try_prepare(&mut trace, &scratch)
                .map_err(|error| trace_admission_error(error, &scratch))?;
            scratch.checkpoint(runtime)?;
            return Ok(Self { _roots: roots });
        }
        let receipt =
            HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, retained)?)?
                .ok_or(RootedValueAdmissionError::Host(
                    HostAllocationError::StaleAuthority,
                ))?;
        let permit = receipt.output_permit(runtime)?;
        let roots = plan.try_prepare(&mut trace, &permit).map_err(|error| {
            receipt
                .checkpoint(runtime)
                .err()
                .map(RootedValueAdmissionError::Host)
                .unwrap_or_else(|| trace_admission_error(error, &scratch))
        })?;
        receipt.checkpoint(runtime)?;
        scratch.checkpoint(runtime)?;
        Ok(Self { _roots: roots })
    }

    pub(crate) fn for_values<'a>(values: impl Iterator<Item = &'a Value>) -> Result<Self, GcError> {
        // The compatibility API accepts a one-pass iterator. Retain only its
        // borrowed references so the canonical GC planner can trace the same
        // protected graph during planning and preparation. This remains an
        // explicitly unbounded path; a finite caller needs a separately
        // admitted trace workspace as well as the retained lease quote.
        let values: Vec<&Value> = values.collect();
        Self::for_traced(&BorrowedValues(&values))
    }

    pub(crate) fn for_values_iterator_unbounded<'a>(
        values: impl Iterator<Item = &'a Value> + Clone,
    ) -> Result<Self, GcError> {
        Self::for_traced(&ValuesIterator(values))
    }

    pub(crate) fn for_values_and_handles_iterator_unbounded<'a>(
        values: impl Iterator<Item = &'a Value> + Clone,
        handles: impl Iterator<Item = runmat_gc::GcHandle> + Clone,
    ) -> Result<Self, GcError> {
        Self::for_traced(&ValuesAndHandlesIterator(values, handles))
    }

    pub(crate) fn for_traced(value: &impl Trace) -> Result<Self, GcError> {
        let plan = PreparedRootPlan::for_traced(value).map_err(root_admission_error)?;
        let roots = plan
            .prepare_unbounded(value)
            .map_err(root_admission_error)?;
        Ok(Self { _roots: roots })
    }

    pub(super) fn is_empty(&self) -> bool {
        self._roots.handles().is_empty()
    }
}

struct ValuesIterator<I>(I);

struct ValuesAndHandlesIterator<I, H>(I, H);
impl<'a, I, H> Trace for ValuesAndHandlesIterator<I, H>
where
    I: Iterator<Item = &'a Value> + Clone,
    H: Iterator<Item = runmat_gc::GcHandle> + Clone,
{
    fn trace(&self, tracer: &mut dyn Tracer) {
        for handle in self.1.clone() {
            tracer.mark(handle);
        }
        ValuesIterator(self.0.clone()).trace(tracer);
    }
}

impl<'a, I: Iterator<Item = &'a Value> + Clone> Trace for ValuesIterator<I> {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in self.0.clone() {
            value.trace(tracer);
        }
    }
}

/// Replays a one-pass caller's borrowed Values without cloning any Value or
/// reimplementing its canonical `Trace` traversal.
struct BorrowedValues<'slice, 'value>(&'slice [&'value Value]);

impl Trace for BorrowedValues<'_, '_> {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in self.0 {
            value.trace(tracer);
        }
    }
}

fn root_admission_error(error: runmat_gc::RootAdmissionError) -> GcError {
    GcError::RootRegistrationFailed(error.to_string())
}

fn trace_admission_error(
    error: RootTraceFailure<ExactHostNumericAllocationError>,
    scratch: &HostTraceScratchPermit,
) -> RootedValueAdmissionError {
    match error {
        RootTraceFailure::Source(error) => {
            RootedValueAdmissionError::Host(scratch.source_failure(error))
        }
        RootTraceFailure::Admission(error) => {
            RootedValueAdmissionError::Gc(root_admission_error(error))
        }
    }
}
