//! Builtin-domain ownership of a logical range in the canonical rooted owner.
use super::{
    ArgumentRangeError, FunctionArgumentIter, FunctionArgumentView, RootedFunctionArguments,
};
use runmat_value::Value;
use std::ops::RangeBounds;

#[derive(Debug)]
pub struct BuiltinArguments {
    source: RootedFunctionArguments,
}
impl BuiltinArguments {
    /// The single borrowed embedding ingress. Contextual calls initialize
    /// their exact final slots before publication; language calls use owners.
    pub fn from_borrowed(
        values: &[Value],
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<Self, crate::RuntimeError> {
        crate::execution::RootedValues::from_borrowed_arguments(values, runtime)
            .map(Into::into)
            .map_err(|mut error| {
                // This is the actual root-admission failure, before the
                // builtin body exists. Gathering cannot repair or replay it.
                // Preserve typed cancellation origin and original cause.
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error
            })
    }
    pub fn new(source: RootedFunctionArguments) -> Self {
        Self { source }
    }
    pub fn fork(&self) -> Self {
        Self::new(self.source.fork())
    }
    pub fn view(&self) -> FunctionArgumentView<'_> {
        self.source.view()
    }
    pub fn len(&self) -> usize {
        self.source.len()
    }
    pub fn is_empty(&self) -> bool {
        self.source.is_empty()
    }
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.source.get(index)
    }
    pub fn first(&self) -> Option<&Value> {
        self.get(0)
    }
    pub fn last(&self) -> Option<&Value> {
        self.len().checked_sub(1).and_then(|index| self.get(index))
    }
    pub fn iter(&self) -> FunctionArgumentIter<'_> {
        self.source.iter()
    }
    pub fn range(&self, range: impl RangeBounds<usize>) -> Result<Self, ArgumentRangeError> {
        Ok(Self::new(self.source.range(range)?))
    }
    pub fn into_range(self, range: impl RangeBounds<usize>) -> Result<Self, ArgumentRangeError> {
        Ok(Self::new(self.source.into_range(range)?))
    }
    /// Transfer the same owner and logical segment ranges to semantic/foreign
    /// dispatch. No Values or registration forests are materialized here.
    pub fn into_function_arguments(self) -> RootedFunctionArguments {
        self.source
    }
    /// Admit one final logical argument collection for an explicit insertion.
    /// Existing payload owners and root registrations are retained, not traced
    /// again; no input Vec is manufactured to perform the transformation.
    pub fn try_insert_in_context(
        self,
        index: usize,
        value: crate::execution::RootedValue,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, crate::RuntimeError> {
        self.try_insert(index, value, Some(runtime))
    }
    pub(crate) fn try_insert(
        self,
        index: usize,
        value: crate::execution::RootedValue,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<Self, crate::RuntimeError> {
        if let Some(runtime) = runtime {
            return crate::execution::RootedValues::insert_argument_in_context(
                &self.source,
                index,
                &value,
                runtime,
            )
            .map(Into::into);
        }
        if index > self.len() {
            return Err(crate::runtime_error::semantic_error(
                "CallableArguments",
                "argument insertion index is out of bounds",
            ));
        }
        let count = self.len().checked_add(1).ok_or_else(|| {
            crate::runtime_error::semantic_error("CallableArguments", "argument count overflows")
        })?;
        crate::execution::RootedValues::transform_arguments(
            &self.source,
            count,
            Some(&value),
            runtime,
            |slot| {
                Ok(if slot == index {
                    value.value().clone()
                } else {
                    self.get(if slot < index { slot } else { slot - 1 })
                        .expect("validated argument slot")
                        .clone()
                })
            },
        )
        .map(Into::into)
    }
    /// Project final slots under the explicitly selected policy, retaining
    /// original roots. The mapper must preserve GC identities.
    pub(crate) fn try_map_preserving_roots(
        self,
        count: usize,
        runtime: Option<&crate::context::RuntimeContext>,
        mut project: impl FnMut(usize, &Self) -> Result<Value, crate::RuntimeError>,
    ) -> Result<Self, crate::RuntimeError> {
        crate::execution::RootedValues::transform_arguments(
            &self.source,
            count,
            None,
            runtime,
            |index| project(index, &self),
        )
        .map(Into::into)
    }

    /// Trusted root-preserving projection: new GC edges are not permitted.
    /// New root-free metadata must be admitted independently by the mapper.
    pub(crate) fn try_map_preserving_roots_in_context(
        self,
        count: usize,
        runtime: &crate::context::RuntimeContext,
        mut project: impl FnMut(usize, &Self) -> Result<Value, crate::RuntimeError>,
    ) -> Result<Self, crate::RuntimeError> {
        crate::execution::RootedValues::transform_arguments(
            &self.source,
            count,
            None,
            Some(runtime),
            |index| project(index, &self),
        )
        .map(Into::into)
    }
    pub(crate) fn rebind_callee(
        self,
        callable: Option<super::RetainedCallableTarget>,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<Self, crate::RuntimeError> {
        self.source.rebind_callee(callable, runtime).map(Self::new)
    }
    /// Use ALL logical Values as a new procedure's explicit inputs. This is
    /// not the selected-closure capture-prefix extraction performed by RFA's
    /// similarly named backend method. Supplied-only input transfers O(1).
    pub fn into_supplied_arguments(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<Self, crate::RuntimeError> {
        self.rebind_callee(None, runtime)
    }
    pub fn into_supplied_function_arguments(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<RootedFunctionArguments, crate::RuntimeError> {
        self.source.rebind_callee(None, runtime)
    }
    pub fn chunks_exact(&self, width: usize) -> super::FunctionArgumentChunksExact<'_> {
        self.view().chunks_exact(width)
    }
    /// Explicit ordinary-only inventory bridge, never canonical dispatch.
    pub(crate) fn materialize_legacy_contiguous(
        &self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<Vec<Value>, crate::RuntimeError> {
        use crate::context::host_allocation::{
            host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
        };
        let policy = HostAllocationReceipt::policy(runtime).map_err(|source| {
            host_allocation_runtime_error(
                runtime,
                "RunMat:Call:Arguments",
                "legacy policy unavailable",
                source,
            )
        })?;
        if policy != HostAllocationPolicy::Unbounded {
            return Err(crate::runtime_error::semantic_error_with_source(
                "RunMat:Call:LegacyInventoryRequiresOrdinaryPolicy",
                "selected historical inventory ABI requires ordinary allocation policy",
                LegacyInventoryPolicy,
            ));
        }
        self.source.materialize_legacy_contiguous(runtime)
    }
    /// Async projection using the same exact final-slot/root builder. The
    /// source owner remains borrowed through publication. No new untracked GC
    /// identities may be introduced by the projection.
    pub(crate) async fn try_map_preserving_roots_async(
        &self,
        count: usize,
        runtime: Option<&crate::context::RuntimeContext>,
        project: impl std::ops::AsyncFnMut(usize) -> Result<Value, crate::RuntimeError>,
    ) -> Result<Self, crate::RuntimeError> {
        crate::execution::RootedValues::transform_arguments_async(
            &self.source,
            count,
            runtime,
            project,
        )
        .await
        .map(Into::into)
    }
    /// Endpoint extraction edits only range metadata. The original owner keeps
    /// every projected Value and its existing roots alive through this carrier.
    pub fn pop_front(&mut self) -> Option<Value> {
        let value = self.first()?.clone();
        for range in &mut self.source.ranges {
            if range.start < range.end {
                range.start += 1;
                break;
            }
        }
        Some(value)
    }
    /// Mixed source/domain materialization. Rooted projections transfer any
    /// new GC registration before the next materializer can run; Source
    /// projections keep original registrations without tracing them again.
    pub(crate) async fn try_map_mixed_async(
        &self,
        count: usize,
        runtime: Option<&crate::context::RuntimeContext>,
        project: impl std::ops::AsyncFnMut(
            usize,
        ) -> Result<
            crate::execution::ArgumentProjection,
            crate::RuntimeError,
        >,
    ) -> Result<Self, crate::RuntimeError> {
        crate::execution::RootedValues::transform_mixed_arguments_async(
            &self.source,
            count,
            runtime,
            project,
        )
        .await
        .map(Into::into)
    }
    pub fn pop(&mut self) -> Option<Value> {
        let value = self.last()?.clone();
        for range in self.source.ranges.iter_mut().rev() {
            if range.start < range.end {
                range.end -= 1;
                break;
            }
        }
        Some(value)
    }
}

#[derive(Debug)]
struct LegacyInventoryPolicy;
impl std::fmt::Display for LegacyInventoryPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("historical inventory ABI has no finite allocation contract")
    }
}
impl std::error::Error for LegacyInventoryPolicy {}
impl Clone for BuiltinArguments {
    fn clone(&self) -> Self {
        self.fork()
    }
}
impl From<RootedFunctionArguments> for BuiltinArguments {
    fn from(source: RootedFunctionArguments) -> Self {
        Self::new(source)
    }
}
impl From<crate::execution::RootedValues> for BuiltinArguments {
    fn from(source: crate::execution::RootedValues) -> Self {
        Self::new(RootedFunctionArguments::ordinary(source))
    }
}
impl std::ops::Index<usize> for BuiltinArguments {
    type Output = Value;
    fn index(&self, index: usize) -> &Value {
        self.get(index)
            .expect("builtin argument index is out of bounds")
    }
}
impl<'a> IntoIterator for &'a BuiltinArguments {
    type Item = &'a Value;
    type IntoIter = FunctionArgumentIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
/// Per-element projections retain their original generation; the iterator owns
/// the complete source forest through partial iteration and early drop.
#[derive(Debug)]
pub struct BuiltinArgumentsIntoIter {
    source: BuiltinArguments,
    front: usize,
    back: usize,
}
impl Iterator for BuiltinArgumentsIntoIter {
    type Item = Value;
    fn next(&mut self) -> Option<Value> {
        if self.front == self.back {
            return None;
        }
        let i = self.front;
        self.front += 1;
        self.source.get(i).cloned()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.back - self.front;
        (n, Some(n))
    }
}
impl DoubleEndedIterator for BuiltinArgumentsIntoIter {
    fn next_back(&mut self) -> Option<Value> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        self.source.get(self.back).cloned()
    }
}
impl ExactSizeIterator for BuiltinArgumentsIntoIter {}
impl std::iter::FusedIterator for BuiltinArgumentsIntoIter {}
impl IntoIterator for BuiltinArguments {
    type Item = Value;
    type IntoIter = BuiltinArgumentsIntoIter;
    fn into_iter(self) -> Self::IntoIter {
        let back = self.len();
        BuiltinArgumentsIntoIter {
            source: self,
            front: 0,
            back,
        }
    }
}
