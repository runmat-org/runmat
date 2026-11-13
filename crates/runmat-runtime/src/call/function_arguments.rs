//! Rooted, segmented arguments for a selected semantic function entry.
//!
//! A closure's by-value captures stay in the immutable callable generation.
//! The invocation owns only its explicit arguments. Keeping the two segments
//! separate avoids manufacturing a second `Vec<Value>` (and a second copy of
//! every captured payload) merely to satisfy a flattened function ABI.

use runmat_value::Value;

use crate::execution::{RetainedValueReadLease, RootedValues};
use crate::runtime_error::{semantic_error, semantic_error_with_source};
use crate::RuntimeError;
use std::ops::Range;
mod view;
mod builtin;
pub use builtin::{BuiltinArguments, BuiltinArgumentsIntoIter};
#[cfg(test)]
mod range_tests;
pub use view::{
    ArgumentRangeError, FunctionArgumentChunksExact, FunctionArgumentIter, FunctionArgumentView,
};

/// The callable lease retains both its immutable capture Values and their GC
/// roots; `supplied` independently retains the explicit argument Values. The
/// owner can cross a queued child invocation without borrowing its caller.
#[derive(Debug)]
pub struct RootedFunctionArguments {
    routing: Option<RootedValues>,
    callable: Option<RetainedCallableTarget>,
    supplied: RootedValues,
    ranges: [Range<usize>; 3],
}

/// One immutable capture projected from the original callable generation.
/// Replacing a function-local slot retires this view, never mutates the
/// captured closure or manufactures a second Value owner.
#[derive(Clone, Debug)]
pub struct CapturedValueLease {
    // The direct generation pins the complete immutable closure graph. The
    // projected address may refer to a capture of a captured closure at any
    // depth, but can never outlive that original generation.
    generation: RetainedValueReadLease,
    value: std::ptr::NonNull<Value>,
}

/// One consumed request argument without flattening a closure generation.
/// Explicit/routing values move from their original collection; captured
/// values retain the callable generation and expose only an immutable view.
#[derive(Debug)]
pub enum RootedArgumentValue {
    Owned(Value),
    Captured(CapturedValueLease),
}

impl RootedArgumentValue {
    pub fn value(&self) -> &Value {
        match self {
            Self::Owned(value) => value,
            Self::Captured(value) => value.value(),
        }
    }

    pub fn into_owned(self) -> Result<Value, Self> {
        match self {
            Self::Owned(value) => Ok(value),
            captured @ Self::Captured(_) => Err(captured),
        }
    }
}

impl From<Value> for RootedArgumentValue {
    fn from(value: Value) -> Self {
        Self::Owned(value)
    }
}

pub struct RootedFunctionArgumentsIntoIter {
    routing: Option<crate::execution::RootedValuesIntoIter>,
    callable: Option<RetainedCallableTarget>,
    capture_index: usize,
    capture_count: usize,
    supplied: crate::execution::RootedValuesIntoIter,
    routing_skip: usize,
    routing_remaining: usize,
    supplied_skip: usize,
    supplied_remaining: usize,
}

impl Iterator for RootedFunctionArgumentsIntoIter {
    type Item = RootedArgumentValue;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(routing) = self.routing.as_mut() {
            while self.routing_skip != 0 {
                routing.next()?;
                self.routing_skip -= 1;
            }
            if self.routing_remaining != 0 {
                self.routing_remaining -= 1;
                let value = routing.next()?;
                return Some(RootedArgumentValue::Owned(value));
            }
            self.routing = None;
        }
        if self.capture_index < self.capture_count {
            let index = self.capture_index;
            self.capture_index += 1;
            return Some(RootedArgumentValue::Captured(
                self.callable
                    .as_ref()
                    .expect("captured count requires callable owner")
                    .capture(index)
                    .expect("captured count was read from the same callable owner"),
            ));
        }
        while self.supplied_skip != 0 {
            self.supplied.next()?;
            self.supplied_skip -= 1;
        }
        if self.supplied_remaining == 0 {
            return None;
        }
        self.supplied_remaining -= 1;
        self.supplied.next().map(RootedArgumentValue::Owned)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let routing = self.routing_remaining;
        let captures = self.capture_count.saturating_sub(self.capture_index);
        let remaining = routing + captures + self.supplied_remaining;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for RootedFunctionArgumentsIntoIter {}
impl std::iter::FusedIterator for RootedFunctionArgumentsIntoIter {}

#[derive(Debug)]
pub enum ContiguousArgumentMaterializationError {
    ManagedValueBacking,
}

impl std::fmt::Display for ContiguousArgumentMaterializationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ManagedValueBacking => formatter
                .write_str("Value payload has no admitted contiguous materialization owner"),
        }
    }
}

impl std::error::Error for ContiguousArgumentMaterializationError {}

impl CapturedValueLease {
    pub fn value(&self) -> &Value {
        // SAFETY: `generation` pins the immutable SharedValueGeneration from
        // which this address was projected. Closure captures are never
        // resized or replaced inside that generation, including for a nested
        // projection, and this lease cannot outlive the generation owner.
        unsafe { self.value.as_ref() }
    }

    pub(crate) fn same_capture(&self, other: &Self) -> bool {
        self.value == other.value && self.generation.shares_generation_with(&other.generation)
    }

    fn capture(&self, index: usize) -> Option<Self> {
        let Value::Closure(closure) = self.value() else {
            return None;
        };
        let value = std::ptr::NonNull::from(closure.captures.get(index)?);
        Some(Self {
            generation: self.generation.clone(),
            value,
        })
    }
}

/// Canonical immutable callback target ownership. A captured target remains a
/// view into its original closure generation; it is never copied into a new
/// standalone Value generation merely to satisfy a foreign adapter ABI.
#[derive(Clone, Debug)]
pub enum RetainedCallableTarget {
    Direct(RetainedValueReadLease),
    Captured(CapturedValueLease),
}

impl RetainedCallableTarget {
    pub fn value(&self) -> &Value {
        match self {
            Self::Direct(owner) => owner.value(),
            Self::Captured(owner) => owner.value(),
        }
    }

    fn capture(&self, index: usize) -> Option<CapturedValueLease> {
        match self {
            Self::Direct(generation) => {
                let Value::Closure(closure) = generation.value() else {
                    return None;
                };
                let value = std::ptr::NonNull::from(closure.captures.get(index)?);
                Some(CapturedValueLease {
                    generation: generation.clone(),
                    value,
                })
            }
            Self::Captured(owner) => owner.capture(index),
        }
    }

    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Direct(left), Self::Direct(right)) => left.shares_generation_with(right),
            (Self::Captured(left), Self::Captured(right)) => left.same_capture(right),
            _ => false,
        }
    }
}

impl From<RetainedValueReadLease> for RetainedCallableTarget {
    fn from(owner: RetainedValueReadLease) -> Self {
        Self::Direct(owner)
    }
}

impl From<CapturedValueLease> for RetainedCallableTarget {
    fn from(owner: CapturedValueLease) -> Self {
        Self::Captured(owner)
    }
}

impl RootedFunctionArguments {
    pub fn ordinary(supplied: RootedValues) -> Self {
        let count = supplied.len();
        Self {
            routing: None,
            callable: None,
            supplied,
            ranges: [0..0, 0..0, 0..count],
        }
    }

    pub fn captured(
        callable: impl Into<RetainedCallableTarget>,
        supplied: RootedValues,
    ) -> Result<Self, RuntimeError> {
        let callable = callable.into();
        if !matches!(callable.value(), Value::Closure(_)) {
            return Err(semantic_error(
                "CallableArguments",
                "capture owner is not a selected closure",
            ));
        }
        let Value::Closure(closure) = callable.value() else {
            unreachable!("validated closure")
        };
        let captures = closure.captures.len();
        let count = supplied.len();
        Ok(Self {
            routing: None,
            callable: Some(callable),
            supplied,
            ranges: [0..0, 0..captures, 0..count],
        })
    }

    pub fn fork(&self) -> Self {
        Self {
            routing: self.routing.as_ref().map(RootedValues::fork),
            callable: self.callable.clone(),
            supplied: self.supplied.fork(),
            ranges: self.ranges.clone(),
        }
    }
    pub(crate) fn retain_existing_roots(
        &self,
        children: &mut crate::execution::RootedValueChildren<'_>,
    ) -> Result<(), crate::context::host_allocation::HostAllocationError> {
        if let Some(routing) = &self.routing {
            routing.retain_existing_roots(children)?;
        }
        self.supplied.retain_existing_roots(children)?;
        if let Some(callable) = &self.callable {
            let generation = match callable {
                RetainedCallableTarget::Direct(generation) => generation,
                RetainedCallableTarget::Captured(capture) => &capture.generation,
            };
            children.try_retain_handles(generation.shared_roots())?;
        }
        Ok(())
    }
    pub fn view(&self) -> FunctionArgumentView<'_> {
        FunctionArgumentView::new(self.routing(), self.captures(), self.supplied())
    }
    pub fn into_supplied_arguments(self) -> BuiltinArguments {
        let start = self.routing().len() + self.capture_count();
        BuiltinArguments::new(
            self.into_range(start..)
                .expect("supplied suffix is in bounds"),
        )
    }

    /// Lexical captures occupy closure-local slots. A bound method's retained
    /// receiver is instead the first ordinary function input.
    pub fn lexical_capture_count(&self) -> usize {
        match self.callable.as_ref().map(RetainedCallableTarget::value) {
            Some(Value::Closure(closure))
                if closure.source_role == runmat_types::FunctionSourceRole::Method =>
            {
                0
            }
            _ => self.capture_count(),
        }
    }

    pub fn function_inputs(&self) -> FunctionArgumentView<'_> {
        self.view()
            .range(self.routing().len() + self.lexical_capture_count()..)
            .expect("function input suffix is in bounds")
    }

    /// Project the final ABI inputs without flattening or copying a bound
    /// receiver out of its immutable callable generation.
    pub fn into_function_inputs(self) -> BuiltinArguments {
        let start = self.routing().len() + self.lexical_capture_count();
        BuiltinArguments::new(
            self.into_range(start..)
                .expect("function input suffix is in bounds"),
        )
    }
    pub fn into_range(
        mut self,
        range: impl std::ops::RangeBounds<usize>,
    ) -> Result<Self, ArgumentRangeError> {
        let range = view::bounds(range, self.len())?;
        let mut cursor = 0usize;
        self.ranges = self.ranges.map(|segment| {
            let width = segment.len();
            let start = range.start.saturating_sub(cursor).min(width);
            let end = range.end.saturating_sub(cursor).min(width);
            cursor += width;
            (segment.start + start)..(segment.start + end)
        });
        Ok(self)
    }
    pub fn range(
        &self,
        range: impl std::ops::RangeBounds<usize>,
    ) -> Result<Self, ArgumentRangeError> {
        self.fork().into_range(range)
    }

    /// Attach one foreign-routing value without flattening a captured prefix.
    /// Repeated prefixes remain in their own exact collection owner.
    pub fn prepend_routing_in_context(
        self,
        prefix: Value,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, RuntimeError> {
        self.prepend_routing(prefix, Some(runtime))
    }

    pub fn prepend_routing(
        mut self,
        prefix: Value,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<Self, RuntimeError> {
        let prefix = match runtime {
            Some(runtime) => crate::execution::RootedValue::in_context(prefix, runtime)?,
            None => {
                use crate::context::host_allocation::{
                    host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
                    HostAllocationReceipt,
                };
                let error = |source| {
                    host_allocation_runtime_error(
                        None,
                        "RunMat:Call:Arguments",
                        "routing prefix policy unavailable",
                        source,
                    )
                };
                if HostAllocationReceipt::policy(None).map_err(error)?
                    != HostAllocationPolicy::Unbounded
                {
                    return Err(error(HostAllocationError::MissingManagedOwner));
                }
                crate::execution::RootedValue::new(prefix).map_err(|e| e.into_runtime_error())?
            }
        };
        let width =
            self.routing().len().checked_add(1).ok_or_else(|| {
                semantic_error("CallableArguments", "routing prefix width overflow")
            })?;
        let routing =
            RootedValues::transform_arguments(&self, width, Some(&prefix), runtime, |index| {
                Ok(if index == 0 {
                    prefix.value().clone()
                } else {
                    self.routing()[index - 1].clone()
                })
            })?;
        self.routing = Some(routing);
        self.ranges[0] = 0..width;
        Ok(self)
    }

    /// Reinterpret the incoming logical Values as a NEW callee's supplied
    /// inputs, then attach that callee's original capture generation.
    /// Ordinary supplied-only owners transfer O(1), including selected ranges.
    /// Existing capture/routing segments require exactly one admitted final
    /// supplied batch; only Value headers are projected, not nested payloads.
    pub(crate) fn rebind_callee(
        self,
        callable: Option<RetainedCallableTarget>,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<Self, RuntimeError> {
        let captures = match callable.as_ref().map(RetainedCallableTarget::value) {
            Some(Value::Closure(closure)) => closure.captures.len(),
            Some(_) => {
                return Err(semantic_error(
                    "CallableArguments",
                    "callback capture owner is not a closure",
                ))
            }
            None => 0,
        };
        if self.ranges[0].is_empty() && self.ranges[1].is_empty() {
            return Ok(Self {
                routing: None,
                callable,
                supplied: self.supplied,
                ranges: [0..0, 0..captures, self.ranges[2].clone()],
            });
        }
        let count = self.len();
        let supplied = RootedValues::transform_arguments(&self, count, None, runtime, |index| {
            Ok(self
                .get(index)
                .expect("stable callback argument view")
                .clone())
        })?;
        Ok(Self {
            routing: None,
            callable,
            supplied,
            ranges: [0..0, 0..captures, 0..count],
        })
    }

    pub fn remove_first_routing_in_context(
        mut self,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, RuntimeError> {
        // This edits logical metadata only. The original registration and
        // physical owner survive every remaining routed argument.
        crate::context::host_allocation::HostAllocationReceipt::capture_owner(Some(runtime))
            .map_err(|source| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Call:Arguments",
                    "routing authority unavailable",
                    source,
                )
            })?;
        if self.ranges[0].is_empty() {
            return Err(semantic_error(
                "ForeignDeclineIdentity",
                "qualified foreign decline lost its routing metadata",
            ));
        }
        self.ranges[0].start += 1;
        if self.ranges[0].is_empty() {
            self.routing = None;
            self.ranges[0] = 0..0;
        }
        Ok(self)
    }

    pub fn capture_count(&self) -> usize {
        self.captures().len()
    }

    pub fn supplied_count(&self) -> usize {
        self.ranges[2].len()
    }

    pub fn len(&self) -> usize {
        self.routing().len() + self.capture_count() + self.supplied_count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&Value> {
        let routing = self.routing();
        if let Some(value) = routing.get(index) {
            return Some(value);
        }
        let index = index.checked_sub(routing.len())?;
        let captures = self.captures();
        if let Some(capture) = captures.get(index) {
            return Some(capture);
        }
        self.supplied().get(index.checked_sub(captures.len())?)
    }

    pub fn first(&self) -> Option<&Value> {
        self.get(0)
    }

    pub fn captures(&self) -> &[Value] {
        match self.callable.as_ref().map(RetainedCallableTarget::value) {
            Some(Value::Closure(closure)) => &closure.captures[self.ranges[1].clone()],
            _ => &[],
        }
    }

    pub fn capture_lease(&self, index: usize) -> Option<CapturedValueLease> {
        self.captures().get(index)?;
        self.callable
            .as_ref()?
            .capture(self.ranges[1].start + index)
    }

    pub fn supplied(&self) -> &[Value] {
        &self.supplied.values()[self.ranges[2].clone()]
    }

    pub fn routing(&self) -> &[Value] {
        self.routing
            .as_ref()
            .map_or(&[], |routing| &routing.values()[self.ranges[0].clone()])
    }

    pub fn callable(&self) -> Option<&RetainedCallableTarget> {
        self.callable.as_ref()
    }

    /// Recover the original explicit argument owner when no captured prefix
    /// was attached. A segmented closure call cannot be misrepresented as an
    /// ordinary contiguous request.
    pub fn try_into_ordinary(self) -> Result<RootedValues, Self> {
        if self.callable.is_some()
            || self.routing.is_some()
            || self.ranges[2] != (0..self.supplied.len())
        {
            return Err(self);
        }
        Ok(self.supplied)
    }

    pub fn iter(&self) -> FunctionArgumentIter<'_> {
        self.view().iter()
    }

    /// Consume the segmented request without duplicating nested payloads.
    /// Unique routing/supplied slots move; aliases retain canonical owners.
    /// Captures retain the original callable generation through read leases.
    pub fn try_into_values(self) -> Result<RootedFunctionArgumentsIntoIter, RuntimeError> {
        let Self {
            routing,
            callable,
            supplied,
            ranges,
        } = self;
        Ok(RootedFunctionArgumentsIntoIter {
            routing: routing.map(RootedValues::try_into_values).transpose()?,
            callable,
            capture_index: ranges[1].start,
            capture_count: ranges[1].end,
            supplied: supplied.try_into_values()?,
            routing_skip: ranges[0].start,
            routing_remaining: ranges[0].len(),
            supplied_skip: ranges[2].start,
            supplied_remaining: ranges[2].len(),
        })
    }

    /// Explicit compatibility boundary for an older consumer that requires
    /// one contiguous `Vec<Value>`. It cannot claim finite allocation merely
    /// because the segmented source generations and roots are admitted.
    /// Selected bound/Native entry never calls this method.
    pub fn materialize_legacy_contiguous(
        &self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> Result<Vec<Value>, RuntimeError> {
        use crate::context::host_allocation::{
            host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
        };

        let policy = HostAllocationReceipt::policy(runtime).map_err(|source| {
            host_allocation_runtime_error(
                runtime,
                "RunMat:Call:Arguments",
                "function argument policy unavailable",
                source,
            )
        })?;
        if !self.is_empty() && policy == HostAllocationPolicy::Managed {
            return Err(semantic_error_with_source(
                "RunMat:Call:ContiguousArgumentsUnavailable",
                "arguments require an admitted contiguous Value owner",
                ContiguousArgumentMaterializationError::ManagedValueBacking,
            ));
        }
        Ok(self.iter().cloned().collect())
    }

    /// Transfer the explicit Value vector exactly once. Its original root
    /// lease remains live throughout the asynchronous child, while the
    /// callable generation keeps the captured prefix live independently.
    pub async fn with_owned_supplied_async<R, F>(
        self,
        consume: impl FnOnce(Option<RetainedCallableTarget>, Vec<Value>) -> F,
    ) -> Result<R, RuntimeError>
    where
        F: std::future::Future<Output = R>,
    {
        let Self {
            routing,
            callable,
            supplied,
            ranges,
        } = self;
        if routing.is_some() {
            return Err(semantic_error(
                "CallableArguments",
                "routed arguments cannot enter a semantic function body",
            ));
        }
        supplied
            .with_unbounded_values_async(|mut values| {
                values.truncate(ranges[2].end);
                values.drain(..ranges[2].start);
                consume(callable, values)
            })
            .await
    }
}

impl std::ops::Index<usize> for RootedFunctionArguments {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
            .expect("rooted function argument index is in bounds")
    }
}

impl<'a> IntoIterator for &'a RootedFunctionArguments {
    type Item = &'a Value;
    type IntoIter = FunctionArgumentIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use runmat_value::Closure;
    use std::rc::Rc;

    #[test]
    fn repeated_reads_share_one_immutable_capture_generation() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let closure = Value::Closure(Closure::new(
            "captured".into(),
            None,
            runmat_types::FunctionSourceRole::Anonymous,
            None,
            vec![Value::Num(17.0)],
            None,
        ));
        let callable = RetainedValueReadLease::prepare_owned(closure, &runtime).unwrap();
        let first = RootedFunctionArguments::captured(
            callable.clone(),
            RootedValues::new(vec![Value::Num(3.0)]).unwrap(),
        )
        .unwrap();
        let second = RootedFunctionArguments::captured(
            callable,
            RootedValues::new(vec![Value::Num(5.0)]).unwrap(),
        )
        .unwrap();
        assert!(std::ptr::eq(first.get(0).unwrap(), second.get(0).unwrap()));
        let capture = first.capture_lease(0).unwrap();
        assert!(std::ptr::eq(capture.value(), first.get(0).unwrap()));
        assert_eq!(first.len(), 2);
        assert_eq!(first.get(0), Some(&Value::Num(17.0)));
        assert_eq!(first.get(1), Some(&Value::Num(3.0)));
        assert_eq!(second.get(1), Some(&Value::Num(5.0)));
    }

    #[test]
    fn nested_capture_projection_keeps_the_original_generation() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let inner = Value::Closure(Closure::new(
            "inner".into(),
            None,
            runmat_types::FunctionSourceRole::Anonymous,
            None,
            vec![Value::Num(29.0)],
            None,
        ));
        let outer = Value::Closure(Closure::new(
            "outer".into(),
            None,
            runmat_types::FunctionSourceRole::Anonymous,
            None,
            vec![inner],
            None,
        ));
        let outer = RetainedValueReadLease::prepare_owned(outer, &runtime).unwrap();
        let outer_arguments =
            RootedFunctionArguments::captured(outer, RootedValues::new(Vec::new()).unwrap())
                .unwrap();
        let inner = outer_arguments.capture_lease(0).unwrap();
        let inner_arguments = RootedFunctionArguments::captured(
            RetainedCallableTarget::Captured(inner),
            RootedValues::new(Vec::new()).unwrap(),
        )
        .unwrap();
        let nested = inner_arguments.capture_lease(0).unwrap();
        assert_eq!(nested.value(), &Value::Num(29.0));
        let mut values = inner_arguments.try_into_values().unwrap();
        assert_eq!(values.next().unwrap().value(), &Value::Num(29.0));
        assert!(values.next().is_none());
    }
}
