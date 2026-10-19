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

/// The callable lease retains both its immutable capture Values and their GC
/// roots; `supplied` independently retains the explicit argument Values. The
/// owner can cross a queued child invocation without borrowing its caller.
#[derive(Debug)]
pub struct RootedFunctionArguments {
    routing: Option<RootedValues>,
    callable: Option<RetainedCallableTarget>,
    supplied: RootedValues,
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
}

impl Iterator for RootedFunctionArgumentsIntoIter {
    type Item = RootedArgumentValue;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(routing) = self.routing.as_mut() {
            if let Some(value) = routing.next() {
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
        self.supplied.next().map(RootedArgumentValue::Owned)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let routing = self.routing.as_ref().map_or(0, ExactSizeIterator::len);
        let captures = self.capture_count.saturating_sub(self.capture_index);
        let remaining = routing + captures + self.supplied.len();
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
        Self {
            routing: None,
            callable: None,
            supplied,
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
        Ok(Self {
            routing: None,
            callable: Some(callable),
            supplied,
        })
    }

    /// Attach one foreign-routing value without flattening a captured prefix.
    /// Repeated prefixes remain in their own exact collection owner.
    pub fn prepend_routing_in_context(
        mut self,
        prefix: Value,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, RuntimeError> {
        self.routing = Some(match self.routing.take() {
            Some(routing) => routing.prepend_in_context(prefix, runtime)?,
            None => RootedValues::in_context(vec![prefix], runtime)?,
        });
        Ok(self)
    }

    pub fn remove_first_routing_in_context(
        mut self,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<Self, RuntimeError> {
        let Some(routing) = self.routing.take() else {
            return Err(semantic_error(
                "ForeignDeclineIdentity",
                "qualified foreign decline lost its routing metadata",
            ));
        };
        self.routing = if routing.len() == 1 {
            drop(routing);
            None
        } else {
            Some(routing.remove_first_in_context(runtime)?)
        };
        Ok(self)
    }

    pub fn capture_count(&self) -> usize {
        self.captures().len()
    }

    pub fn supplied_count(&self) -> usize {
        self.supplied.len()
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
        self.supplied.get(index.checked_sub(captures.len())?)
    }

    pub fn first(&self) -> Option<&Value> {
        self.get(0)
    }

    pub fn captures(&self) -> &[Value] {
        match self.callable.as_ref().map(RetainedCallableTarget::value) {
            Some(Value::Closure(closure)) => &closure.captures,
            _ => &[],
        }
    }

    pub fn capture_lease(&self, index: usize) -> Option<CapturedValueLease> {
        self.captures().get(index)?;
        self.callable.as_ref()?.capture(index)
    }

    pub fn supplied(&self) -> &[Value] {
        self.supplied.values()
    }

    pub fn routing(&self) -> &[Value] {
        self.routing
            .as_ref()
            .map_or(&[], |routing| routing.values())
    }

    pub fn callable(&self) -> Option<&RetainedCallableTarget> {
        self.callable.as_ref()
    }

    /// Recover the original explicit argument owner when no captured prefix
    /// was attached. A segmented closure call cannot be misrepresented as an
    /// ordinary contiguous request.
    pub fn try_into_ordinary(self) -> Result<RootedValues, Self> {
        if self.callable.is_some() || self.routing.is_some() {
            return Err(self);
        }
        Ok(self.supplied)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.routing()
            .iter()
            .chain(self.captures().iter())
            .chain(self.supplied.values().iter())
    }

    /// Consume the segmented request without cloning any nested Value. Values
    /// owned by routing/supplied collections move; closure captures retain the
    /// original callable generation through per-element read leases.
    pub fn try_into_values(self) -> Result<RootedFunctionArgumentsIntoIter, RuntimeError> {
        let Self {
            routing,
            callable,
            supplied,
        } = self;
        let capture_count = match callable.as_ref().map(RetainedCallableTarget::value) {
            Some(Value::Closure(closure)) => closure.captures.len(),
            _ => 0,
        };
        Ok(RootedFunctionArgumentsIntoIter {
            routing: routing.map(RootedValues::try_into_values).transpose()?,
            callable,
            capture_index: 0,
            capture_count,
            supplied: supplied.try_into_values()?,
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
        } = self;
        if routing.is_some() {
            return Err(semantic_error(
                "CallableArguments",
                "routed arguments cannot enter a semantic function body",
            ));
        }
        supplied
            .with_unbounded_values_async(|values| consume(callable, values))
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
    type IntoIter = Box<dyn Iterator<Item = &'a Value> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
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
