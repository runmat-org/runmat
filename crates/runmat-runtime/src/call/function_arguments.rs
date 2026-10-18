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
    callable: Option<RetainedValueReadLease>,
    supplied: RootedValues,
}

/// One immutable capture projected from the original callable generation.
/// Replacing a function-local slot retires this view, never mutates the
/// captured closure or manufactures a second Value owner.
#[derive(Clone, Debug)]
pub struct CapturedValueLease {
    callable: RetainedValueReadLease,
    index: usize,
}

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
        let Value::Closure(closure) = self.callable.value() else {
            unreachable!("capture lease was validated at construction")
        };
        &closure.captures[self.index]
    }
}

impl RootedFunctionArguments {
    pub fn ordinary(supplied: RootedValues) -> Self {
        Self {
            callable: None,
            supplied,
        }
    }

    pub fn captured(
        callable: RetainedValueReadLease,
        supplied: RootedValues,
    ) -> Result<Self, RuntimeError> {
        if !matches!(callable.value(), Value::Closure(_)) {
            return Err(semantic_error(
                "CallableArguments",
                "capture owner is not a selected closure",
            ));
        }
        Ok(Self {
            callable: Some(callable),
            supplied,
        })
    }

    pub fn capture_count(&self) -> usize {
        self.captures().len()
    }

    pub fn supplied_count(&self) -> usize {
        self.supplied.len()
    }

    pub fn len(&self) -> usize {
        self.capture_count() + self.supplied_count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&Value> {
        let captures = self.captures();
        if let Some(capture) = captures.get(index) {
            return Some(capture);
        }
        self.supplied.get(index.checked_sub(captures.len())?)
    }

    pub fn captures(&self) -> &[Value] {
        match self.callable.as_ref().map(RetainedValueReadLease::value) {
            Some(Value::Closure(closure)) => &closure.captures,
            _ => &[],
        }
    }

    pub fn capture_lease(&self, index: usize) -> Option<CapturedValueLease> {
        self.captures().get(index)?;
        Some(CapturedValueLease {
            callable: self.callable.as_ref()?.clone(),
            index,
        })
    }

    pub fn supplied(&self) -> &[Value] {
        self.supplied.values()
    }

    pub fn callable(&self) -> Option<&RetainedValueReadLease> {
        self.callable.as_ref()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.captures().iter().chain(self.supplied.values().iter())
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
        consume: impl FnOnce(Option<RetainedValueReadLease>, Vec<Value>) -> F,
    ) -> R
    where
        F: std::future::Future<Output = R>,
    {
        let Self { callable, supplied } = self;
        supplied
            .with_owned_values_async(|values| consume(callable, values))
            .await
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
}
