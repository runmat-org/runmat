//! Explicit final-slot argument transforms retaining existing registrations.
use super::{RootedValue, RootedValues};
use crate::call::RootedFunctionArguments;
use crate::context::{host_allocation::HostAllocationError, RuntimeContext};
use crate::RuntimeError;
use runmat_value::Value;
#[path = "argument_transform/storage.rs"]
mod storage;
use storage::{allocation, ArgumentTransform};

/// Construction ownership provenance, not a new Value representation.
#[derive(Debug)]
pub(crate) enum ArgumentProjection {
    /// An unchanged managed-edge projection protected by original sources.
    Source(Value),
    /// A domain-produced owner carrying any newly introduced GC identities.
    Rooted(RootedValue),
}

impl RootedValues {
    pub(crate) fn from_borrowed_arguments(
        values: &[Value],
        runtime: Option<&RuntimeContext>,
    ) -> Result<Self, RuntimeError> {
        use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
        let roots = match runtime {
            Some(runtime) => {
                super::RootedValueLease::for_values_iterator_in_context(values.iter(), runtime)?
            }
            None => {
                if HostAllocationReceipt::policy(None).map_err(|e| allocation(None, e))?
                    != HostAllocationPolicy::Unbounded
                {
                    return Err(allocation(None, HostAllocationError::MissingManagedOwner));
                }
                super::RootedValueLease::for_values(values.iter())
                    .map_err(|error| error.into_runtime_error())?
            }
        };
        // The raw embedding batch is rooted before final-slot admission can
        // trigger any external allocator callback. Its SAME registration is
        // retained by the destination, without tracing the cloned slots again.
        let mut destination = ArgumentTransform::from_source_lease(values.len(), &roots, runtime)?;
        for value in values {
            destination.push(value.clone())?;
        }
        let result = destination.finish();
        drop(roots);
        result
    }
    pub(crate) fn insert_argument_in_context(
        source: &RootedFunctionArguments,
        index: usize,
        inserted: &RootedValue,
        runtime: &RuntimeContext,
    ) -> Result<Self, RuntimeError> {
        if index > source.len() {
            return Err(crate::runtime_error::semantic_error(
                "CallableArguments",
                "argument insertion index is out of bounds",
            ));
        }
        let count = source
            .len()
            .checked_add(1)
            .ok_or_else(|| allocation(Some(runtime), HostAllocationError::Overflow))?;
        let mut destination = ArgumentTransform::new(source, count, Some(inserted), Some(runtime))?;
        for slot in 0..count {
            destination.push(if slot == index {
                inserted.value().clone()
            } else {
                source
                    .get(if slot < index { slot } else { slot - 1 })
                    .expect("quoted source argument slot")
                    .clone()
            })?;
        }
        destination.finish()
    }

    /// Trusted root-preserving transform: new GC identities are not permitted;
    /// new root-free metadata must be admitted independently by the mapper.
    pub(crate) fn transform_arguments(
        source: &RootedFunctionArguments,
        count: usize,
        inserted: Option<&RootedValue>,
        runtime: Option<&RuntimeContext>,
        mut project: impl FnMut(usize) -> Result<Value, RuntimeError>,
    ) -> Result<Self, RuntimeError> {
        let mut destination = ArgumentTransform::new(source, count, inserted, runtime)?;
        for slot in 0..count {
            destination.checkpoint()?;
            destination.push(project(slot)?)?;
        }
        destination.finish()
    }

    pub(crate) async fn transform_arguments_async(
        source: &RootedFunctionArguments,
        count: usize,
        runtime: Option<&RuntimeContext>,
        mut project: impl std::ops::AsyncFnMut(usize) -> Result<Value, RuntimeError>,
    ) -> Result<Self, RuntimeError> {
        Self::transform_mixed_arguments_async(source, count, runtime, async move |slot| {
            project(slot).await.map(ArgumentProjection::Source)
        })
        .await
    }
    pub(crate) async fn transform_mixed_arguments_async(
        source: &RootedFunctionArguments,
        count: usize,
        runtime: Option<&RuntimeContext>,
        mut project: impl std::ops::AsyncFnMut(usize) -> Result<ArgumentProjection, RuntimeError>,
    ) -> Result<Self, RuntimeError> {
        let mut destination = ArgumentTransform::new(source, count, None, runtime)?;
        for slot in 0..count {
            destination.checkpoint()?;
            let result = project(slot).await;
            destination.checkpoint()?;
            match result? {
                ArgumentProjection::Source(value) => destination.push(value)?,
                ArgumentProjection::Rooted(value) => destination.push_owned(value)?,
            }
        }
        destination.finish()
    }
}

#[cfg(test)]
#[path = "argument_transform/tests.rs"]
mod tests;
