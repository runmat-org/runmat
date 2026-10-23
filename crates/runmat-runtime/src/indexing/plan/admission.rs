//! Allocation adapters for the existing selector/geometry planner.
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, initialize_host_array, HostAllocationCheckpoint,
        HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
        HostTypedElement, HostTypedWorkspace,
    },
    RuntimeContext,
};
use crate::{execution::ExecutionServiceError, RuntimeError};
use runmat_value::{ExactArrayStorage, SharedShape};

/// Retain the invocation's original allocation authority across callbacks.
/// Existing storage keeps its own physical claims; this is only its fence.
pub(crate) struct Owner<'a> {
    context: Option<&'a RuntimeContext>,
    checkpoint: Option<HostAllocationCheckpoint>,
}
impl<'a> Owner<'a> {
    pub(crate) fn new(context: Option<&'a RuntimeContext>) -> Result<Self, RuntimeError> {
        let checkpoint = HostAllocationReceipt::capture_owner(context)
            .map_err(|source| allocation(context, source))?;
        let owner = Self {
            context,
            checkpoint,
        };
        owner.check()?;
        Ok(owner)
    }
    pub(crate) fn check(&self) -> Result<(), RuntimeError> {
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint
                .check(self.context.expect("managed planner has an explicit owner"))
                .map_err(|source| allocation(self.context, source))?;
        } else if HostAllocationReceipt::policy(self.context)
            .map_err(|source| allocation(self.context, source))?
            != HostAllocationPolicy::Unbounded
        {
            return Err(allocation(
                self.context,
                HostAllocationError::StaleAuthority,
            ));
        }
        checkpoint(self.context)
    }
    pub(crate) async fn await_stage<F: std::future::Future>(
        &self,
        future: F,
    ) -> Result<F::Output, RuntimeError> {
        self.check()?;
        let result = match self.context {
            Some(runtime) => runtime
                .cpu_cooperation()
                .await_cancellable(future)
                .await
                .map_err(|source| source.into_runtime_error_in_context(Some(runtime)))?,
            None => future.await,
        };
        self.check()?;
        Ok(result)
    }
}

fn allocation(context: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        context,
        "RunMat:Indexing:PlanAllocation",
        "index selector/plan storage",
        source,
    )
}

pub(in crate::indexing) fn checkpoint(
    context: Option<&RuntimeContext>,
) -> Result<(), RuntimeError> {
    if context.is_some_and(RuntimeContext::is_cancelled) {
        return Err(ExecutionServiceError::Cancelled.into_runtime_error_in_context(context));
    }
    Ok(())
}

pub(crate) fn array<T>(
    context: Option<&RuntimeContext>,
    count: usize,
    mut initialize: impl FnMut(usize) -> Result<T, RuntimeError>,
) -> Result<ExactArrayStorage<T>, RuntimeError> {
    checkpoint(context)?;
    let result = initialize_host_array(context, count, |index| {
        checkpoint(context)?;
        initialize(index)
    })?;
    checkpoint(context)?;
    Ok(result)
}

pub(crate) fn shape(
    context: Option<&RuntimeContext>,
    rank: usize,
    mut dimension: impl FnMut(usize) -> Result<usize, RuntimeError>,
) -> Result<SharedShape, RuntimeError> {
    checkpoint(context)?;
    let raw = SharedShape::requested_bytes_for_rank(rank)
        .map_err(|source| allocation(context, source.into()))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| allocation(context, HostAllocationError::Overflow))?,
        1,
    )
    .map_err(|source| allocation(context, source))?;
    let receipt = HostAllocationReceipt::admit(
        context,
        HostAllocationPlan::new(0, retained).map_err(|source| allocation(context, source))?,
    )
    .map_err(|source| allocation(context, source))?;
    let result = if let Some(receipt) = receipt {
        let runtime = context.expect("managed shape receipt has an explicit owner");
        receipt
            .with_output_construction(runtime, |permit| {
                let mut builder = SharedShape::begin_exact(rank, permit)
                    .map_err(|source| allocation(context, source.into()))?;
                for index in 0..rank {
                    checkpoint(context)?;
                    builder
                        .try_push(dimension(index)?)
                        .map_err(|_| geometry())?;
                }
                builder.try_finish().map_err(|_| geometry())
            })
            .map_err(|source| allocation(context, source))??
    } else {
        let mut dimensions = runmat_value::exact_capacity_vec(rank)
            .map_err(|source| allocation(context, source.into()))?;
        for index in 0..rank {
            checkpoint(context)?;
            dimensions.push(dimension(index)?);
        }
        SharedShape::new_unmanaged(dimensions)
    };
    checkpoint(context)?;
    Ok(result)
}

pub(crate) fn geometry() -> RuntimeError {
    ExecutionServiceError::Infrastructure(
        "index planner initialization differs from its exact admitted geometry".into(),
    )
    .into_runtime_error()
}

/// Existing temporary owner, not a publishable plan or another allocator.
pub(in crate::indexing) enum Scratch<T: HostTypedElement> {
    Managed(HostTypedWorkspace<T>),
    Ordinary(Vec<T>),
}
impl<T: HostTypedElement> Scratch<T> {
    pub(in crate::indexing) fn new(
        context: Option<&RuntimeContext>,
        count: usize,
    ) -> Result<Self, RuntimeError> {
        checkpoint(context)?;
        let bytes = HostTypedWorkspace::<T>::quoted_bytes(count)
            .map_err(|source| allocation(context, source))?;
        let receipt = HostAllocationReceipt::admit(
            context,
            HostAllocationPlan::new(bytes, 0).map_err(|source| allocation(context, source))?,
        )
        .map_err(|source| allocation(context, source))?;
        match receipt {
            Some(receipt) => receipt
                .temporary_typed(
                    context.expect("managed scratch has an explicit owner"),
                    count,
                )
                .map(Self::Managed)
                .map_err(|source| allocation(context, source)),
            None => {
                let mut values = runmat_value::exact_capacity_vec(count)
                    .map_err(|source| allocation(context, source.into()))?;
                values.resize(count, T::default());
                Ok(Self::Ordinary(values))
            }
        }
    }
    pub(in crate::indexing) fn as_slice(&self) -> &[T] {
        match self {
            Self::Managed(values) => values.as_slice(),
            Self::Ordinary(values) => values,
        }
    }
    pub(in crate::indexing) fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Managed(values) => values.as_mut_slice(),
            Self::Ordinary(values) => values,
        }
    }
}
