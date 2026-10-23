//! Exact physical host-array construction at the canonical allocation owner.
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::RuntimeError;
use runmat_value::{ExactArrayStorage, ExactArrayStorageBuilder};

pub(crate) fn allocation_error(
    context: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> RuntimeError {
    host_allocation_runtime_error(
        context,
        "RunMat:HostAllocation:Array",
        "exact host array storage",
        source,
    )
}

fn receipt<T>(
    context: Option<&RuntimeContext>,
    count: usize,
) -> Result<Option<HostAllocationReceipt>, RuntimeError> {
    let bytes = ExactArrayStorage::<T>::requested_bytes(count)
        .map_err(|source| allocation_error(context, source.into()))?;
    let bytes = u64::try_from(bytes)
        .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(bytes, usize::from(count != 0))
        .map_err(|source| allocation_error(context, source))?;
    HostAllocationReceipt::admit(
        context,
        HostAllocationPlan::new(0, retained).map_err(|source| allocation_error(context, source))?,
    )
    .map_err(|source| allocation_error(context, source))
}

/// Same exact immutable array owner for path/control metadata and selectors.
/// The ordinary branch has no fabricated permit and allocates no second store.
pub fn initialize_host_array<T>(
    context: Option<&RuntimeContext>,
    count: usize,
    mut initialize: impl FnMut(usize) -> Result<T, RuntimeError>,
) -> Result<ExactArrayStorage<T>, RuntimeError> {
    match receipt::<T>(context, count)? {
        Some(receipt) => {
            let runtime = context.expect("managed receipt has an explicit context");
            let result = receipt
                .with_output_construction(runtime, |permit| {
                    ExactArrayStorage::try_initialize(count, permit, &mut initialize)
                })
                .map_err(|source| allocation_error(context, source))?;
            result.map_err(|source| match source {
                runmat_value::ExactArrayInitializationError::Allocation(source) => {
                    allocation_error(context, source.into())
                }
                runmat_value::ExactArrayInitializationError::Element(error) => error,
            })
        }
        None => {
            let mut values = runmat_value::exact_capacity_vec(count)
                .map_err(|source| allocation_error(context, source.into()))?;
            for index in 0..count {
                values.push(initialize(index)?);
            }
            Ok(ExactArrayStorage::shared_unbounded(values))
        }
    }
}

/// Finite and ordinary construction share one typed selector publication.
/// Both branches allocate the full capacity before the caller consumes CSLs.
pub struct PreparedHostArray<T> {
    body: IndexArrayBody<T>,
    count: usize,
}
impl<T> std::fmt::Debug for PreparedHostArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedHostArray")
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}
enum IndexArrayBody<T> {
    Managed(ExactArrayStorageBuilder<T>),
    Ordinary(Vec<T>),
}
pub fn prepare_host_array<T>(
    context: &RuntimeContext,
    count: usize,
) -> Result<PreparedHostArray<T>, RuntimeError> {
    prepare_host_array_optional(Some(context), count)
}

/// The same builder for existing ordinary APIs without an active context.
/// A finite policy still needs the canonical explicit managed owner.
pub fn prepare_host_array_optional<T>(
    context: Option<&RuntimeContext>,
    count: usize,
) -> Result<PreparedHostArray<T>, RuntimeError> {
    let body = match receipt::<T>(context, count)? {
        Some(receipt) => IndexArrayBody::Managed(
            receipt
                .with_output_allocation(
                    context.expect("managed array receipt has an explicit owner"),
                    |permit| ExactArrayStorage::try_begin(count, permit),
                )
                .map_err(|source| allocation_error(context, source))?,
        ),
        None => IndexArrayBody::Ordinary(
            runmat_value::exact_capacity_vec(count)
                .map_err(|source| allocation_error(context, source.into()))?,
        ),
    };
    Ok(PreparedHostArray { body, count })
}
impl<T> PreparedHostArray<T> {
    /// Borrow the initialized prefix without publishing or reallocating it.
    pub fn initialized(&self) -> &[T] {
        match &self.body {
            IndexArrayBody::Managed(builder) => builder.initialized(),
            IndexArrayBody::Ordinary(values) => values,
        }
    }

    /// Publish the initialized prefix with its original capacity owner/claim.
    pub fn finish_initialized(self) -> ExactArrayStorage<T> {
        match self.body {
            IndexArrayBody::Managed(builder) => builder.finish_initialized(),
            IndexArrayBody::Ordinary(values) => ExactArrayStorage::shared_unbounded(values),
        }
    }

    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        match &mut self.body {
            IndexArrayBody::Managed(builder) => builder.try_push(value),
            IndexArrayBody::Ordinary(values) if values.len() < self.count => {
                values.push(value);
                Ok(())
            }
            IndexArrayBody::Ordinary(_) => Err(value),
        }
    }
    pub fn try_finish(self) -> Result<ExactArrayStorage<T>, Self> {
        match self.body {
            IndexArrayBody::Managed(builder) => builder.try_finish().map_err(|builder| Self {
                body: IndexArrayBody::Managed(builder),
                count: self.count,
            }),
            IndexArrayBody::Ordinary(values) if values.len() == self.count => {
                Ok(ExactArrayStorage::shared_unbounded(values))
            }
            IndexArrayBody::Ordinary(values) => Err(Self {
                body: IndexArrayBody::Ordinary(values),
                count: self.count,
            }),
        }
    }
}
