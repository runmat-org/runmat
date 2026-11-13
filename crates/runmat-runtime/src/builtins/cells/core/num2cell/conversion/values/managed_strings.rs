//! Exact publication of partitioned string values. Unique source text bytes
//! move into the result; shared ordinary text is copied under this receipt.

use runmat_value::{
    CellArray, ExactArrayInitializationError, ExactArrayStorage, ExactHostAllocationPermit,
    OutputListStorage, OwnedExactArrayValues, SharedShape, StringArray, StringElement, Value,
};

use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

use super::super::super::{
    error,
    plan::{ManagedPartitionPlan, PartitionGeometry},
};

mod quote;

enum Source {
    Unique(OwnedExactArrayValues<StringElement>),
    Shared(ExactArrayStorage<StringElement>),
}

fn admission(runtime: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(runtime),
        "RunMat:num2cell:HostAllocation",
        "num2cell: string output admission failed",
        cause,
    )
}

fn numeric(
    runtime: &RuntimeContext,
    cause: runmat_value::ExactHostNumericAllocationError,
) -> RuntimeError {
    admission(runtime, HostAllocationError::Numeric(cause))
}

impl Source {
    fn take(
        &mut self,
        index: usize,
        permit: &dyn ExactHostAllocationPermit,
        runtime: &RuntimeContext,
        receipt: &HostAllocationReceipt,
    ) -> BuiltinResult<StringElement> {
        match self {
            Self::Unique(values) => {
                let text = values
                    .as_mut_slice()
                    .get_mut(index)
                    .ok_or_else(|| error::internal("string partition index is out of bounds"))?;
                let text = std::mem::take(text);
                if text.has_retained_claim() || text.capacity() == 0 {
                    Ok(text)
                } else {
                    text.try_admit_exact(permit)
                        .map_err(|failure| numeric(runtime, failure.source()))
                }
            }
            Self::Shared(values) => {
                let text = values
                    .get(index)
                    .ok_or_else(|| error::internal("string partition index is out of bounds"))?;
                if text.has_retained_claim() || text.capacity() == 0 {
                    return Ok(text.clone());
                }
                text.try_copy_exact_with(permit, |_| {
                    receipt
                        .checkpoint(runtime)
                        .map_err(|cause| admission(runtime, cause))
                })
                .map_err(|cause| match cause {
                    ExactArrayInitializationError::Allocation(cause) => numeric(runtime, cause),
                    ExactArrayInitializationError::Element(cause) => cause,
                })
            }
        }
    }
}

pub(in crate::builtins::cells::core::num2cell) fn convert(
    input: StringArray,
    dimensions: Option<&Value>,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    let plan = ManagedPartitionPlan::prepare(&input.shape, dimensions, runtime)?;
    let unique = input.data_owner().is_unique();
    let retained = quote::retained_bytes(runtime, &input, &plan, unique)?;
    let allocation =
        HostAllocationPlan::new(0, retained).map_err(|cause| admission(runtime, cause))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), allocation)
        .map_err(|cause| admission(runtime, cause))?
        .ok_or_else(|| admission(runtime, HostAllocationError::MissingManagedOwner))?;
    let (data, _source_shape, _, _) = input.into_parts();
    let mut source = if unique {
        Source::Unique(
            data.try_into_owned()
                .map_err(|_| admission(runtime, HostAllocationError::StaleAuthority))?,
        )
    } else {
        Source::Shared(data)
    };
    let result = receipt
        .with_output_construction(runtime, |permit| {
            let shape: &[usize] = if plan.output_shape().is_empty() && plan.output_count() == 1 {
                &[1, 1]
            } else {
                plan.output_shape()
            };
            let output_shape = SharedShape::new_admitted_from_slice(shape, permit)
                .map_err(|cause| numeric(runtime, cause))?;
            let outputs =
                OutputListStorage::try_initialize(plan.output_count(), permit, |output| {
                    if output % 4096 == 0 {
                        receipt
                            .checkpoint(runtime)
                            .map_err(|cause| admission(runtime, cause))?;
                    }
                    if plan.slice_count() == 1 {
                        let source_index = plan.source_index(output, 0)?;
                        return source
                            .take(source_index, permit, runtime, &receipt)
                            .map(|text| Value::String(text.into()));
                    }
                    let group_shape =
                        SharedShape::new_admitted_from_slice(plan.slice_shape(), permit)
                            .map_err(|cause| numeric(runtime, cause))?;
                    let group =
                        ExactArrayStorage::try_initialize(plan.slice_count(), permit, |slice| {
                            if slice % 4096 == 0 {
                                receipt
                                    .checkpoint(runtime)
                                    .map_err(|cause| admission(runtime, cause))?;
                            }
                            let source_index = plan.source_index(output, slice)?;
                            source.take(source_index, permit, runtime, &receipt)
                        })
                        .map_err(|cause| match cause {
                            ExactArrayInitializationError::Allocation(cause) => {
                                numeric(runtime, cause)
                            }
                            ExactArrayInitializationError::Element(cause) => cause,
                        })?;
                    StringArray::from_admitted(group, group_shape)
                        .map(Value::StringArray)
                        .map_err(error::internal)
                })
                .map_err(|cause| match cause {
                    ExactArrayInitializationError::Allocation(cause) => numeric(runtime, cause),
                    ExactArrayInitializationError::Element(cause) => cause,
                })?;
            CellArray::from_admitted(outputs, output_shape)
                .map(Value::Cell)
                .map_err(error::internal)
        })
        .map_err(|cause| admission(runtime, cause))?;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| admission(runtime, cause))?;
    result
}
