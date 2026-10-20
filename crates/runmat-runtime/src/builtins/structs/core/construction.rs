//! Context-aware publication shared by structure metadata operations.

use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::execution::{RootedValue, RootedValueLease, RootedValues};
use runmat_value::{OutputListStorage, SharedShape, UnboundedStructArrayBuilder, Value};

/// The caller retains its input root lease until the returned root is published.
pub(super) fn scalar<'a, I>(
    runtime: Option<&RuntimeContext>,
    fields: I,
) -> crate::BuiltinResult<RootedValue>
where
    I: Iterator<Item = (&'a str, &'a Value)> + Clone,
{
    crate::execution::structure::construct_struct_from_fields_in_optional_context(runtime, fields)
}

pub(super) fn root(
    runtime: Option<&RuntimeContext>,
    value: Value,
) -> crate::BuiltinResult<RootedValue> {
    RootedValue::in_optional_context(value, runtime)
}

pub(super) fn array<'a, I>(
    runtime: Option<&RuntimeContext>,
    columns: I,
    shape: &SharedShape,
) -> crate::BuiltinResult<RootedValue>
where
    I: Iterator<Item = (&'a str, &'a OutputListStorage)> + Clone,
{
    if let Some(runtime) = runtime {
        return crate::execution::construct_struct_array_from_columns_in_context(
            runtime, columns, shape,
        );
    }
    require_unbounded()?;
    let mut builder =
        UnboundedStructArrayBuilder::with_capacity(shape.clone(), columns.clone().count())
            .map_err(|source| {
                crate::runtime_error::semantic_error_with_source(
                    "StructArrayConstruction",
                    "invalid structure-array destination",
                    source,
                )
            })?;
    for (name, values) in columns {
        builder
            .insert(name.to_owned(), values.clone())
            .map_err(|source| match source {
                runmat_value::StructArrayFieldError::InvalidColumn { .. } => {
                    crate::runtime_error::semantic_error(
                        "StructArrayConstruction",
                        "invalid structure-array column",
                    )
                }
                runmat_value::StructArrayFieldError::Map(source) => {
                    crate::runtime_error::semantic_error_with_source(
                        "StructArrayConstruction",
                        "structure-array column insertion failed",
                        source.into_parts().0,
                    )
                }
            })?;
    }
    root(None, Value::StructArray(builder.finish()))
}

pub(super) fn input(
    runtime: Option<&RuntimeContext>,
    value: &Value,
) -> crate::BuiltinResult<RootedValueLease> {
    match runtime {
        Some(runtime) => {
            RootedValueLease::for_values_in_context(std::slice::from_ref(value), runtime)
        }
        None => {
            require_unbounded()?;
            RootedValueLease::for_traced(value).map_err(|error| error.into_runtime_error())
        }
    }
}

fn require_unbounded() -> crate::BuiltinResult<()> {
    let policy = HostAllocationReceipt::policy(None).map_err(|source| allocation(None, source))?;
    match policy {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => {
            Err(allocation(None, HostAllocationError::MissingManagedOwner))
        }
    }
}

/// Initialize one replacement column; callers keep its source values rooted.
pub(super) fn column(
    runtime: Option<&RuntimeContext>,
    count: usize,
    mut value: impl FnMut(usize) -> Value,
) -> crate::BuiltinResult<RootedValues> {
    if let Some(runtime) = runtime {
        return RootedValues::try_initialize_in_context(count, runtime, |index| {
            Ok::<_, crate::RuntimeError>(value(index))
        });
    }
    require_unbounded()?;
    let mut values = runmat_value::exact_capacity_vec(count)
        .map_err(|source| allocation(None, HostAllocationError::Numeric(source)))?;
    for index in 0..count {
        values.push(value(index));
    }
    RootedValues::new(values).map_err(|error| error.into_runtime_error())
}

pub(super) fn empty_array(runtime: Option<&RuntimeContext>) -> crate::BuiltinResult<RootedValue> {
    use runmat_value::{NumericDType, Tensor};

    let shape = [0, 0];
    let allocation = |source| allocation(runtime, source);
    let bytes =
        HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &shape).map_err(allocation)?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(allocation)?;
    let receipt = HostAllocationReceipt::admit(runtime, plan).map_err(allocation)?;
    let tensor = match (runtime, receipt) {
        (Some(runtime), Some(receipt)) => {
            let tensor = receipt
                .exact_output_tensor(runtime, NumericDType::F64, &shape)
                .map_err(allocation)?
                .into_tensor()
                .map_err(|source| allocation(HostAllocationError::Numeric(source)))?;
            receipt.checkpoint(runtime).map_err(allocation)?;
            tensor
        }
        (_, None) => {
            if HostAllocationReceipt::policy(runtime).map_err(allocation)?
                != HostAllocationPolicy::Unbounded
            {
                return Err(allocation(HostAllocationError::StaleAuthority));
            }
            let tensor = Tensor::zeros(shape.to_vec());
            if HostAllocationReceipt::policy(runtime).map_err(allocation)?
                != HostAllocationPolicy::Unbounded
            {
                return Err(allocation(HostAllocationError::StaleAuthority));
            }
            tensor
        }
        (None, Some(_)) => return Err(allocation(HostAllocationError::MissingManagedOwner)),
    };
    root(runtime, Value::Tensor(tensor))
}

fn allocation(
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> crate::RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:Struct:Allocation",
        "structure construction requires its selected allocation context",
        source,
    )
}
