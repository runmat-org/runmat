use crate::sequence::ValueSequence;
use crate::{
    context::{
        host_allocation::{
            host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
            HostAllocationReceipt,
        },
        RuntimeContext,
    },
    execution::{structure::replacement, RootedValue, RootedValueLease, RootedValueSequence},
    RuntimeError,
};
use runmat_value::{OutputListStorage, StructArray, UnboundedStructArrayBuilder, Value};

pub fn gather_member(
    runtime: Option<&RuntimeContext>,
    array: StructArray,
    field: &str,
) -> Result<RootedValueSequence, RuntimeError> {
    let original = Value::StructArray(array.clone());
    let _roots = replacement::lease(&original, runtime)?;
    let values = array
        .column_owner(field)
        .ok_or_else(|| RuntimeError::from(format!("Undefined field '{field}'")))?;
    if let Some(runtime) = runtime {
        promote_column(runtime, values)?;
    }
    let sequence = ValueSequence::from_output_storage(values.clone())
        .map_err(crate::sequence::sequence_error_to_runtime)?;
    match runtime {
        Some(runtime) => RootedValueSequence::in_context(sequence, runtime),
        None => RootedValueSequence::new(sequence).map_err(|e| e.into_runtime_error()),
    }
}
pub fn assign_member_values<OnWrite>(
    runtime: Option<&RuntimeContext>,
    array: StructArray,
    field: String,
    values: Vec<Value>,
    on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    if values.len() != array.len() {
        return Err(format!("structure-array field assignment requires exactly one value per destination element (expected {}, received {})",array.len(),values.len()).into());
    }
    let original = Value::StructArray(array.clone());
    let _base_roots = replacement::lease(&original, runtime)?;
    let _value_roots = match runtime {
        Some(runtime) => RootedValueLease::for_value_slices_in_context(&[&values], runtime)?,
        None => RootedValueLease::for_values(values.iter()).map_err(|e| e.into_runtime_error())?,
    };
    let column = adopt_column(runtime, values)?;
    assign_column(runtime, array, &field, column, on_write)
}

/// Retain the canonical RHS column, including shared exact backing, without
/// materializing a second collection of member values.
pub fn assign_member_sequence<OnWrite>(
    runtime: Option<&RuntimeContext>,
    array: StructArray,
    field: runmat_value::Utf8Text,
    values: ValueSequence,
    on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    if values.len() != array.len() {
        return Err(format!("structure-array field assignment requires exactly one value per destination element (expected {}, received {})",array.len(),values.len()).into());
    }
    let original = Value::StructArray(array.clone());
    let _base_roots = replacement::lease(&original, runtime)?;
    let _value_roots = match runtime {
        Some(runtime) => {
            RootedValueLease::for_value_slices_in_context(&[values.as_slice()], runtime)?
        }
        None => RootedValueLease::for_values(values.iter()).map_err(|e| e.into_runtime_error())?,
    };
    let column = match values.into_legacy_value() {
        Value::OutputList(column) if column.is_shallow_clone() => {
            if let Some(runtime) = runtime {
                promote_column(runtime, &column)?;
            }
            column
        }
        Value::OutputList(column) => {
            let values = crate::sequence::into_unbounded_output_values(column)?;
            adopt_column(runtime, values)?
        }
        value => {
            // A structure column needs its own shared backing even when the
            // source expression carries one inline value. Construct that final
            // backing directly under the existing output allocation owner.
            let values = match runtime {
                Some(runtime) => {
                    let mut value = Some(value);
                    crate::execution::RootedValues::try_initialize_in_context(1, runtime, |_| {
                        Ok::<_, RuntimeError>(value.take().expect("one column element"))
                    })?
                }
                None => crate::execution::RootedValues::new(vec![value])
                    .map_err(|error| error.into_runtime_error())?,
            };
            values.with_owned_output_storage(|values| values)
        }
    };
    assign_column(runtime, array, field.as_str(), column, on_write)
}

fn assign_column<OnWrite>(
    runtime: Option<&RuntimeContext>,
    array: StructArray,
    field: &str,
    column: OutputListStorage,
    mut on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    let existing = array.column_owner(field);
    let columns = array
        .column_owners()
        .map(|(name, previous)| (name, if name == field { &column } else { previous }))
        .chain(existing.is_none().then_some((field, &column)));
    let output = if let Some(runtime) = runtime {
        crate::execution::construct_struct_array_from_columns_in_context(
            runtime,
            columns,
            array.shape_owner(),
        )?
    } else {
        let mut builder = UnboundedStructArrayBuilder::with_capacity(
            array.shape_owner().clone(),
            columns.clone().count(),
        )
        .map_err(|e| RuntimeError::from(e.to_string()))?;
        for (name, values) in columns {
            builder
                .insert(name.to_owned(), values.clone())
                .map_err(|e| RuntimeError::from(e.to_string()))?;
        }
        RootedValue::new(Value::StructArray(builder.finish()))
            .map_err(|e| e.into_runtime_error())?
    };
    if let Some(previous) = existing {
        for (previous, next) in previous.iter().zip(column.iter()) {
            on_write(previous, next);
        }
    }
    Ok(output)
}
fn promote_column(
    runtime: &RuntimeContext,
    column: &OutputListStorage,
) -> Result<(), RuntimeError> {
    use crate::context::host_allocation::HostAllocationPolicy;
    if HostAllocationReceipt::policy(Some(runtime)).map_err(|e| allocation(Some(runtime), e))?
        == HostAllocationPolicy::Unbounded
    {
        return Ok(());
    }
    let bytes = column.unclaimed_shared_bytes().map_err(|e| {
        crate::runtime_error::semantic_error_with_source(
            "StructureMember",
            "structure column is not canonical shared storage",
            e,
        )
    })?;
    if let Some(bytes) = bytes {
        let receipt = admit(Some(runtime), bytes, 1)?.expect("managed policy checked");
        let permit = receipt
            .output_permit(runtime)
            .map_err(|e| allocation(Some(runtime), e))?;
        let result = column.try_promote_shared_claim(&permit);
        receipt
            .checkpoint(runtime)
            .map_err(|e| allocation(Some(runtime), e))?;
        result.map_err(|e| {
            crate::runtime_error::semantic_error_with_source(
                "StructureMember",
                "column claim promotion failed",
                e,
            )
        })?;
    }
    Ok(())
}
fn adopt_column(
    runtime: Option<&RuntimeContext>,
    values: Vec<Value>,
) -> Result<OutputListStorage, RuntimeError> {
    let capacity = values.capacity();
    let bytes = OutputListStorage::requested_bytes_for_capacity(capacity)
        .map_err(|e| allocation(runtime, e.into()))?;
    let receipt = admit(runtime, bytes, usize::from(capacity != 0))?;
    match receipt {
        Some(receipt) => {
            let runtime = runtime.expect("managed context");
            let permit = receipt
                .output_permit(runtime)
                .map_err(|e| allocation(Some(runtime), e))?;
            let values = OutputListStorage::try_adopt_exact(values, &permit);
            receipt
                .checkpoint(runtime)
                .map_err(|e| allocation(Some(runtime), e))?;
            values.map_err(|e| allocation(Some(runtime), (*e.source()).into()))
        }
        None => Ok(OutputListStorage::shared_unbounded(values)),
    }
}
fn admit(
    runtime: Option<&RuntimeContext>,
    bytes: usize,
    claims: usize,
) -> Result<Option<HostAllocationReceipt>, RuntimeError> {
    let bytes =
        u64::try_from(bytes).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?;
    let bytes = HostAllocationPlan::retained_exact_output_bytes(bytes, claims)
        .map_err(|e| allocation(runtime, e))?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(|e| allocation(runtime, e))?;
    HostAllocationReceipt::admit(runtime, plan).map_err(|e| allocation(runtime, e))
}
fn allocation(runtime: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:StructureMember:Allocation",
        "structure member admission failed",
        source,
    )
}
