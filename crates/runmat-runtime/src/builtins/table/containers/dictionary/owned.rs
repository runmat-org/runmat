//! Move dictionary cells out of the original object. Exact backing remains
//! claimed through edits and is republished without cloning child Values.

use super::*;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationPolicy,
    HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use runmat_value::{
    ExactArrayInitializationError, OutputListStorage, OwnedExactArrayValues, SharedShape,
};

fn admission(runtime: &RuntimeContext, cause: HostAllocationError) -> crate::RuntimeError {
    host_allocation_runtime_error(
        Some(runtime),
        "RunMat:dictionary:HostAllocation",
        "dictionary: replacement admission failed",
        cause,
    )
}

fn take_cell(
    object: &mut ObjectInstance,
    name: &'static str,
) -> BuiltinResult<(String, CellArray)> {
    let (key, value) = object
        .properties
        .remove_entry(name)
        .ok_or_else(|| invalid_variable(format!("dictionary: missing {name} storage")))?;
    match value {
        Value::Cell(cell) => Ok((key, cell)),
        other => Err(invalid_variable(format!(
            "dictionary: {name} storage must be a cell array, got {other:?}"
        ))),
    }
}

fn move_values(mut cell: CellArray, managed: bool) -> BuiltinResult<OwnedExactArrayValues<Value>> {
    if !managed {
        cell.data
            .make_unique_unbounded()
            .map_err(|error| invalid_variable(error.to_string()))?;
    }
    cell.into_owned_data().map_err(|_| {
        invalid_variable("dictionary: shared admitted cell backing requires a new output owner")
    })
}

fn collect_exact(
    old: OwnedExactArrayValues<Value>,
    appended: Vec<Value>,
    count: usize,
    permit: &dyn runmat_value::ExactHostAllocationPermit,
) -> BuiltinResult<OutputListStorage> {
    let mut original = old.into_iter();
    let mut appended = appended.into_iter();
    OutputListStorage::try_initialize(count, permit, |_| {
        original
            .next()
            .or_else(|| appended.next())
            .ok_or_else(|| invalid_variable("dictionary: replacement collection is incomplete"))
    })
    .map_err(|cause| match cause {
        ExactArrayInitializationError::Allocation(cause) => {
            invalid_variable(format!("dictionary: output allocation failed: {cause}"))
        }
        ExactArrayInitializationError::Element(cause) => cause,
    })
}

pub(super) fn assign(
    runtime: Option<&RuntimeContext>,
    mut object: ObjectInstance,
    payload: &Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let (query_keys, query_shape) = dictionary_selector_values(payload)?;
    dictionary_cells(&object, "Keys")?;
    dictionary_cells(&object, "Values")?;
    let (keys_name, keys) = take_cell(&mut object, "Keys")?;
    let (values_name, values) = take_cell(&mut object, "Values")?;
    let query_keys = convert_queries_to_key_class(&keys, query_keys)?;
    let policy = HostAllocationReceipt::policy(runtime).map_err(|cause| match runtime {
        Some(runtime) => admission(runtime, cause),
        None => invalid_variable(format!("dictionary: allocation policy failed: {cause}")),
    })?;
    let managed = policy == HostAllocationPolicy::Managed;
    let keys_capacity = if keys.data.is_unique() {
        keys.data.capacity()
    } else {
        keys.data.len()
    };
    let values_capacity = if values.data.is_unique() {
        values.data.capacity()
    } else {
        values.data.len()
    };
    let keys_exact = keys.data.is_exact();
    let values_exact = values.data.is_exact();
    let mut keys = move_values(keys, managed)?;
    let mut values = move_values(values, managed)?;
    let mut append_keys = Vec::new();
    let mut append_values = Vec::new();
    if option_value_is_empty(&rhs) {
        for query in query_keys {
            if let Some(index) = keys
                .as_slice()
                .iter()
                .position(|key| dictionary_keys_equal(key, &query))
            {
                keys.try_remove(index);
                values.try_remove(index);
            }
        }
    } else {
        let mut rhs_values = value_elements(&rhs)?;
        if rhs_values.len() == 1 && query_keys.len() != 1 {
            rhs_values.resize(query_keys.len(), rhs_values[0].clone());
        } else if rhs_values.len() != query_keys.len() {
            return Err(invalid_index("dictionary: assignment keys and values must have the same size unless values is scalar"));
        } else if rhs_values.len() > 1 && dictionary_element_shape(&rhs) != query_shape {
            return Err(invalid_index(
                "dictionary: non-scalar assignment values must have the same size as the keys",
            ));
        }
        normalize_to_existing_class(values.as_slice(), &mut rhs_values)?;
        for (query, mut value) in query_keys.into_iter().zip(rhs_values) {
            if let Some(index) = keys
                .as_slice()
                .iter()
                .position(|key| dictionary_keys_equal(key, &query))
            {
                std::mem::swap(&mut values.as_mut_slice()[index], &mut value);
            } else {
                append_keys.push(query);
                append_values.push(value);
            }
        }
    }
    let count = keys
        .len()
        .checked_add(append_keys.len())
        .ok_or_else(|| invalid_variable("dictionary: key count exceeds platform limits"))?;
    if Some(count) != values.len().checked_add(append_values.len()) {
        return Err(invalid_variable("dictionary: key/value storage diverged"));
    }
    let in_place = count <= keys_capacity && count <= values_capacity;
    if in_place {
        for (key, value) in append_keys.drain(..).zip(append_values.drain(..)) {
            keys.try_push(key)
                .map_err(|_| invalid_variable("dictionary: key capacity changed"))?;
            values
                .try_push(value)
                .map_err(|_| invalid_variable("dictionary: value capacity changed"))?;
        }
    }
    let mut publication_receipt = None;
    let (keys_data, values_data, shape) = if managed {
        let runtime =
            runtime.ok_or_else(|| invalid_variable("dictionary: Managed runtime is absent"))?;
        let replace = !in_place || !keys_exact || !values_exact;
        let raw_list = if replace {
            OutputListStorage::requested_bytes(count)
                .map_err(|cause| admission(runtime, HostAllocationError::Numeric(cause)))?
                .checked_mul(2)
                .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?
        } else {
            0
        };
        let raw_shape = SharedShape::requested_bytes_for_rank(2)
            .map_err(|cause| admission(runtime, HostAllocationError::Numeric(cause)))?;
        let raw = u64::try_from(
            raw_list
                .checked_add(raw_shape)
                .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?,
        )
        .map_err(|_| admission(runtime, HostAllocationError::Overflow))?;
        let claims = 1 + if replace && count != 0 { 2 } else { 0 };
        let retained = HostAllocationPlan::retained_exact_output_bytes(raw, claims)
            .map_err(|cause| admission(runtime, cause))?;
        let receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(0, retained).map_err(|cause| admission(runtime, cause))?,
        )
        .map_err(|cause| admission(runtime, cause))?
        .ok_or_else(|| admission(runtime, HostAllocationError::MissingManagedOwner))?;
        let result = receipt.with_output_construction(runtime, |permit| {
            let shape =
                SharedShape::new_admitted_from_slice(&[usize::from(count != 0), count], permit)
                    .map_err(|cause| admission(runtime, HostAllocationError::Numeric(cause)))?;
            let (keys_data, values_data) = if replace {
                (
                    collect_exact(keys, append_keys, count, permit)?,
                    collect_exact(values, append_values, count, permit)?,
                )
            } else {
                (
                    keys.try_into_storage().map_err(|_| {
                        invalid_variable("dictionary: key header allocation failed")
                    })?,
                    values.try_into_storage().map_err(|_| {
                        invalid_variable("dictionary: value header allocation failed")
                    })?,
                )
            };
            Ok::<_, crate::RuntimeError>((keys_data, values_data, shape))
        });
        receipt
            .checkpoint(runtime)
            .map_err(|cause| admission(runtime, cause))?;
        let (keys_data, values_data, shape) =
            result.map_err(|cause| admission(runtime, cause))??;
        publication_receipt = Some(receipt);
        (keys_data, values_data, shape)
    } else {
        let keys_data = if in_place {
            keys.try_into_storage()
                .map_err(|_| invalid_variable("dictionary: key header allocation failed"))?
        } else {
            OutputListStorage::shared_unbounded(keys.into_iter().chain(append_keys).collect())
        };
        let values_data = if in_place {
            values
                .try_into_storage()
                .map_err(|_| invalid_variable("dictionary: value header allocation failed"))?
        } else {
            OutputListStorage::shared_unbounded(values.into_iter().chain(append_values).collect())
        };
        (
            keys_data,
            values_data,
            SharedShape::new_unmanaged(vec![usize::from(count != 0), count]),
        )
    };
    let keys_cell = if managed {
        CellArray::from_admitted(keys_data, shape.clone())
    } else {
        CellArray::from_shared_storage(keys_data, shape.clone())
    }
    .map_err(invalid_variable)?;
    let values_cell = if managed {
        CellArray::from_admitted(values_data, shape)
    } else {
        CellArray::from_shared_storage(values_data, shape)
    }
    .map_err(invalid_variable)?;
    object.properties.insert(keys_name, Value::Cell(keys_cell));
    object
        .properties
        .insert(values_name, Value::Cell(values_cell));
    if let (Some(runtime), Some(receipt)) = (runtime, publication_receipt.as_ref()) {
        receipt
            .checkpoint(runtime)
            .map_err(|cause| admission(runtime, cause))?;
    }
    Ok(Value::Object(object))
}
