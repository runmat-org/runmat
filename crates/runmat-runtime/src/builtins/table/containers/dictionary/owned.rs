//! Preflight a replayable native edit plan before allocating outputs or changing
//! either property. Only usize origins live in scratch, never unrooted Values.
use super::allocation::{add, allocation, exact_bytes, exact_error, Indices, Work};
use super::stream::{element_at, element_count, element_shape, Element, Query};
use super::*;
use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use runmat_value::{OutputListStorage, SharedShape};

const UNCHANGED: usize = usize::MAX;
const REMOVED: usize = usize::MAX - 1;

pub(super) fn assign(
    runtime: Option<&RuntimeContext>,
    mut object: ObjectInstance,
    query: Query<'_>,
    rhs: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let _source_root = lookup::rooted(runtime, Value::Object(object.clone()))?;
    let _rhs_root = lookup::rooted(runtime, rhs.clone())?;
    let _query_roots = query.roots(runtime)?;
    let keys = dictionary_cells(&object, "Keys")?;
    let values = dictionary_cells(&object, "Values")?;
    let old = keys.data().len();
    if old != values.data().len() {
        return Err(invalid_variable("dictionary: key/value storage diverged"));
    }
    let queries = query.count()?;
    let query_shape = query.shape(queries)?;
    let remove = option_value_is_empty(&rhs);
    let rhs_count = element_count(&rhs);
    if !remove {
        if rhs_count != 1 && rhs_count != queries {
            return Err(invalid_index("dictionary: assignment keys and values must have the same size unless values is scalar"));
        }
        if rhs_count > 1 && element_shape(&rhs).as_slice() != query_shape.as_slice() {
            return Err(invalid_index(
                "dictionary: non-scalar assignment values must have the same size as the keys",
            ));
        }
    }
    let capacity = old
        .checked_add(queries)
        .filter(|count| *count < REMOVED)
        .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    let scratch_len = capacity
        .checked_mul(2)
        .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    // Two exact native arrays: first key origin and last RHS origin for every
    // final slot. This same plan resolves duplicates in pending appended keys.
    let mut plan = Indices::new(runtime, scratch_len)?;
    for index in 0..old {
        plan.as_mut_slice()[index] = index;
        plan.as_mut_slice()[capacity + index] = UNCHANGED;
    }
    let key_class = keys.data().first().map(Element::Borrowed);
    let value_class = values.data().first().map(Element::Borrowed);
    let query_key = |index| query.at(index)?.normalize(key_class);
    let rhs_value =
        |index| element_at(&rhs, if rhs_count == 1 { 0 } else { index })?.normalize(value_class);
    let mut slots = old;
    for index in 0..queries {
        plan.check()?;
        let key = query_key(index)?;
        // Validate even shadowed RHS elements before any mutation/publication.
        if !remove {
            rhs_value(index)?;
        }
        let mut target = None;
        for slot in 0..slots {
            plan.check()?;
            if remove && plan.as_slice()[capacity + slot] == REMOVED {
                continue;
            }
            let origin = plan.as_slice()[slot];
            let existing = if origin < old {
                Element::Borrowed(&keys.data()[origin])
            } else {
                query_key(origin - old)?
            };
            if key.equal(existing)? {
                target = Some(slot);
                break;
            }
        }
        match (target, remove) {
            (Some(slot), true) => plan.as_mut_slice()[capacity + slot] = REMOVED,
            (None, true) => {}
            (Some(slot), false) => plan.as_mut_slice()[capacity + slot] = index,
            (None, false) => {
                plan.as_mut_slice()[slots] = old + index;
                plan.as_mut_slice()[capacity + slots] = index;
                slots += 1;
            }
        }
    }
    // Compact surviving origins in the same admitted scratch. Output replay
    // is linear even when a large dictionary receives one small update.
    let mut count = 0;
    for slot in 0..slots {
        plan.check()?;
        if plan.as_slice()[capacity + slot] != REMOVED {
            let key_origin = plan.as_slice()[slot];
            let rhs_origin = plan.as_slice()[capacity + slot];
            plan.as_mut_slice()[count] = key_origin;
            plan.as_mut_slice()[capacity + count] = rhs_origin;
            count += 1;
        }
    }
    let key_at = |slot| {
        let origin = plan.as_slice()[slot];
        if origin < old {
            Ok(Element::Borrowed(&keys.data()[origin]))
        } else {
            query_key(origin - old)
        }
    };
    let value_at = |slot| {
        let origin = plan.as_slice()[capacity + slot];
        if origin == UNCHANGED {
            Ok(Element::Borrowed(&values.data()[plan.as_slice()[slot]]))
        } else {
            rhs_value(origin)
        }
    };
    let raw_list = OutputListStorage::requested_bytes(count)
        .map_err(|cause| allocation(runtime, cause.into()))?;
    let raw_shape = SharedShape::requested_bytes_for_rank(2)
        .map_err(|cause| allocation(runtime, cause.into()))?;
    let mut retained = add(
        exact_bytes(raw_list, usize::from(count != 0), runtime)?,
        exact_bytes(raw_list, usize::from(count != 0), runtime)?,
        runtime,
    )?;
    retained = add(retained, exact_bytes(raw_shape, 1, runtime)?, runtime)?;
    for index in 0..count {
        plan.check()?;
        retained = add(retained, key_at(index)?.retained_bytes(runtime)?, runtime)?;
        retained = add(retained, value_at(index)?.retained_bytes(runtime)?, runtime)?;
    }
    // This receipt quotes all newly retained scalar payloads/shape claims as
    // well as both slot lists and their shared shape before construction.
    let work = Work::new(runtime, 0, retained)?;
    let (new_keys, new_values) = work.construct(|permit| {
        let materialize_key = |index| {
            plan.check()?;
            work.check()?;
            key_at(index)?.materialize(runtime, permit)
        };
        let materialize_value = |index| {
            plan.check()?;
            work.check()?;
            value_at(index)?.materialize(runtime, permit)
        };
        let dimensions = [usize::from(count != 0), count];
        let (keys, values) = match permit {
            Some(permit) => {
                let shape = SharedShape::new_admitted_from_slice(&dimensions, permit)
                    .map_err(|cause| allocation(runtime, cause.into()))?;
                let keys = OutputListStorage::try_initialize(count, permit, materialize_key)
                    .map_err(|cause| exact_error(runtime, cause))?;
                let values = OutputListStorage::try_initialize(count, permit, materialize_value)
                    .map_err(|cause| exact_error(runtime, cause))?;
                (
                    CellArray::from_admitted(keys, shape.clone()).map_err(invalid_variable)?,
                    CellArray::from_admitted(values, shape).map_err(invalid_variable)?,
                )
            }
            None => {
                let keys = (0..count)
                    .map(materialize_key)
                    .collect::<BuiltinResult<Vec<_>>>()?;
                let values = (0..count)
                    .map(materialize_value)
                    .collect::<BuiltinResult<Vec<_>>>()?;
                (
                    CellArray::new_with_shape(keys, dimensions.to_vec())
                        .map_err(invalid_variable)?,
                    CellArray::new_with_shape(values, dimensions.to_vec())
                        .map_err(invalid_variable)?,
                )
            }
        };
        Ok((Value::Cell(keys), Value::Cell(values)))
    })?;
    // Root both complete outputs before snapshot preparation. Any root,
    // quota, cancellation, or object-copy failure leaves object untouched.
    let new_keys = lookup::rooted(runtime, new_keys)?;
    let new_values = lookup::rooted(runtime, new_values)?;
    plan.check()?;
    work.check()?;
    let updates = [("Keys", new_keys.value()), ("Values", new_values.value())];
    crate::execution::object::replacement::replace_properties(
        runtime,
        &mut object,
        updates.into_iter(),
        |_, cause| allocation(runtime, cause),
        |message| invalid_variable(format!("dictionary: {message}")),
    )?;
    // The snapshot is still local, not a workspace publication. Root it under
    // the selected context before retiring either original admission fence.
    let output = lookup::rooted(runtime, Value::Object(object))?;
    plan.check()?;
    work.check()?;
    Ok(output)
}
