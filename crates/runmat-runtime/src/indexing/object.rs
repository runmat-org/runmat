//! Default parenthesis indexing for homogeneous value and handle-object arrays.
//!
//! Class-defined `subsref` and `subsasgn` methods are resolved before these
//! functions are called. This module owns the ordinary array behavior shared
//! by the bytecode and native executors.

use runmat_types::ClassIdentity;
use runmat_value::{ObjectArray, Value};

use crate::indexing::plan::IndexPlan;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

mod growth;

pub fn shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Object(_) | Value::HandleObject(_) => Some(vec![1, 1]),
        Value::ObjectArray(array) => Some(array.shape().to_vec()),
        _ => None,
    }
}

pub fn read_with_plan(value: &Value, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    read_with_plan_in_context(context.as_ref(), value, plan)
}

pub fn read_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    value: &Value,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    if plan
        .output_shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        != Some(plan.indices.len())
    {
        return Err(semantic_error(
            "InvalidIndexSelectorPlan",
            "Object-array index shape differs from its selection count",
        ));
    }
    let array = as_array_in_context(context, value.clone())?;
    if let [index] = plan.indices.as_slice() {
        return array.get_linear(*index as usize).cloned().ok_or_else(|| {
            semantic_error("IndexOutOfBounds", "Object array index is out of bounds")
        });
    }
    crate::execution::object_array::select(
        &array,
        plan.indices.iter().map(|&index| index as usize),
        &plan.output_shape,
        context,
    )
    .map(|output| output.with_owned(|value| value))
}

pub fn assign_scalar_indices(
    base: Value,
    indices: &[usize],
    rhs: Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    if delete {
        return delete_scalar_indices_in_context(context.as_ref(), base, indices, &rhs);
    }
    assign_scalar_indices_in_context(context.as_ref(), base, indices, rhs)
}

/// Scalar geometry is borrowed from the original receiver. Only the existing
/// assignment owner constructs a destination; no scalar ObjectArray mirror
/// or heap selector vector is needed to prepare the canonical plan.
pub fn assign_scalar_indices_in_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    indices: &[usize],
    rhs: Value,
) -> Result<Value, RuntimeError> {
    let shape = crate::object::indexing::class_name_from_base(&base)
        .and_then(|_| crate::indexing::value::shape_borrowed(&base))
        .ok_or_else(|| semantic_error("ObjectArrayIndexing", "value is not an object array"))?;
    if !matches!(indices.len(), 1 | 2) {
        return Err(semantic_error(
            "UnsupportedAssignmentRank",
            "Object assignment supports one or two scalar subscripts",
        ));
    }
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let _roots = match context {
        Some(context) => crate::execution::RootedValueLease::for_values_iterator_in_context(
            [&base, &rhs].into_iter(),
            context,
        )?,
        None => crate::execution::RootedValueLease::for_values([&base, &rhs].into_iter())
            .map_err(|error| error.into_runtime_error())?,
    };
    // The preceding rank gate is the authoritative bound for these slots.
    let selectors = [
        crate::indexing::selectors::SliceSelector::Scalar(indices[0]),
        crate::indexing::selectors::SliceSelector::Scalar(indices.get(1).copied().unwrap_or(1)),
    ];
    let selectors = &selectors[..indices.len()];
    let plan = match context {
        Some(context) => crate::indexing::plan::build_assignment_plan_in_context(
            context,
            selectors,
            indices.len(),
            shape.as_slice(),
        )?,
        None => crate::indexing::plan::build_assignment_plan(
            selectors,
            indices.len(),
            shape.as_slice(),
        )?,
    };
    owner.check()?;
    let result = assign_with_plan_in_context(context, base, &plan, rhs)?;
    owner.check()?;
    Ok(result)
}

pub fn delete_scalar_indices_in_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    indices: &[usize],
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let shape = crate::object::indexing::class_name_from_base(&base)
        .and_then(|_| crate::indexing::value::shape_borrowed(&base))
        .ok_or_else(|| semantic_error("ObjectArrayIndexing", "value is not an object array"))?;
    if !matches!(indices.len(), 1 | 2) {
        return Err(semantic_error(
            "UnsupportedAssignmentRank",
            "Object assignment supports one or two scalar subscripts",
        ));
    }
    if indices.len() == 2 {
        return Err(semantic_error(
            "UnsupportedDeletion",
            "Object-array deletion with two scalar subscripts is not supported",
        ));
    }
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let _roots = crate::execution::structure::replacement::lease(&base, context)?;
    let selectors = [crate::indexing::selectors::SliceSelector::Scalar(
        indices[0],
    )];
    let plan = match context {
        Some(context) => crate::indexing::plan::build_index_plan_in_context(
            context,
            &selectors,
            1,
            shape.as_slice(),
        )?,
        None => crate::indexing::plan::build_index_plan(&selectors, 1, shape.as_slice())?,
    };
    owner.check()?;
    let result = delete_with_plan_in_context(context, base, &plan, rhs)?;
    owner.check()?;
    Ok(result)
}

pub fn assign_with_plan(
    base: Value,
    plan: &IndexPlan,
    rhs: Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    if delete {
        return delete_with_plan_in_context(context.as_ref(), base, plan, &rhs);
    }
    assign_with_plan_in_context(context.as_ref(), base, plan, rhs)
}

pub fn assign_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    let array = as_array_in_context(context, base)?;
    growth::assign(context, &array, plan, &rhs)
}

pub fn delete_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let result = match base {
        Value::ObjectArray(array) => delete_array(context, array, plan, rhs),
        base @ (Value::Object(_) | Value::HandleObject(_)) => {
            validate_delete(plan, rhs, 1, 1)?;
            let _roots = crate::execution::structure::replacement::lease(&base, context)?;
            for &index in &plan.indices {
                owner.check()?;
                if index != 0 {
                    return Err(object_index_error());
                }
            }
            if plan.indices.is_empty() {
                owner.check()?;
                return Ok(base);
            }
            let class = crate::object::indexing::class_name_owner_from_base(&base)
                .expect("validated object receiver");
            let mut output =
                crate::execution::object_array::ObjectArrayConstruction::with_capacity(
                    context,
                    crate::execution::object_array::Class::Owner(class),
                    0,
                    &[1, 0],
                    1,
                    false,
                )?;
            match crate::object::protocol::selected_receiver_facts(&base)? {
                Some(facts) => output.push_provenance(facts)?,
                None => output.mark_unselected(),
            }
            let result = output.finish()?;
            owner.check()?;
            Ok(result.with_owned(|value| value))
        }
        _ => Err(semantic_error(
            "ObjectArrayIndexing",
            "value is not an object array",
        )),
    }?;
    owner.check()?;
    Ok(result)
}

fn validate_delete(
    plan: &IndexPlan,
    rhs: &Value,
    rows: usize,
    columns: usize,
) -> Result<(), RuntimeError> {
    if plan.dims != 1 {
        return Err(semantic_error(
            "UnsupportedDeletion",
            "Object-array selector deletion requires one linear subscript",
        ));
    }
    if !is_empty_delete_rhs(rhs) {
        return Err(semantic_error(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires an empty right-hand side",
        ));
    }
    if rows != 1 && columns != 1 {
        return Err(semantic_error(
            "UnsupportedDeletion",
            "Linear object-array deletion is only defined for vectors",
        ));
    }
    Ok(())
}

fn object_index_error() -> RuntimeError {
    semantic_error("IndexOutOfBounds", "Object array index is out of bounds")
}

fn delete_array(
    context: Option<&crate::context::RuntimeContext>,
    array: ObjectArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let rows = array.shape().first().copied().unwrap_or(1);
    let columns = array.shape().get(1).copied().unwrap_or(1);
    validate_delete(plan, rhs, rows, columns)?;
    let source = Value::ObjectArray(array.clone());
    let _roots = crate::execution::structure::replacement::lease(&source, context)?;
    let mut removed = crate::execution::object_array::indices::Indices::new(context, array.len())?;
    let mut count = 0usize;
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            removed.checkpoint()?;
        }
        let slot = removed
            .values_mut()
            .get_mut(index as usize)
            .ok_or_else(|| {
                semantic_error("IndexOutOfBounds", "Object array index is out of bounds")
            })?;
        if *slot == 0 {
            *slot = 1;
            count += 1;
        }
    }
    removed.checkpoint()?;
    let length = array.len() - count;
    let shape = if rows == 1 { [1, length] } else { [length, 1] };
    let selected = crate::execution::object_array::select(
        &array,
        removed
            .values()
            .iter()
            .enumerate()
            .filter_map(|(index, &removed)| (removed == 0).then_some(index)),
        &shape,
        context,
    )?;
    removed.checkpoint()?;
    Ok(selected.with_owned(|value| match value {
        Value::ObjectArray(array) if array.len() == 1 && array.shape() == [1, 1] => {
            array.data()[0].clone()
        }
        value => value,
    }))
}

fn as_array_in_context(
    context: Option<&crate::context::RuntimeContext>,
    value: Value,
) -> Result<ObjectArray, RuntimeError> {
    if let Value::ObjectArray(array) = value {
        return Ok(array);
    }
    let _roots = crate::execution::structure::replacement::lease(&value, context)?;
    let class = match &value {
        Value::Object(object) => object.class_name_owner(),
        Value::HandleObject(handle) => &handle.class_name,
        _ => {
            return Err(semantic_error(
                "ObjectArrayIndexing",
                "value is not an object array",
            ))
        }
    };
    let mut output = crate::execution::object_array::ObjectArrayConstruction::new(
        context,
        crate::execution::object_array::Class::Owner(class),
        1,
        &[1, 1],
        std::iter::empty(),
        false,
    )?;
    output.push(value)?;
    output.finish()?.with_owned(|value| match value {
        Value::ObjectArray(array) => Ok(array),
        _ => unreachable!("canonical array construction"),
    })
}

fn assignment_scalar<'a>(
    rhs: &'a Value,
    class_name: &ClassIdentity,
) -> Result<&'a Value, RuntimeError> {
    let rhs_class = match rhs {
        Value::Object(object) => Some(object.class_name()),
        Value::HandleObject(handle) => Some(handle.class_name.as_identity()),
        Value::ObjectArray(array) if array.len() == 1 => Some(array.class_name()),
        _ => None,
    };
    if rhs_class != Some(class_name) {
        return Err(semantic_error(
            "ObjectArrayClassMismatch",
            format!("Object-array assignment requires class '{class_name}'"),
        ));
    }
    match rhs {
        Value::Object(_) | Value::HandleObject(_) => Ok(rhs),
        Value::ObjectArray(array) => Ok(&array.data()[0]),
        _ => unreachable!("class check admitted only object values"),
    }
}

fn is_empty_delete_rhs(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.shape().iter().copied().product::<usize>() == 0)
        || matches!(value, Value::ObjectArray(array) if array.is_empty())
}

#[cfg(test)]
mod tests {
    use runmat_value::{ObjectInstance, Value};

    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn numeric_delete_rhs_uses_exact_empty_shape_cardinality() {
        let scalar = Value::Tensor(runmat_value::Tensor::new(vec![0.0], Vec::new()).unwrap());
        let empty = Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0]).unwrap());
        assert!(!is_empty_delete_rhs(&scalar));
        assert!(is_empty_delete_rhs(&empty));
    }

    fn object(name: &str) -> Value {
        let mut object = ObjectInstance::new("parallel.FevalFuture");
        object.replace_property_unbounded("Name".into(), Value::String(name.into()));
        Value::Object(object)
    }

    #[test]
    fn contiguous_linear_assignment_promotes_a_scalar_to_a_row_array() {
        let value = assign_scalar_indices(object("first"), &[2], object("second"), false)
            .expect("contiguous object assignment");
        let Value::ObjectArray(array) = value else {
            panic!("expected object array");
        };
        assert_eq!(array.shape(), &[1, 2]);
        assert_eq!(array.len(), 2);
    }

    #[test]
    fn assignment_rejects_class_drift_and_gap_filling_without_a_constructor() {
        let mut other = ObjectInstance::new("Other");
        other.replace_property_unbounded("Name".into(), Value::String("other".into()));
        assert!(assign_scalar_indices(object("first"), &[2], Value::Object(other), false).is_err());
        let error = assign_scalar_indices(object("first"), &[3], object("third"), false)
            .expect_err("gap requires class construction");
        assert_eq!(
            error.identifier(),
            Some("RunMat:ObjectArrayDefaultConstructionRequired")
        );
    }
}
