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
    let array = as_array(value.clone())?;
    if let [index] = plan.indices.as_slice() {
        return array.get_linear(*index as usize).cloned().ok_or_else(|| {
            semantic_error("IndexOutOfBounds", "Object array index is out of bounds")
        });
    }
    let indices = plan
        .indices
        .iter()
        .map(|index| *index as usize)
        .collect::<Vec<_>>();
    Ok(Value::ObjectArray(select_linear_with_provenance(
        &array,
        &indices,
        plan.output_shape.clone(),
    )?))
}

pub(crate) fn select_linear_with_provenance(
    array: &ObjectArray,
    indices: &[usize],
    shape: Vec<usize>,
) -> Result<ObjectArray, RuntimeError> {
    let mut selected = array
        .select_linear(indices, shape)
        .map_err(|error| semantic_error("IndexOutOfBounds", error))?;
    if selected.is_empty() {
        let (provenance, unselected) = empty_result_provenance(array)?;
        selected = selected
            .with_selected_provenance(provenance)
            .map_err(|error| semantic_error("SelectedObjectReceiverInvalid", error))?;
        if unselected {
            selected = selected.with_unselected_provenance();
        }
    }
    Ok(selected)
}

pub fn assign_scalar_indices(
    base: Value,
    indices: &[usize],
    rhs: Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let array = as_array(base)?;
    if !matches!(indices.len(), 1 | 2) {
        return Err(semantic_error(
            "UnsupportedAssignmentRank",
            "Object assignment supports one or two scalar subscripts",
        ));
    }
    if delete && indices.len() == 2 {
        return Err(semantic_error(
            "UnsupportedDeletion",
            "Object-array deletion with two scalar subscripts is not supported",
        ));
    }
    let selectors = indices
        .iter()
        .copied()
        .map(crate::indexing::selectors::SliceSelector::Scalar)
        .collect::<Vec<_>>();
    let plan = if delete {
        crate::indexing::plan::build_index_plan(&selectors, indices.len(), array.shape())?
    } else {
        crate::indexing::plan::build_assignment_plan(&selectors, indices.len(), array.shape())?
    };
    assign_with_plan(Value::ObjectArray(array), &plan, rhs, delete)
}

pub fn assign_with_plan(
    base: Value,
    plan: &IndexPlan,
    rhs: Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let array = as_array(base)?;
    if delete {
        return delete_with_plan(array, plan, &rhs);
    }
    let rhs_values = assignment_values(&rhs, array.class_name(), plan.indices.len())?;
    let data = growth::assign(&array, plan, rhs_values)?;
    let value = finish_array(array.class_name(), data, plan.base_shape.clone())?;
    if let Value::ObjectArray(result) = value {
        if result.is_empty() {
            let (provenance, unselected) = empty_result_provenance(&array)?;
            let result = result
                .with_selected_provenance(provenance)
                .map_err(|error| semantic_error("ObjectArrayAssignment", error))?;
            return Ok(Value::ObjectArray(if unselected {
                result.with_unselected_provenance()
            } else {
                result
            }));
        }
        return Ok(Value::ObjectArray(result));
    }
    Ok(value)
}

fn delete_with_plan(
    array: ObjectArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
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
    let rows = array.shape().first().copied().unwrap_or(1);
    let columns = array.shape().get(1).copied().unwrap_or(1);
    if rows != 1 && columns != 1 {
        return Err(semantic_error(
            "UnsupportedDeletion",
            "Linear object-array deletion is only defined for vectors",
        ));
    }
    let mut removed = plan
        .indices
        .iter()
        .map(|index| *index as usize)
        .collect::<Vec<_>>();
    removed.sort_unstable();
    removed.dedup();
    let class_name = array.class_name().clone();
    let empty_provenance = if removed.len() == array.len() {
        Some(empty_result_provenance(&array)?)
    } else {
        None
    };
    let mut data = array.into_data();
    for index in removed.into_iter().rev() {
        if index >= data.len() {
            return Err(semantic_error(
                "IndexOutOfBounds",
                "Object array index is out of bounds",
            ));
        }
        data.remove(index);
    }
    let shape = if rows == 1 {
        vec![1, data.len()]
    } else {
        vec![data.len(), 1]
    };
    let value = finish_array(&class_name, data, shape)?;
    if let Value::ObjectArray(array) = value {
        if array.is_empty() {
            let (selected_provenance, unselected_provenance) =
                empty_provenance.expect("all object-array elements were removed");
            let array = array
                .with_selected_provenance(selected_provenance)
                .map_err(|error| semantic_error("ObjectArrayAssignment", error))?;
            return Ok(Value::ObjectArray(if unselected_provenance {
                array.with_unselected_provenance()
            } else {
                array
            }));
        }
        return Ok(Value::ObjectArray(array));
    }
    Ok(value)
}

pub(crate) fn empty_result_provenance(
    array: &ObjectArray,
) -> Result<
    (
        Vec<std::rc::Rc<runmat_value::SelectedObjectClassFacts>>,
        bool,
    ),
    RuntimeError,
> {
    // A handle receiver keeps its selected facts in the GC target. Capture
    // those exact facts before an empty operation discards every element.
    let mut provenance = array.selected_provenance().to_vec();
    let mut unselected = array.has_unselected_provenance();
    for element in array.data() {
        match crate::object::protocol::selected_receiver_facts(element)? {
            Some(facts) => provenance.push(facts),
            None => unselected = true,
        }
    }
    Ok((provenance, unselected))
}

fn finish_array(
    class_name: &ClassIdentity,
    mut data: Vec<Value>,
    shape: Vec<usize>,
) -> Result<Value, RuntimeError> {
    if data.len() == 1 && shape.as_slice() == [1, 1] {
        return Ok(data.remove(0));
    }
    ObjectArray::new(class_name.clone(), data, shape)
        .map(Value::ObjectArray)
        .map_err(|error| semantic_error("ObjectArrayAssignment", error))
}

fn as_array(value: Value) -> Result<ObjectArray, RuntimeError> {
    match value {
        Value::Object(object) => {
            let class_name = object.class_name.clone();
            ObjectArray::new(class_name, vec![Value::Object(object)], vec![1, 1])
        }
        Value::HandleObject(handle) => {
            let class_name = handle.class_name.clone();
            ObjectArray::new(class_name, vec![Value::HandleObject(handle)], vec![1, 1])
        }
        Value::ObjectArray(array) => Ok(array),
        _ => Err("value is not an object array".into()),
    }
    .map_err(|error| semantic_error("ObjectArrayIndexing", error))
}

fn assignment_values(
    rhs: &Value,
    class_name: &ClassIdentity,
    count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    match rhs {
        Value::ObjectArray(array) if array.class_name() == class_name && array.len() == count => {
            Ok(array.data().to_vec())
        }
        _ => Ok(vec![assignment_scalar(rhs, class_name)?; count]),
    }
}

fn assignment_scalar(rhs: &Value, class_name: &ClassIdentity) -> Result<Value, RuntimeError> {
    let rhs_class = match rhs {
        Value::Object(object) => Some(&object.class_name),
        Value::HandleObject(handle) => Some(&handle.class_name),
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
        Value::Object(_) | Value::HandleObject(_) => Ok(rhs.clone()),
        Value::ObjectArray(array) => Ok(array.data()[0].clone()),
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
        object
            .properties
            .insert("Name".into(), Value::String(name.into()));
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
        other
            .properties
            .insert("Name".into(), Value::String("other".into()));
        assert!(assign_scalar_indices(object("first"), &[2], Value::Object(other), false).is_err());
        let error = assign_scalar_indices(object("first"), &[3], object("third"), false)
            .expect_err("gap requires class construction");
        assert_eq!(
            error.identifier(),
            Some("RunMat:ObjectArrayDefaultConstructionRequired")
        );
    }
}
