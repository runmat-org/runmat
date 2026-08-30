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
    array
        .select_linear(&indices, plan.output_shape.clone())
        .map(Value::ObjectArray)
        .map_err(|error| semantic_error("IndexOutOfBounds", error))
}

pub fn assign_scalar_indices(
    base: Value,
    indices: &[usize],
    rhs: Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let mut array = as_array(base)?;
    match indices {
        [index] => assign_linear(&mut array, *index, rhs, delete),
        [row, column] => assign_subscript(&mut array, *row, *column, rhs, delete),
        _ => Err(semantic_error(
            "UnsupportedAssignmentRank",
            "Object assignment supports one or two scalar subscripts",
        )),
    }
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
    let mut data = array.data().to_vec();
    for (index, value) in plan.indices.iter().zip(rhs_values) {
        let destination = data.get_mut(*index as usize).ok_or_else(|| {
            semantic_error("IndexOutOfBounds", "Object array index is out of bounds")
        })?;
        *destination = value;
    }
    finish_array(array.class_name(), data, array.shape().to_vec())
}

fn assign_linear(
    array: &mut ObjectArray,
    index: usize,
    rhs: Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if index == 0 {
        return Err(semantic_error(
            "IndexOutOfBounds",
            "Object array index is out of bounds",
        ));
    }
    if delete {
        return delete_linear(array.clone(), index, &rhs);
    }
    let value = assignment_scalar(&rhs, array.class_name())?;
    let mut data = array.data().to_vec();
    let mut shape = array.shape().to_vec();
    match index.cmp(&data.len()) {
        std::cmp::Ordering::Less | std::cmp::Ordering::Equal => data[index - 1] = value,
        std::cmp::Ordering::Greater if index == data.len() + 1 => {
            let rows = shape.first().copied().unwrap_or(1);
            let columns = shape.get(1).copied().unwrap_or(1);
            if rows == 1 {
                shape[1] = index;
            } else if columns == 1 {
                shape[0] = index;
            } else {
                return Err(semantic_error(
                    "IndexOutOfBounds",
                    "Linear object-array growth is only defined for vectors",
                ));
            }
            data.push(value);
        }
        std::cmp::Ordering::Greater => {
            return Err(semantic_error(
                "ObjectArrayDefaultConstructionRequired",
                "Object-array growth with gaps requires a default constructor",
            ));
        }
    }
    finish_array(array.class_name(), data, shape)
}

fn assign_subscript(
    array: &mut ObjectArray,
    row: usize,
    column: usize,
    rhs: Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if delete {
        return Err(semantic_error(
            "UnsupportedDeletion",
            "Object-array deletion with two scalar subscripts is not supported",
        ));
    }
    let rows = array.shape().first().copied().unwrap_or(1);
    let columns = array.shape().get(1).copied().unwrap_or(1);
    if row == 0 || column == 0 || row > rows || column > columns {
        return Err(semantic_error(
            "ObjectArrayDefaultConstructionRequired",
            "Object-array dimensional growth requires a default constructor",
        ));
    }
    let offset = (row - 1) + (column - 1) * rows;
    let value = assignment_scalar(&rhs, array.class_name())?;
    let mut data = array.data().to_vec();
    data[offset] = value;
    finish_array(array.class_name(), data, array.shape().to_vec())
}

fn delete_linear(array: ObjectArray, index: usize, rhs: &Value) -> Result<Value, RuntimeError> {
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
    if index > array.len() {
        return Err(semantic_error(
            "IndexOutOfBounds",
            "Object array index is out of bounds",
        ));
    }
    let class_name = array.class_name().clone();
    let mut data = array.into_data();
    data.remove(index - 1);
    let shape = if rows == 1 {
        vec![1, data.len()]
    } else {
        vec![data.len(), 1]
    };
    finish_array(&class_name, data, shape)
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
    finish_array(&class_name, data, shape)
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
    matches!(value, Value::Tensor(tensor) if tensor.shape.iter().copied().product::<usize>() == 0)
        || matches!(value, Value::ObjectArray(array) if array.is_empty())
}

#[cfg(test)]
mod tests {
    use runmat_value::{ObjectInstance, Value};

    use super::*;

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
