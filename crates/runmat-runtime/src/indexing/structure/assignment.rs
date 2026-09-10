use super::{deletion, mex, zero_based_indices};
use crate::indexing::plan::IndexPlan;
use crate::RuntimeError;
use runmat_value::{StructArray, StructValue, Tensor, Value};

enum Replacement {
    Scalar(StructValue),
    Array(StructArray),
}

pub fn assign_with_plan(
    base: Value,
    plan: &IndexPlan,
    rhs: Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if delete {
        if !is_empty_delete_rhs(&rhs) {
            return Err(mex(
                "InvalidStructDeletionRhs",
                "Structure-array deletion requires an empty right-hand side",
            ));
        }
        return deletion::delete_with_plan(base, plan);
    }
    if plan.indices.is_empty() {
        return Ok(base);
    }
    let schema = structure_schema(&base)?;
    let replacements = assignment_values(&schema, rhs, plan.indices.len())?;
    let target_len = checked_len(&plan.base_shape)?;
    if target_len == 1 {
        if plan.indices.as_slice() != [0] {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        return match replacements {
            Replacement::Scalar(value) => Ok(Value::Struct(value)),
            Replacement::Array(_) => Err(mex(
                "StructAssignmentSizeMismatch",
                "Structure-array assignment requires one structure per selected element",
            )),
        };
    }
    let fill = empty_structure(&schema);
    let array = match base {
        Value::Struct(structure) => {
            StructArray::grow_scalar(structure, plan.base_shape.clone(), &fill)
        }
        Value::StructArray(array) if array.shape() == plan.base_shape => Ok(array),
        Value::StructArray(array) => array.grow(plan.base_shape.clone(), &fill),
        _ => unreachable!("structure_schema admitted only structure values"),
    }
    .map_err(|error| mex("StructAssignment", error))?;
    let indices = zero_based_indices(plan);
    let array = match replacements {
        Replacement::Scalar(value) => array.replace_linear_scalar(&indices, value),
        Replacement::Array(values) => array.replace_linear_array(&indices, values),
    }
    .map_err(|error| mex("StructAssignment", error))?;
    Ok(Value::StructArray(array))
}

fn structure_schema(value: &Value) -> Result<Vec<String>, RuntimeError> {
    match value {
        Value::Struct(structure) => Ok(structure.field_names().cloned().collect()),
        Value::StructArray(array) => Ok(array.field_names().cloned().collect()),
        _ => Err(mex(
            "StructAssignmentTypeMismatch",
            "Structure-array assignment requires a structure target",
        )),
    }
}

fn assignment_values(
    schema: &[String],
    rhs: Value,
    count: usize,
) -> Result<Replacement, RuntimeError> {
    match rhs {
        Value::Struct(value) => {
            validate_schema(schema, &value)?;
            Ok(Replacement::Scalar(value))
        }
        Value::StructArray(value) if value.len() == count => {
            if !value.field_names().eq(schema.iter()) {
                return Err(dissimilar_structure_error());
            }
            Ok(Replacement::Array(value))
        }
        Value::StructArray(_) => Err(mex(
            "StructAssignmentSizeMismatch",
            "Structure-array assignment requires one structure per selected element",
        )),
        _ => Err(mex(
            "StructAssignmentTypeMismatch",
            "Structure-array assignment requires a structure value",
        )),
    }
}

fn validate_schema(schema: &[String], value: &StructValue) -> Result<(), RuntimeError> {
    if value.field_names().eq(schema.iter()) {
        Ok(())
    } else {
        Err(dissimilar_structure_error())
    }
}

fn dissimilar_structure_error() -> RuntimeError {
    mex(
        "DissimilarStructureAssignment",
        "Subscripted assignment between dissimilar structures",
    )
}

fn empty_structure(schema: &[String]) -> StructValue {
    let mut structure = StructValue::new();
    for name in schema {
        structure.insert(
            name.clone(),
            Value::Tensor(
                Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor shape is valid"),
            ),
        );
    }
    structure
}

fn is_empty_delete_rhs(value: &Value) -> bool {
    match value {
        Value::Tensor(value) => value.is_empty(),
        Value::ComplexTensor(value) => value.is_empty(),
        _ => false,
    }
}

fn checked_len(shape: &[usize]) -> Result<usize, RuntimeError> {
    shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))
    })
}
