use super::{deletion, mex};
use crate::indexing::plan::IndexPlan;
use crate::RuntimeError;
use runmat_value::{StructArray, StructValue, Value};

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
    let context = crate::context::legacy::active();
    assign_with_plan_in_context(context.as_ref(), base, plan, rhs)
}

pub fn assign_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        return Ok(base);
    }
    structure_schema(&base)?;
    let replacements = assignment_values(&base, rhs, plan.indices.len())?;
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
    let rhs = match replacements {
        Replacement::Scalar(value) => Value::Struct(value),
        Replacement::Array(value) => Value::StructArray(value),
    };
    crate::aggregate::structure::operations::assign_in_context(context, &base, &rhs, plan).map_err(
        |error| {
            crate::aggregate::structure::operations::remap(error, |message| {
                mex("StructAssignment", message)
            })
        },
    )
}

fn structure_schema(value: &Value) -> Result<(), RuntimeError> {
    match value {
        Value::Struct(_) | Value::StructArray(_) => Ok(()),
        _ => Err(mex(
            "StructAssignmentTypeMismatch",
            "Structure-array assignment requires a structure target",
        )),
    }
}
fn assignment_values(
    schema: &Value,
    rhs: Value,
    count: usize,
) -> Result<Replacement, RuntimeError> {
    let valid = match (schema, &rhs) {
        (Value::Struct(left), Value::Struct(right)) => left.field_names().eq(right.field_names()),
        (Value::Struct(left), Value::StructArray(right)) => {
            left.field_names().eq(right.field_names())
        }
        (Value::StructArray(left), Value::Struct(right)) => {
            left.field_names().eq(right.field_names())
        }
        (Value::StructArray(left), Value::StructArray(right)) => {
            left.field_names().eq(right.field_names())
        }
        _ => {
            return Err(mex(
                "StructAssignmentTypeMismatch",
                "Structure-array assignment requires a structure value",
            ))
        }
    };
    match rhs {
        Value::Struct(value) => {
            if !valid {
                return Err(dissimilar_structure_error());
            }
            Ok(Replacement::Scalar(value))
        }
        Value::StructArray(value) if value.len() == count => {
            if !valid {
                return Err(dissimilar_structure_error());
            }
            Ok(Replacement::Array(value))
        }
        Value::StructArray(_) => Err(mex(
            "StructAssignmentSizeMismatch",
            "Structure-array assignment requires one structure per selected element",
        )),
        _ => unreachable!("validated structure RHS"),
    }
}

fn dissimilar_structure_error() -> RuntimeError {
    mex(
        "DissimilarStructureAssignment",
        "Subscripted assignment between dissimilar structures",
    )
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
