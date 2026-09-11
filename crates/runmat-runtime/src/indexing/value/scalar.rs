use runmat_value::{CharArray, IntValue, IntegerStorage, LogicalArray, StringArray, Tensor, Value};

use crate::indexing::plan::IndexPlan;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub(super) fn read_logical(value: &LogicalArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let data = plan
        .indices
        .iter()
        .map(|index| {
            usize::try_from(*index)
                .ok()
                .and_then(|i| value.data.get(i).copied())
                .ok_or_else(index_out_of_bounds)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let [value] = data.as_slice() {
        return Ok(Value::Bool(*value != 0));
    }
    LogicalArray::new(data, plan.output_shape.clone())
        .map(Value::LogicalArray)
        .map_err(shape_error)
}

pub(super) fn read_cell(
    value: runmat_value::CellArray,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let indices = one_based_indices(plan)?;
    crate::object::cell::gather_cell_paren_linear_indices(&value, &indices, &plan.output_shape)
}

pub(super) fn read_char(value: &CharArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let source = value.to_column_major();
    let data = plan
        .indices
        .iter()
        .map(|index| {
            usize::try_from(*index)
                .ok()
                .and_then(|i| source.get(i).copied())
                .ok_or_else(index_out_of_bounds)
        })
        .collect::<Result<Vec<_>, _>>()?;
    CharArray::from_column_major(data, plan.output_shape.clone())
        .map(Value::CharArray)
        .map_err(shape_error)
}

pub(super) fn read_scalar_struct(
    value: runmat_value::StructValue,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    if plan.indices.iter().any(|index| *index != 0) {
        return Err(index_out_of_bounds());
    }
    runmat_value::StructArray::replicate_scalar(value, plan.output_shape.clone())
        .map_err(|error| semantic_error("StructIndexing", error))
}

pub(super) fn assign_logical(
    mut value: LogicalArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let target_len = checked_shape_len(&plan.base_shape)?;
    if value.data.len() < target_len {
        value.data.resize(target_len, 0);
        value.shape = plan.base_shape.clone();
    }
    let replacements = match rhs {
        Value::Bool(value) => vec![u8::from(*value); plan.indices.len()],
        Value::LogicalArray(value) if value.data.len() == plan.indices.len() => value.data.to_vec(),
        Value::Num(value) => vec![u8::from(*value != 0.0); plan.indices.len()],
        Value::Int(value) => vec![u8::from(value.to_f64() != 0.0); plan.indices.len()],
        Value::Tensor(value) if value.len() == 1 => {
            let scalar = value.numeric_value_at(0).ok_or_else(|| {
                semantic_error(
                    "AssignmentTypeMismatch",
                    "logical assignment requires a numeric or logical value",
                )
            })?;
            vec![u8::from(scalar.materialize_f64() != 0.0); plan.indices.len()]
        }
        _ => return Err(semantic_error(
            "AssignmentTypeMismatch",
            "logical assignment requires a numeric or logical scalar, or a matching logical array",
        )),
    };
    for (index, replacement) in plan.indices.iter().zip(replacements) {
        let index = usize::try_from(*index).map_err(|_| index_out_of_bounds())?;
        *value.data.get_mut(index).ok_or_else(index_out_of_bounds)? = replacement;
    }
    Ok(Value::LogicalArray(value))
}

pub(super) fn assign_cell(
    value: runmat_value::CellArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    crate::object::cell::assign_cell_paren_linear_indices_with_policy(
        value,
        &one_based_indices(plan)?,
        rhs,
        false,
    )
}

pub(super) fn assign_char(
    value: CharArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let replacements = match rhs {
        Value::CharArray(value) => value.to_column_major(),
        Value::String(value) => value.chars().collect(),
        _ => {
            return Err(semantic_error(
                "AssignmentTypeMismatch",
                "character assignment requires character or text data",
            ))
        }
    };
    let replacements = if replacements.len() == 1 {
        vec![replacements[0]; plan.indices.len()]
    } else if replacements.len() == plan.indices.len() {
        replacements
    } else {
        return Err(semantic_error(
            "AssignmentShapeMismatch",
            "character assignment size does not match the indexed selection",
        ));
    };
    let mut data = value.to_column_major();
    data.resize(checked_shape_len(&plan.base_shape)?, ' ');
    for (index, replacement) in plan.indices.iter().zip(replacements) {
        let index = usize::try_from(*index).map_err(|_| index_out_of_bounds())?;
        *data.get_mut(index).ok_or_else(index_out_of_bounds)? = replacement;
    }
    CharArray::from_column_major(data, plan.base_shape.clone())
        .map(Value::CharArray)
        .map_err(shape_error)
}

pub(super) fn integer_scalar_tensor(value: IntValue) -> Result<Tensor, RuntimeError> {
    let storage = match value {
        IntValue::I8(v) => IntegerStorage::I8(vec![v]),
        IntValue::I16(v) => IntegerStorage::I16(vec![v]),
        IntValue::I32(v) => IntegerStorage::I32(vec![v]),
        IntValue::I64(v) => IntegerStorage::I64(vec![v]),
        IntValue::U8(v) => IntegerStorage::U8(vec![v]),
        IntValue::U16(v) => IntegerStorage::U16(vec![v]),
        IntValue::U32(v) => IntegerStorage::U32(vec![v]),
        IntValue::U64(v) => IntegerStorage::U64(vec![v]),
    };
    Tensor::new_integer(storage, vec![1, 1]).map_err(shape_error)
}

pub(super) fn shape_error(error: impl std::fmt::Display) -> RuntimeError {
    semantic_error("ShapeMismatch", error.to_string())
}

pub(super) fn grow_string_array(
    value: &mut StringArray,
    shape: &[usize],
) -> Result<(), RuntimeError> {
    let target_len = checked_shape_len(shape)?;
    if value.data.len() < target_len {
        value.data.resize(target_len, String::new());
        value.shape = shape.to_vec();
    }
    Ok(())
}

fn checked_shape_len(shape: &[usize]) -> Result<usize, RuntimeError> {
    shape.iter().try_fold(1usize, |len, extent| {
        len.checked_mul(*extent).ok_or_else(|| {
            semantic_error(
                "ShapeOverflow",
                "indexed result shape exceeds platform limits",
            )
        })
    })
}

fn one_based_indices(plan: &IndexPlan) -> Result<Vec<usize>, RuntimeError> {
    plan.indices
        .iter()
        .map(|index| {
            usize::try_from(*index)
                .ok()
                .and_then(|i| i.checked_add(1))
                .ok_or_else(index_out_of_bounds)
        })
        .collect()
}

fn index_out_of_bounds() -> RuntimeError {
    semantic_error("IndexOutOfBounds", "index is out of bounds")
}
