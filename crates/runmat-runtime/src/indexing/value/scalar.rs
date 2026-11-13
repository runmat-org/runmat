use runmat_value::{IntValue, IntegerStorage, Tensor, Value};

use crate::indexing::plan::IndexPlan;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub(super) fn read_cell(
    context: Option<&crate::context::RuntimeContext>,
    value: runmat_value::CellArray,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let indices = one_based_indices(context, plan)?;
    crate::object::cell::gather_cell_paren_linear_indices_in_context(
        context,
        &value,
        &indices,
        &plan.output_shape,
    )
}

pub(super) fn read_scalar_struct(
    context: Option<&crate::context::RuntimeContext>,
    value: runmat_value::StructValue,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    if plan.indices.iter().any(|index| *index != 0) {
        return Err(index_out_of_bounds());
    }
    crate::aggregate::structure::operations::replicate_scalar_in_context(
        context,
        &value,
        &plan.output_shape,
    )
    .map_err(|error| {
        crate::aggregate::structure::operations::remap(error, |message| {
            semantic_error("StructIndexing", message)
        })
    })
}

pub(super) fn assign_cell(
    context: Option<&crate::context::RuntimeContext>,
    value: runmat_value::CellArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    crate::object::cell::assign_cell_paren_linear_indices_with_policy_in_context(
        context,
        value,
        &one_based_indices(context, plan)?,
        rhs,
        false,
    )
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

pub(super) fn checked_shape_len(shape: &[usize]) -> Result<usize, RuntimeError> {
    shape.iter().try_fold(1usize, |len, extent| {
        len.checked_mul(*extent).ok_or_else(|| {
            semantic_error(
                "ShapeOverflow",
                "indexed result shape exceeds platform limits",
            )
        })
    })
}

fn one_based_indices(
    context: Option<&crate::context::RuntimeContext>,
    plan: &IndexPlan,
) -> Result<runmat_value::ExactArrayStorage<usize>, RuntimeError> {
    crate::context::host_allocation::initialize_host_array(context, plan.indices.len(), |index| {
        usize::try_from(plan.indices[index])
            .ok()
            .and_then(|i| i.checked_add(1))
            .ok_or_else(index_out_of_bounds)
    })
}

pub(super) fn index_out_of_bounds() -> RuntimeError {
    semantic_error("IndexOutOfBounds", "index is out of bounds")
}
