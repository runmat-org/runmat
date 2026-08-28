use runmat_value::{CellArray, LogicalArray, Value};

use crate::indexing::plan::{build_index_plan, IndexPlan};
use crate::indexing::read_slice;
use crate::indexing::selectors::{index_scalar_from_value, IndexScalar, SliceSelector};
use crate::indexing::write_slice;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

/// Whether a value has shared host indexing semantics suitable for deterministic
/// sliced-result assembly. Device and foreign handles require a placement-aware
/// assembler and therefore remain on the serial correctness path.
pub fn supports_sliced_assembly(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::SparseTensor(_)
            | Value::LogicalArray(_)
            | Value::Cell(_)
    )
}

pub async fn merge_slice(
    destination: Value,
    worker_value: &Value,
    access: runmat_types::ParallelSliceAccess,
    iteration: &Value,
    offset_value: Option<&Value>,
) -> Result<Value, RuntimeError> {
    if !same_assembly_family(&destination, worker_value) {
        return Err(semantic_error(
            "ParallelSliceType",
            "worker sliced output changed its runtime storage family",
        ));
    }
    let shape = value_shape(&destination).ok_or_else(|| {
        semantic_error(
            "ParallelSliceType",
            "sliced output does not support deterministic host assembly",
        )
    })?;
    if value_shape(worker_value).as_deref() != Some(shape.as_slice()) {
        return Err(semantic_error(
            "ParallelSliceShape",
            "worker sliced output changed its source array shape",
        ));
    }
    let index = resolved_slice_index(access.offset, iteration, offset_value).await?;
    let (selectors, selector_dimensions) = match access.axis {
        runmat_types::ParallelSliceAxis::Linear => (vec![SliceSelector::Scalar(index)], 1),
        runmat_types::ParallelSliceAxis::Dimension(dimension) => {
            let dimension = usize::try_from(dimension)
                .ok()
                .and_then(|dimension| dimension.checked_sub(1))
                .filter(|dimension| *dimension < shape.len())
                .ok_or_else(|| {
                    semantic_error(
                        "ParallelSliceDimension",
                        "parallel slice dimension is outside the output rank",
                    )
                })?;
            let mut selectors = vec![SliceSelector::Colon; shape.len()];
            selectors[dimension] = SliceSelector::Scalar(index);
            (selectors, shape.len())
        }
    };
    let plan = build_index_plan(&selectors, selector_dimensions, &shape)?;
    let slice = read_with_plan(worker_value, &plan)?;
    assign_with_plan(destination, &plan, &slice).await
}

async fn resolved_slice_index(
    offset: runmat_types::ParallelSliceOffset,
    iteration: &Value,
    offset_value: Option<&Value>,
) -> Result<usize, RuntimeError> {
    let iteration = index_scalar_from_value(iteration)
        .await?
        .map(index_scalar_i128)
        .ok_or_else(|| {
            semantic_error(
                "ParallelIterationIndex",
                "parallel sliced output requires an integer loop value",
            )
        })?;
    let resolved = match offset {
        runmat_types::ParallelSliceOffset::None => iteration,
        runmat_types::ParallelSliceOffset::Add(operand) => iteration
            .checked_add(resolve_offset_operand(operand, offset_value).await?)
            .ok_or_else(|| semantic_error("ParallelIterationIndex", "slice index overflowed"))?,
        runmat_types::ParallelSliceOffset::Subtract(operand) => iteration
            .checked_sub(resolve_offset_operand(operand, offset_value).await?)
            .ok_or_else(|| semantic_error("ParallelIterationIndex", "slice index overflowed"))?,
    };
    if resolved < 1 {
        return Err(semantic_error(
            "ParallelIterationIndex",
            "parallel sliced output requires a positive integer index",
        ));
    }
    usize::try_from(resolved).map_err(|_| {
        semantic_error(
            "ParallelIterationIndex",
            "parallel slice index exceeds the host index range",
        )
    })
}

async fn resolve_offset_operand(
    operand: runmat_types::ParallelSliceOffsetOperand,
    offset_value: Option<&Value>,
) -> Result<i128, RuntimeError> {
    match operand {
        runmat_types::ParallelSliceOffsetOperand::Constant(value) => Ok(match value {
            runmat_types::ParallelIndexConstant::Signed(value) => i128::from(value),
            runmat_types::ParallelIndexConstant::Unsigned(value) => i128::from(value),
        }),
        runmat_types::ParallelSliceOffsetOperand::Broadcast(_) => {
            let value = offset_value.ok_or_else(|| {
                semantic_error(
                    "ParallelSliceOffset",
                    "parallel slice is missing its broadcast offset value",
                )
            })?;
            index_scalar_from_value(value)
                .await?
                .map(index_scalar_i128)
                .ok_or_else(|| {
                    semantic_error(
                        "ParallelSliceOffset",
                        "parallel slice offset must be an integer scalar",
                    )
                })
        }
    }
}

fn index_scalar_i128(value: IndexScalar) -> i128 {
    match value {
        IndexScalar::Signed(value) => i128::from(value),
        IndexScalar::Unsigned(value) => i128::from(value),
    }
}

fn value_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Tensor(value) => Some(value.shape.clone()),
        Value::ComplexTensor(value) => Some(value.shape.clone()),
        Value::SparseTensor(value) => Some(value.shape()),
        Value::LogicalArray(value) => Some(value.shape.clone()),
        Value::Cell(value) => Some(value.shape.clone()),
        _ => None,
    }
}

fn same_assembly_family(left: &Value, right: &Value) -> bool {
    matches!(
        (left, right),
        (Value::Tensor(_), Value::Tensor(_))
            | (Value::ComplexTensor(_), Value::ComplexTensor(_))
            | (Value::SparseTensor(_), Value::SparseTensor(_))
            | (Value::LogicalArray(_), Value::LogicalArray(_))
            | (Value::Cell(_), Value::Cell(_))
    )
}

fn read_with_plan(value: &Value, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    match value {
        Value::Tensor(value) => read_slice::read_tensor_slice_from_plan(value, plan),
        Value::ComplexTensor(value) => read_slice::read_complex_slice_from_plan(value, plan),
        Value::SparseTensor(value) => read_slice::read_sparse_slice_from_plan(value, plan),
        Value::LogicalArray(value) => read_logical_with_plan(value, plan),
        Value::Cell(value) => read_cell_with_plan(value, plan),
        _ => Err(semantic_error(
            "ParallelSliceType",
            "sliced output does not support deterministic host reads",
        )),
    }
}

async fn assign_with_plan(
    destination: Value,
    plan: &IndexPlan,
    source: &Value,
) -> Result<Value, RuntimeError> {
    match destination {
        Value::Tensor(value) => write_slice::assign_tensor_with_plan(value, plan, source).await,
        Value::ComplexTensor(value) => {
            write_slice::assign_complex_with_plan(value, plan, source).await
        }
        Value::SparseTensor(value) => {
            write_slice::assign_sparse_with_plan(value, plan, source).await
        }
        Value::LogicalArray(value) => assign_logical_with_plan(value, plan, source),
        Value::Cell(value) => assign_cell_with_plan(value, plan, source),
        _ => Err(semantic_error(
            "ParallelSliceType",
            "sliced output does not support deterministic host writes",
        )),
    }
}

fn read_logical_with_plan(value: &LogicalArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let selected = plan
        .indices
        .iter()
        .map(|index| {
            value.data.get(*index as usize).copied().ok_or_else(|| {
                semantic_error("ParallelSliceShape", "logical slice index is out of bounds")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let [selected] = selected.as_slice() {
        return Ok(Value::Bool(*selected != 0));
    }
    LogicalArray::new(selected, plan.output_shape.clone())
        .map(Value::LogicalArray)
        .map_err(|error| semantic_error("ParallelSliceShape", error))
}

fn assign_logical_with_plan(
    mut destination: LogicalArray,
    plan: &IndexPlan,
    source: &Value,
) -> Result<Value, RuntimeError> {
    let values = match source {
        Value::Bool(value) => vec![u8::from(*value); plan.indices.len()],
        Value::LogicalArray(value) if value.data.len() == plan.indices.len() => value.data.to_vec(),
        _ => {
            return Err(semantic_error(
                "ParallelSliceType",
                "logical sliced output returned an incompatible value",
            ))
        }
    };
    for (index, value) in plan.indices.iter().zip(values) {
        let Some(destination) = destination.data.get_mut(*index as usize) else {
            return Err(semantic_error(
                "ParallelSliceShape",
                "logical slice index is out of bounds",
            ));
        };
        *destination = value;
    }
    Ok(Value::LogicalArray(destination))
}

fn read_cell_with_plan(value: &CellArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let values = plan
        .indices
        .iter()
        .map(|index| {
            cell_storage_index(*index as usize, &value.shape)
                .and_then(|index| value.data.get(index))
                .cloned()
                .ok_or_else(|| {
                    semantic_error("ParallelSliceShape", "cell slice index is out of bounds")
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    CellArray::from_column_major(values, plan.output_shape.clone())
        .map(Value::Cell)
        .map_err(|error| semantic_error("ParallelSliceShape", error))
}

fn assign_cell_with_plan(
    mut destination: CellArray,
    plan: &IndexPlan,
    source: &Value,
) -> Result<Value, RuntimeError> {
    let Value::Cell(source) = source else {
        return Err(semantic_error(
            "ParallelSliceType",
            "cell sliced output returned a non-cell value",
        ));
    };
    let source = source.to_column_major();
    if source.len() != plan.indices.len() {
        return Err(semantic_error(
            "ParallelSliceShape",
            "cell sliced output has an incompatible shape",
        ));
    }
    for (index, value) in plan.indices.iter().zip(source) {
        let storage_index =
            cell_storage_index(*index as usize, &destination.shape).ok_or_else(|| {
                semantic_error("ParallelSliceShape", "cell slice index is out of bounds")
            })?;
        let Some(destination) = destination.data.get_mut(storage_index) else {
            return Err(semantic_error(
                "ParallelSliceShape",
                "cell slice index is out of bounds",
            ));
        };
        *destination = value;
    }
    Ok(Value::Cell(destination))
}

fn cell_storage_index(column_major: usize, shape: &[usize]) -> Option<usize> {
    let rows = shape.first().copied()?;
    let cols = shape.get(1).copied().unwrap_or(1);
    if rows == 0 || cols == 0 {
        return None;
    }
    let page_len = rows.checked_mul(cols)?;
    let total_len = shape
        .iter()
        .try_fold(1usize, |length, dimension| length.checked_mul(*dimension))?;
    if column_major >= total_len {
        return None;
    }
    let page = column_major / page_len;
    let within_page = column_major % page_len;
    let row = within_page % rows;
    let column = within_page / rows;
    page.checked_mul(page_len)?
        .checked_add(row.checked_mul(cols)?)?
        .checked_add(column)
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor};

    use super::*;

    #[test]
    fn sliced_assembly_preserves_native_integer_storage() {
        let destination = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![0; 6]), vec![2, 3]).expect("destination"),
        );
        let worker = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![1, u64::MAX, 3, u64::MAX - 1, 5, 6]),
                vec![2, 3],
            )
            .expect("worker"),
        );
        let assembled = block_on(merge_slice(
            destination,
            &worker,
            runmat_types::ParallelSliceAccess::dimension(2),
            &Value::Num(2.0),
            None,
        ))
        .expect("assemble second column");
        let Value::Tensor(assembled) = assembled else {
            panic!("expected integer tensor");
        };
        assert!(matches!(
            assembled.integer_storage(),
            Some(IntegerStorage::U64(values))
                if values == &[0, 0, 3, u64::MAX - 1, 0, 0]
        ));
    }

    #[test]
    fn sliced_assembly_distinguishes_linear_from_dimensional_indexing() {
        let destination = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![0; 4]), vec![1, 4])
                .expect("row destination"),
        );
        let worker = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![0, 7, 0, 0]), vec![1, 4])
                .expect("row worker"),
        );
        let assembled = block_on(merge_slice(
            destination,
            &worker,
            runmat_types::ParallelSliceAccess::linear(),
            &Value::Num(2.0),
            None,
        ))
        .expect("assemble second linear element");
        assert!(matches!(
            assembled,
            Value::Tensor(value)
                if matches!(value.integer_storage(), Some(IntegerStorage::U16(values)) if values == &[0, 7, 0, 0])
        ));
    }

    #[test]
    fn sliced_assembly_applies_typed_constant_and_broadcast_offsets() {
        let destination = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![0; 4]), vec![1, 4])
                .expect("row destination"),
        );
        let worker = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![0, 0, 9, 0]), vec![1, 4])
                .expect("row worker"),
        );
        let constant_access = runmat_types::ParallelSliceAccess {
            axis: runmat_types::ParallelSliceAxis::Linear,
            offset: runmat_types::ParallelSliceOffset::Add(
                runmat_types::ParallelSliceOffsetOperand::Constant(
                    runmat_types::ParallelIndexConstant::Unsigned(1),
                ),
            ),
        };
        let assembled = block_on(merge_slice(
            destination.clone(),
            &worker,
            constant_access,
            &Value::Num(2.0),
            None,
        ))
        .expect("assemble affine constant slice");
        assert!(matches!(
            assembled,
            Value::Tensor(value)
                if matches!(value.integer_storage(), Some(IntegerStorage::U16(values)) if values == &[0, 0, 9, 0])
        ));

        let broadcast_access = runmat_types::ParallelSliceAccess {
            axis: runmat_types::ParallelSliceAxis::Linear,
            offset: runmat_types::ParallelSliceOffset::Add(
                runmat_types::ParallelSliceOffsetOperand::Broadcast(runmat_types::RegionValueId {
                    function: runmat_types::ProgramFunctionId(0),
                    local: 1,
                }),
            ),
        };
        block_on(merge_slice(
            destination,
            &worker,
            broadcast_access,
            &Value::Num(2.0),
            Some(&Value::Int(runmat_value::IntValue::I32(1))),
        ))
        .expect("assemble affine broadcast slice");
    }

    #[test]
    fn sliced_assembly_uses_column_major_indices_for_logical_and_nd_cells() {
        let logical = Value::LogicalArray(
            LogicalArray::new(vec![0; 4], vec![2, 2]).expect("logical destination"),
        );
        let worker_logical = Value::LogicalArray(
            LogicalArray::new(vec![0, 0, 1, 1], vec![2, 2]).expect("logical worker"),
        );
        let logical = block_on(merge_slice(
            logical,
            &worker_logical,
            runmat_types::ParallelSliceAccess::dimension(2),
            &Value::Num(2.0),
            None,
        ))
        .expect("assemble logical column");
        assert!(matches!(
            logical,
            Value::LogicalArray(value) if value.data.as_slice() == &[0, 0, 1, 1]
        ));

        let cells = (0..8).map(|_| Value::Num(0.0)).collect::<Vec<_>>();
        let destination = Value::Cell(
            CellArray::from_column_major(cells, vec![2, 2, 2]).expect("cell destination"),
        );
        let worker_values = (1..=8).map(|value| Value::Num(f64::from(value))).collect();
        let worker = Value::Cell(
            CellArray::from_column_major(worker_values, vec![2, 2, 2]).expect("cell worker"),
        );
        let assembled = block_on(merge_slice(
            destination,
            &worker,
            runmat_types::ParallelSliceAccess::dimension(3),
            &Value::Num(2.0),
            None,
        ))
        .expect("assemble second cell page");
        let Value::Cell(assembled) = assembled else {
            panic!("expected cell array");
        };
        assert_eq!(
            assembled.to_column_major(),
            vec![
                Value::Num(0.0),
                Value::Num(0.0),
                Value::Num(0.0),
                Value::Num(0.0),
                Value::Num(5.0),
                Value::Num(6.0),
                Value::Num(7.0),
                Value::Num(8.0),
            ]
        );
    }
}
