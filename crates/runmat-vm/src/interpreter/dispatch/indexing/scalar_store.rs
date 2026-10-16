use super::{
    assign_object_scalar_indices, build_assignment_plan, build_index_plan, idx_write_linear,
    idx_write_slice, index_scalar_from_value, map_slice_shape_error, IntValue, RuntimeError,
    SliceSelector, Value,
};
use runmat_runtime::indexing::plan::{build_indexed_store_plan, IndexedStorePlanMode};
use runmat_runtime::indexing::value as index_value;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    _vars: &mut [Value],
    _function_registry: &crate::bytecode::FunctionRegistry,
    pc: usize,
    runtime: &runmat_runtime::context::RuntimeContext,
    _sequence_register: &mut super::super::SequenceState,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::StoreIndex(num_indices)
        | crate::bytecode::Instr::StoreIndexDelete(num_indices) => {
            let delete = matches!(instr, crate::bytecode::Instr::StoreIndexDelete(_));
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _rhs_roots = crate::call::root_values([&rhs])?;
            let mut indices: Vec<usize> = Vec::new();
            for _ in 0..*num_indices {
                let value = stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?;
                let idx_val = index_scalar_from_value(&value).await?.ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "ScalarIndexRequired",
                        "StoreIndex requires scalar indices; use StoreSlice for vector, range, or logical indices",
                    )
                })?;
                if idx_val.is_below_one() {
                    return Err(crate::interpreter::errors::mex(
                        "IndexOutOfBounds",
                        "Index out of bounds",
                    ));
                }
                let index = idx_val.positive_usize().ok_or_else(|| {
                    crate::interpreter::errors::mex("IndexOutOfBounds", "Index out of bounds")
                })?;
                indices.push(index);
            }
            indices.reverse();
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            match base {
                Value::Foreign(reference) => {
                    if delete {
                        return Err(crate::interpreter::errors::mex(
                            "ForeignIndexDelete",
                            "Indexed deletion is not supported for foreign objects",
                        ));
                    }
                    let indices = indices
                        .into_iter()
                        .map(|index| Value::Int(IntValue::U64(index as u64)))
                        .collect();
                    runmat_runtime::foreign::assign_foreign_resource_index(reference, indices, rhs)
                        .await?
                        .with_owned(|value| stack.push(value));
                }
                Value::Object(obj) => {
                    assign_object_scalar_indices(runtime, Value::Object(obj), indices, rhs, delete)
                        .await?
                        .with_owned(|value| stack.push(value))
                }
                Value::HandleObject(handle) => assign_object_scalar_indices(
                    runtime,
                    Value::HandleObject(handle),
                    indices,
                    rhs,
                    delete,
                )
                .await?
                .with_owned(|value| stack.push(value)),
                Value::ObjectArray(array) => assign_object_scalar_indices(
                    runtime,
                    Value::ObjectArray(array),
                    indices,
                    rhs,
                    delete,
                )
                .await?
                .with_owned(|value| stack.push(value)),
                Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle(_)
                | Value::Closure(_) => {
                    return Err(crate::interpreter::errors::mex(
                        "UnsupportedFunctionHandleSelector",
                        "Function handle call does not support assignment selector syntax",
                    ));
                }
                base @ (Value::Tensor(_)
                | Value::ComplexTensor(_)
                | Value::Num(_)
                | Value::Int(_)
                | Value::GpuTensor(_)) => {
                    if indices.is_empty() || indices.len() > 2 {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedAssignmentRank",
                            "Only 1D/2D scalar assignment supported",
                        ));
                    }
                    if delete && indices.len() != 1 {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedDeletion",
                            "Indexed deletion is only supported for linear vector indices",
                        ));
                    }
                    let shape = index_value::shape(&base).ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "IndexAssignmentUnsupportedBase",
                            "Numeric indexed assignment requires a numeric base",
                        )
                    })?;
                    let numeric = indices
                        .iter()
                        .copied()
                        .map(|index| {
                            u64::try_from(index)
                                .map(|index| Value::Int(IntValue::U64(index)))
                                .map_err(|_| {
                                    crate::interpreter::errors::mex(
                                        "IndexOutOfBounds",
                                        "Index out of bounds",
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let mode = if delete {
                        IndexedStorePlanMode::Bounded
                    } else {
                        IndexedStorePlanMode::Grow
                    };
                    let plan =
                        build_indexed_store_plan(indices.len(), 0, 0, &numeric, &shape, mode)
                            .await?;
                    let updated = if delete {
                        match base {
                            Value::Tensor(t) => {
                                idx_write_slice::delete_tensor_with_plan(t, &plan, &rhs)?
                            }
                            Value::ComplexTensor(t) => {
                                idx_write_slice::delete_complex_with_plan(t, &plan, &rhs)?
                            }
                            value @ (Value::Num(_) | Value::Int(_)) => {
                                let tensor = index_value::numeric_scalar_tensor(value)?;
                                idx_write_slice::delete_tensor_with_plan(tensor, &plan, &rhs)?
                            }
                            Value::GpuTensor(handle) => {
                                idx_write_slice::delete_gpu_slice_with_plan(&handle, &plan, &rhs)
                                    .await?
                            }
                            _ => {
                                return Err(crate::interpreter::errors::mex(
                                    "IndexAssignmentUnsupportedBase",
                                    "Numeric indexed assignment requires a numeric base",
                                ));
                            }
                        }
                    } else {
                        index_value::assign_with_plan(base, &plan, rhs).await?
                    };
                    stack.push(updated);
                }
                Value::Bool(b) => {
                    let scalar =
                        runmat_value::Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                            .map_err(|e| map_slice_shape_error("scalar index assign", e))?;
                    stack.push(
                        idx_write_linear::assign_tensor_scalar(scalar, &indices, &rhs, delete)
                            .await?,
                    );
                }
                Value::StringArray(mut sa) => {
                    if delete {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedSliceDeletion",
                            "Slice deletion currently supports cell arrays only",
                        ));
                    }
                    let selectors: Vec<SliceSelector> =
                        indices.iter().copied().map(SliceSelector::Scalar).collect();
                    let plan = build_index_plan(&selectors, indices.len(), &sa.shape)?;
                    if !plan.indices.is_empty() {
                        let rhs_view =
                            idx_write_slice::build_string_rhs_view(&rhs, &plan.selection_lengths)?;
                        idx_write_slice::scatter_string_with_plan(&mut sa, &plan, &rhs_view)?;
                    }
                    stack.push(Value::StringArray(sa));
                }
                Value::Cell(ca) => {
                    stack.push(runmat_runtime::object::cell::assign_cell_paren_with_policy(
                        ca, &indices, &rhs, delete,
                    )?)
                }
                Value::Struct(structure) => {
                    let selectors = indices
                        .iter()
                        .copied()
                        .map(SliceSelector::Scalar)
                        .collect::<Vec<_>>();
                    let plan = if delete {
                        build_index_plan(&selectors, indices.len(), &[1, 1])?
                    } else {
                        build_assignment_plan(&selectors, indices.len(), &[1, 1])?
                    };
                    stack.push(runmat_runtime::indexing::structure::assign_with_plan(
                        Value::Struct(structure),
                        &plan,
                        rhs,
                        delete,
                    )?);
                }
                Value::StructArray(array) => {
                    let selectors = indices
                        .iter()
                        .copied()
                        .map(SliceSelector::Scalar)
                        .collect::<Vec<_>>();
                    let plan = if delete {
                        build_index_plan(&selectors, indices.len(), array.shape())?
                    } else {
                        build_assignment_plan(&selectors, indices.len(), array.shape())?
                    };
                    stack.push(runmat_runtime::indexing::structure::assign_with_plan(
                        Value::StructArray(array),
                        &plan,
                        rhs,
                        delete,
                    )?);
                }
                Value::SparseTensor(sparse) => stack.push(
                    idx_write_linear::assign_sparse_scalar(sparse, &indices, &rhs, delete).await?,
                ),
                _ => {
                    if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                        let kind = |v: &Value| match v {
                            Value::Object(_) => "Object",
                            Value::Struct(_) => "Struct",
                            Value::Tensor(_) => "Tensor",
                            Value::GpuTensor(_) => "GpuTensor",
                            Value::Num(_) => "Num",
                            Value::Int(_) => "Int",
                            _ => "Other",
                        };
                        log::debug!(
                            "[vm] StoreIndex default branch pc={} base_kind={} rhs_kind={} indices={:?}",
                            pc,
                            kind(&base),
                            kind(&rhs),
                            indices
                        );
                    }
                    return Err(crate::interpreter::errors::mex(
                        "IndexAssignmentUnsupportedBase",
                        "Index assignment only for tensors",
                    ));
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
