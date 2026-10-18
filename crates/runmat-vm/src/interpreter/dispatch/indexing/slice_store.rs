use super::{
    build_index_plan, build_slice_selectors, build_sparse_assignment_plan, idx_write_slice,
    logical_value_from_tensor, map_slice_plan_error, map_slice_shape_error, object_slice_selector,
    resolve_object_index_protocol, ObjectIndexOp, ObjectSubscript, ObjectSubscriptPath,
    RuntimeError, Value,
};
use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::indexing::plan::{build_indexed_store_plan, IndexedStorePlanMode};

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    _vars: &mut InterpreterVars,
    _function_registry: &crate::bytecode::FunctionRegistry,
    _pc: usize,
    runtime: &runmat_runtime::context::RuntimeContext,
    _sequence_register: &mut super::super::SequenceState,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::StoreSlice(dims, numeric_count, colon_mask, end_mask)
        | crate::bytecode::Instr::StoreSliceDelete(dims, numeric_count, colon_mask, end_mask) => {
            let delete = matches!(instr, crate::bytecode::Instr::StoreSliceDelete(_, _, _, _));
            let plan_mode = if delete {
                IndexedStorePlanMode::Bounded
            } else {
                IndexedStorePlanMode::Grow
            };
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let mut numeric: Vec<Value> = Vec::with_capacity(*numeric_count);
            for _ in 0..*numeric_count {
                numeric.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            numeric.reverse();
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _operand_roots = crate::call::root_values(numeric.iter().chain([&base, &rhs]))?;
            let base = match base {
                value @ (Value::Num(_) | Value::Int(_)) => Value::Tensor(
                    runmat_runtime::indexing::value::numeric_scalar_tensor(value)?,
                ),
                other => other,
            };
            match base {
                base @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
                    let resolution =
                        resolve_object_index_protocol(&base, ObjectIndexOp::Subsasgn, None)?;
                    if resolution.is_some() {
                        let path = ObjectSubscriptPath::single(ObjectSubscript::parentheses(
                            object_slice_selector(*dims, *colon_mask, *end_mask, &numeric)?,
                        ));
                        let Some(method) = resolution else {
                            unreachable!()
                        };
                        runmat_runtime::object::protocol::invoke_resolved_object_assignment_in_context(
                                runtime,
                                &method,
                                base,
                                path,
                                runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs)?
                                    .into_values(),
                            )
                            .await?
                            .with_owned(|value| stack.push(value));
                    } else {
                        let shape = runmat_runtime::indexing::object::shape(&base)
                            .expect("object-like values have an object-array shape");
                        let selectors =
                            build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &shape)
                                .await?;
                        let plan = build_index_plan(&selectors, *dims, &shape)?;
                        stack.push(runmat_runtime::indexing::object::assign_with_plan(
                            base, &plan, rhs, delete,
                        )?);
                    }
                }
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
                Value::Tensor(t) => {
                    let plan = build_indexed_store_plan(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        t.shape(),
                        plan_mode,
                    )
                    .await?;
                    stack.push(if delete {
                        idx_write_slice::delete_tensor_with_plan(t, &plan, &rhs)?
                    } else {
                        idx_write_slice::assign_tensor_with_plan(t, &plan, &rhs).await?
                    });
                }
                Value::LogicalArray(la) => {
                    let data: Vec<f64> = la
                        .data
                        .iter()
                        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                        .collect();
                    let tensor = runmat_value::Tensor::new(data, la.shape.clone())
                        .map_err(|e| map_slice_shape_error("slice assign", e))?;
                    let plan = build_indexed_store_plan(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        tensor.shape(),
                        plan_mode,
                    )
                    .await?;
                    let updated = if delete {
                        idx_write_slice::delete_tensor_with_plan(tensor, &plan, &rhs)?
                    } else {
                        idx_write_slice::assign_tensor_with_plan(tensor, &plan, &rhs).await?
                    };
                    stack.push(match updated {
                        Value::Tensor(t) => logical_value_from_tensor(t)?,
                        Value::Bool(_) | Value::LogicalArray(_) => updated,
                        Value::Num(n) => Value::Bool(n != 0.0),
                        other => other,
                    });
                }
                Value::GpuTensor(handle) => stack.push({
                    let plan = build_indexed_store_plan(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        handle.shape(),
                        plan_mode,
                    )
                    .await?;
                    if delete {
                        idx_write_slice::delete_gpu_slice_with_plan(&handle, &plan, &rhs).await?
                    } else {
                        idx_write_slice::assign_gpu_slice_with_plan(&handle, &plan, &rhs).await?
                    }
                }),
                Value::ComplexTensor(ct) => {
                    let plan = build_indexed_store_plan(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        ct.shape(),
                        plan_mode,
                    )
                    .await
                    .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    if delete {
                        stack.push(idx_write_slice::delete_complex_with_plan(ct, &plan, &rhs)?);
                        return Ok(true);
                    }
                    stack.push(
                        idx_write_slice::assign_complex_with_plan(ct, &plan, &rhs)
                            .await
                            .map_err(|e| map_slice_plan_error("slice assign", e))?,
                    );
                }
                Value::Cell(ca) => {
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &ca.shape)
                            .await
                            .map_err(|e| map_slice_plan_error("cell slice assign", e))?;
                    let plan = build_index_plan(&selectors, *dims, &ca.shape)
                        .map_err(|e| map_slice_plan_error("cell slice assign", e))?;
                    let selected: Vec<usize> =
                        plan.indices.iter().map(|idx| (*idx as usize) + 1).collect();
                    stack.push(
                        runmat_runtime::object::cell::assign_cell_paren_linear_indices_with_policy(
                            ca, &selected, &rhs, delete,
                        )?,
                    );
                }
                base @ (Value::Struct(_) | Value::StructArray(_)) => {
                    let shape = match &base {
                        Value::StructArray(array) => array.shape().to_vec(),
                        _ => vec![1, 1],
                    };
                    let plan = build_indexed_store_plan(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        &shape,
                        plan_mode,
                    )
                    .await?;
                    stack.push(runmat_runtime::indexing::structure::assign_with_plan(
                        base, &plan, rhs, delete,
                    )?);
                }
                Value::SparseTensor(sparse) => {
                    let shape = sparse.shape();
                    let selectors = if delete {
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &shape).await
                    } else {
                        runmat_runtime::indexing::selectors::build_sparse_assignment_selectors(
                            *dims,
                            *colon_mask,
                            *end_mask,
                            &numeric,
                            &shape,
                        )
                        .await
                    }
                    .map_err(|e| map_slice_plan_error("sparse slice assign", e))?;
                    let plan = if delete {
                        build_index_plan(&selectors, *dims, &shape)
                    } else {
                        build_sparse_assignment_plan(&selectors, *dims, &shape)
                    }
                    .map_err(|e| map_slice_plan_error("sparse slice assign", e))?;
                    stack.push(if delete {
                        idx_write_slice::delete_sparse_with_plan(sparse, &plan, &rhs)?
                    } else {
                        idx_write_slice::assign_sparse_with_plan(sparse, &plan, &rhs).await?
                    });
                }
                Value::StringArray(mut sa) => {
                    if delete {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedSliceDeletion",
                            "Slice deletion currently supports cell arrays only",
                        ));
                    }
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &sa.shape)
                            .await
                            .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    let plan = build_index_plan(&selectors, *dims, &sa.shape)
                        .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    if plan.indices.is_empty() {
                        idx_write_slice::validate_string_rhs_for_plan(
                            &rhs,
                            &plan.selection_lengths,
                        )
                        .map_err(|e| map_slice_plan_error("slice assign", e))?;
                        stack.push(Value::StringArray(sa));
                        return Ok(true);
                    }
                    let rhs_view =
                        idx_write_slice::build_string_rhs_view(&rhs, &plan.selection_lengths)
                            .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    idx_write_slice::scatter_string_with_plan(&mut sa, &plan, &rhs_view)
                        .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    stack.push(Value::StringArray(sa));
                }
                other => {
                    log::warn!(
                        "StoreSlice: unsupported base {:?} dims={} numeric={:?} rhs={:?}",
                        other,
                        dims,
                        numeric,
                        rhs
                    );
                    return Err(crate::interpreter::errors::mex(
                        "SliceNonTensor",
                        "Slicing assignment only supported on tensors or string arrays",
                    ));
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
