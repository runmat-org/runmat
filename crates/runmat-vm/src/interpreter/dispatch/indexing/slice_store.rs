use super::{
    idx_write_slice, map_slice_plan_error, object_slice_selector, resolve_object_index_protocol,
    ObjectIndexOp, ObjectSubscript, ObjectSubscriptPath, RuntimeError, Value,
};
use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::indexing::plan::IndexedStorePlanMode;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut crate::interpreter::stack::InterpreterStack,
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
            let (numeric_owner, _operand_roots) =
                super::extract_selector_operands(stack, runtime, *numeric_count, true)?;
            let numeric = numeric_owner.as_slice();
            let rhs = stack
                .pop()
                .expect("validated assignment window contains RHS");
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = match base {
                value @ (Value::Num(_) | Value::Int(_)) => Value::Tensor(
                    runmat_runtime::indexing::value::numeric_scalar_tensor_in_context(
                        Some(runtime),
                        value,
                    )?,
                ),
                other => other,
            };
            match base {
                base @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
                    let resolution =
                        resolve_object_index_protocol(&base, ObjectIndexOp::Subsasgn, None)?;
                    if resolution.is_some() {
                        let path = ObjectSubscriptPath::single_in_context(
                            Some(runtime),
                            ObjectSubscript::parentheses(object_slice_selector(
                                runtime,
                                *dims,
                                *colon_mask,
                                *end_mask,
                                numeric,
                            )?),
                        )?;
                        let Some(method) = resolution else {
                            unreachable!()
                        };
                        runmat_runtime::object::protocol::invoke_resolved_object_assignment_sequence_in_context(
                                runtime,
                                &method,
                                base,
                                path,
                                runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs)?,
                            )
                            .await?
                            .with_owned(|value| stack.try_push(value))?;
                    } else {
                        let shape = runmat_runtime::indexing::value::shape_borrowed(&base)
                            .expect("object-like values have an object-array shape");
                        let selectors =
                            runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                                runtime,
                                *dims,
                                *colon_mask,
                                *end_mask,
                                numeric,
                                shape.as_slice(),
                            )
                            .await?;
                        let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                            runtime,
                            &selectors,
                            *dims,
                            shape.as_slice(),
                        )?;
                        stack.try_push(if delete {
                            runmat_runtime::indexing::object::delete_with_plan_in_context(
                                Some(runtime),
                                base,
                                &plan,
                                &rhs,
                            )?
                        } else {
                            runmat_runtime::indexing::object::assign_with_plan_in_context(
                                Some(runtime),
                                base,
                                &plan,
                                rhs,
                            )?
                        })?;
                    }
                }
                Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle(_)
                | Value::Closure(_)
                | Value::ScopedFunctionHandle(_) => {
                    return Err(crate::interpreter::errors::mex(
                        "UnsupportedFunctionHandleSelector",
                        "Function handle call does not support assignment selector syntax",
                    ));
                }
                Value::Tensor(t) => {
                    let plan = runmat_runtime::indexing::plan::build_indexed_store_plan_in_context(
                        runtime,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        numeric,
                        t.shape(),
                        plan_mode,
                    )
                    .await?;
                    stack.try_push(if delete {
                        idx_write_slice::delete_tensor_with_plan(t, &plan, &rhs)?
                    } else {
                        idx_write_slice::assign_tensor_with_plan_in_context(
                            Some(runtime),
                            t,
                            &plan,
                            &rhs,
                        )
                        .await?
                    })?;
                }
                Value::LogicalArray(value) => {
                    let plan = runmat_runtime::indexing::plan::build_indexed_store_plan_in_context(
                        runtime,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        numeric,
                        &value.shape,
                        plan_mode,
                    )
                    .await?;
                    let updated = if delete {
                        idx_write_slice::delete_logical_with_plan(Some(runtime), value, &plan, &rhs)
                            .await?
                    } else {
                        idx_write_slice::assign_logical_with_plan_in_context(
                            Some(runtime),
                            value,
                            &plan,
                            &rhs,
                        )
                        .await?
                    };
                    stack.try_push(match updated {
                        Value::LogicalArray(value)
                            if value.len() == 1
                                && runmat_runtime::builtins::common::shape::is_scalar_shape(
                                    &value.shape,
                                ) =>
                        {
                            Value::Bool(value.data[0] != 0)
                        }
                        other => other,
                    })?;
                }
                Value::GpuTensor(handle) => stack.try_push({
                    let plan = runmat_runtime::indexing::plan::build_indexed_store_plan_in_context(
                        runtime,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        numeric,
                        handle.shape(),
                        plan_mode,
                    )
                    .await?;
                    if delete {
                        idx_write_slice::delete_gpu_slice_with_plan_in_context(
                            Some(runtime),
                            &handle,
                            &plan,
                            &rhs,
                        )
                        .await?
                    } else {
                        idx_write_slice::assign_gpu_slice_with_plan_in_context(
                            Some(runtime),
                            &handle,
                            &plan,
                            &rhs,
                        )
                        .await?
                    }
                })?,
                Value::ComplexTensor(ct) => {
                    let plan = runmat_runtime::indexing::plan::build_indexed_store_plan_in_context(
                        runtime,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        numeric,
                        ct.shape(),
                        plan_mode,
                    )
                    .await
                    .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    if delete {
                        stack.try_push(idx_write_slice::delete_complex_with_plan(
                            ct, &plan, &rhs,
                        )?)?;
                        return Ok(true);
                    }
                    stack.try_push(
                        idx_write_slice::assign_complex_with_plan_in_context(
                            Some(runtime),
                            ct,
                            &plan,
                            &rhs,
                        )
                        .await
                        .map_err(|e| map_slice_plan_error("slice assign", e))?,
                    )?;
                }
                Value::Cell(ca) => {
                    let selectors =
                        runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                            runtime,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                            &ca.shape,
                        )
                        .await
                        .map_err(|e| map_slice_plan_error("cell slice assign", e))?;
                    let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                        runtime, &selectors, *dims, &ca.shape,
                    )
                    .map_err(|e| map_slice_plan_error("cell slice assign", e))?;
                    if delete {
                        stack.try_push(
                            runmat_runtime::object::cell::delete_cell_with_plan_in_context(
                                Some(runtime),
                                ca,
                                &plan,
                                &rhs,
                            )?,
                        )?;
                        return Ok(true);
                    }
                    let selected = runmat_runtime::context::host_allocation::initialize_host_array(
                        Some(runtime),
                        plan.indices.len(),
                        |ordinal| Ok(plan.indices[ordinal] as usize + 1),
                    )?;
                    stack.try_push(
                        runmat_runtime::object::cell::assign_cell_paren_linear_indices_with_policy_in_context(
                            Some(runtime), ca, &selected, &rhs, false,
                        )?,
                    )?;
                }
                base @ (Value::Struct(_) | Value::StructArray(_)) => {
                    let shape = runmat_runtime::indexing::value::shape_borrowed(&base)
                        .expect("structure stored geometry");
                    let plan = runmat_runtime::indexing::plan::build_indexed_store_plan_in_context(
                        runtime,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        numeric,
                        shape.as_slice(),
                        plan_mode,
                    )
                    .await?;
                    stack.try_push(if delete {
                        runmat_runtime::indexing::structure::delete_with_plan_in_context(
                            Some(runtime),
                            base,
                            &plan,
                            &rhs,
                        )?
                    } else {
                        runmat_runtime::indexing::structure::assign_with_plan_in_context(
                            Some(runtime),
                            base,
                            &plan,
                            rhs,
                        )?
                    })?;
                }
                Value::SparseTensor(sparse) => {
                    let shape = sparse.shape();
                    let selectors = if delete {
                        runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                            runtime,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                            &shape,
                        )
                        .await
                    } else {
                        runmat_runtime::indexing::selectors::build_assignment_selectors_in_context(
                            runtime,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                            &shape,
                        )
                        .await
                    }
                    .map_err(|e| map_slice_plan_error("sparse slice assign", e))?;
                    let plan = if delete {
                        runmat_runtime::indexing::plan::build_index_plan_in_context(
                            runtime, &selectors, *dims, &shape,
                        )
                    } else {
                        runmat_runtime::indexing::plan::build_assignment_plan_in_context(
                            runtime, &selectors, *dims, &shape,
                        )
                    }
                    .map_err(|e| map_slice_plan_error("sparse slice assign", e))?;
                    stack.try_push(if delete {
                        idx_write_slice::delete_sparse_with_plan_in_context(
                            Some(runtime),
                            sparse,
                            &plan,
                            &rhs,
                        )?
                    } else {
                        idx_write_slice::assign_sparse_with_plan_in_context(
                            Some(runtime),
                            sparse,
                            &plan,
                            &rhs,
                        )
                        .await?
                    })?;
                }
                Value::StringArray(sa) => {
                    if delete {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedSliceDeletion",
                            "Slice deletion currently supports cell arrays only",
                        ));
                    }
                    let selectors =
                        runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                            runtime,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                            &sa.shape,
                        )
                        .await
                        .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                        runtime, &selectors, *dims, &sa.shape,
                    )
                    .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    if plan.indices.is_empty() {
                        idx_write_slice::validate_string_rhs_for_plan(
                            &rhs,
                            &plan.selection_lengths,
                        )
                        .map_err(|e| map_slice_plan_error("slice assign", e))?;
                        stack.try_push(Value::StringArray(sa))?;
                        return Ok(true);
                    }
                    stack.try_push(
                        idx_write_slice::assign_string_with_plan_in_context(
                            Some(runtime),
                            sa,
                            &plan,
                            &rhs,
                        )
                        .map_err(|e| map_slice_plan_error("slice assign", e))?,
                    )?;
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
