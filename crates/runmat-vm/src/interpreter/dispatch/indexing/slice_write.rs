use super::*;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    vars: &mut [Value],
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::StoreSlice(dims, numeric_count, colon_mask, end_mask)
        | crate::bytecode::Instr::StoreSliceDelete(dims, numeric_count, colon_mask, end_mask) => {
            let delete = matches!(instr, crate::bytecode::Instr::StoreSliceDelete(_, _, _, _));
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
            match base {
                Value::Object(obj) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::Object(obj.clone()),
                        ObjectIndexOp::Subsasgn,
                    ) {
                        return Err(err);
                    }
                    let descriptor = ObjectIndexDescriptor::subsasgn_paren_from_slice(
                        Value::Object(obj),
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        rhs,
                    )?;
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::HandleObject(handle) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::HandleObject(handle.clone()),
                        ObjectIndexOp::Subsasgn,
                    ) {
                        return Err(err);
                    }
                    let descriptor = ObjectIndexDescriptor::subsasgn_paren_from_slice(
                        Value::HandleObject(handle),
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        rhs,
                    )?;
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle { .. }
                | Value::Closure(_) => {
                    return Err(crate::interpreter::errors::mex(
                        "UnsupportedFunctionHandleSelector",
                        "Function handle call does not support assignment selector syntax",
                    ));
                }
                Value::Tensor(t) => {
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &t.shape)
                            .await?;
                    let plan = build_index_plan(&selectors, *dims, &t.shape)?;
                    stack.push(if delete {
                        idx_write_slice::delete_tensor_with_plan(t, &plan, &rhs)?
                    } else {
                        idx_write_slice::assign_tensor_with_plan(t, &plan, &rhs).await?
                    });
                }
                Value::Int(value) => {
                    let tensor = integer_scalar_tensor(value.clone())?;
                    let selectors = build_slice_selectors(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        &tensor.shape,
                    )
                    .await?;
                    let plan = build_index_plan(&selectors, *dims, &tensor.shape)?;
                    stack.push(assign_integer_scalar_with_plan(value, &plan, &rhs, delete).await?);
                }
                Value::LogicalArray(la) => {
                    let data: Vec<f64> = la
                        .data
                        .iter()
                        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                        .collect();
                    let tensor = runmat_value::Tensor::new(data, la.shape.clone())
                        .map_err(|e| map_slice_shape_error("slice assign", e))?;
                    let selectors = build_slice_selectors(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        &tensor.shape,
                    )
                    .await?;
                    let plan = build_index_plan(&selectors, *dims, &tensor.shape)?;
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
                    let selectors = build_slice_selectors(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        &handle.shape,
                    )
                    .await?;
                    let plan = build_index_plan(&selectors, *dims, &handle.shape)?;
                    if delete {
                        idx_write_slice::delete_gpu_slice_with_plan(&handle, &plan, &rhs).await?
                    } else {
                        idx_write_slice::assign_gpu_slice_with_plan(&handle, &plan, &rhs).await?
                    }
                }),
                Value::ComplexTensor(ct) => {
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &ct.shape)
                            .await
                            .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    let plan = build_index_plan(&selectors, *dims, &ct.shape)
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
        crate::bytecode::Instr::StoreSliceExpr {
            dims,
            numeric_count,
            colon_mask,
            end_mask,
            range_dims,
            range_has_step,
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
        }
        | crate::bytecode::Instr::StoreSliceExprDelete {
            dims,
            numeric_count,
            colon_mask,
            end_mask,
            range_dims,
            range_has_step,
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
        } => {
            validate_expr_range_step_metadata(range_dims, range_has_step)?;
            let delete = matches!(instr, crate::bytecode::Instr::StoreSliceExprDelete { .. });
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let mut range_params: Vec<(f64, f64)> = Vec::with_capacity(range_dims.len());
            for i in (0..range_dims.len()).rev() {
                let has = range_has_step[i];
                let step = if has {
                    let v = stack.pop().ok_or(crate::interpreter::errors::mex(
                        "StackUnderflow",
                        "stack underflow",
                    ))?;
                    range_selector_scalar_to_f64(&v).await?
                } else {
                    1.0
                };
                let st = stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?;
                let st = range_selector_scalar_to_f64(&st).await?;
                range_params.push((st, step));
            }
            range_params.reverse();
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
            if !end_numeric_exprs.is_empty() {
                numeric = match &base {
                    Value::GpuTensor(handle) => {
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(
                                *dims,
                                *colon_mask,
                                *end_mask,
                                range_dims,
                                &handle.shape,
                            ),
                            end_numeric_exprs,
                            vars,
                        )
                        .await?
                    }
                    Value::Tensor(t) => {
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(*dims, *colon_mask, *end_mask, range_dims, &t.shape),
                            end_numeric_exprs,
                            vars,
                        )
                        .await?
                    }
                    Value::ComplexTensor(t) => {
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(*dims, *colon_mask, *end_mask, range_dims, &t.shape),
                            end_numeric_exprs,
                            vars,
                        )
                        .await?
                    }
                    Value::SparseTensor(s) => {
                        let shape = s.shape();
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(*dims, *colon_mask, *end_mask, range_dims, &shape),
                            end_numeric_exprs,
                            vars,
                        )
                        .await?
                    }
                    Value::StringArray(sa) => {
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(*dims, *colon_mask, *end_mask, range_dims, &sa.shape),
                            end_numeric_exprs,
                            vars,
                        )
                        .await?
                    }
                    Value::Cell(ca) => {
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(*dims, *colon_mask, *end_mask, range_dims, &ca.shape),
                            end_numeric_exprs,
                            vars,
                        )
                        .await?
                    }
                    _ => numeric,
                };
            }
            match base {
                Value::ComplexTensor(t) => {
                    let vm_plan = build_expr_slice_plan(
                        ExprPlanSpec {
                            dims: *dims,
                            colon_mask: *colon_mask,
                            end_mask: *end_mask,
                            range_dims,
                            range_params: &range_params,
                            range_start_exprs,
                            range_step_exprs,
                            range_end_exprs,
                            numeric: &numeric,
                            shape: &t.shape,
                        },
                        vars,
                    )
                    .await?;
                    if delete {
                        stack.push(idx_write_slice::delete_complex_with_plan(
                            t, &vm_plan, &rhs,
                        )?);
                        return Ok(true);
                    }
                    stack.push(idx_write_slice::assign_complex_with_plan(t, &vm_plan, &rhs).await?);
                }
                Value::Tensor(t) => {
                    let vm_plan = build_expr_slice_plan(
                        ExprPlanSpec {
                            dims: *dims,
                            colon_mask: *colon_mask,
                            end_mask: *end_mask,
                            range_dims,
                            range_params: &range_params,
                            range_start_exprs,
                            range_step_exprs,
                            range_end_exprs,
                            numeric: &numeric,
                            shape: &t.shape,
                        },
                        vars,
                    )
                    .await?;
                    stack.push(if delete {
                        idx_write_slice::delete_tensor_with_plan(t, &vm_plan, &rhs)?
                    } else {
                        idx_write_slice::assign_tensor_with_plan(t, &vm_plan, &rhs).await?
                    });
                }
                Value::Int(value) => {
                    let tensor = integer_scalar_tensor(value.clone())?;
                    let vm_plan = build_expr_slice_plan(
                        ExprPlanSpec {
                            dims: *dims,
                            colon_mask: *colon_mask,
                            end_mask: *end_mask,
                            range_dims,
                            range_params: &range_params,
                            range_start_exprs,
                            range_step_exprs,
                            range_end_exprs,
                            numeric: &numeric,
                            shape: &tensor.shape,
                        },
                        vars,
                    )
                    .await?;
                    stack.push(
                        assign_integer_scalar_with_plan(value, &vm_plan, &rhs, delete).await?,
                    );
                }
                Value::GpuTensor(h) => {
                    let vm_plan = build_expr_slice_plan(
                        ExprPlanSpec {
                            dims: *dims,
                            colon_mask: *colon_mask,
                            end_mask: *end_mask,
                            range_dims,
                            range_params: &range_params,
                            range_start_exprs,
                            range_step_exprs,
                            range_end_exprs,
                            numeric: &numeric,
                            shape: &h.shape,
                        },
                        vars,
                    )
                    .await?;
                    let updated = if delete {
                        idx_write_slice::delete_gpu_slice_with_plan(&h, &vm_plan, &rhs).await?
                    } else {
                        idx_write_slice::assign_gpu_slice_with_plan(&h, &vm_plan, &rhs).await?
                    };
                    stack.push(updated);
                }
                Value::SparseTensor(sparse) => {
                    let shape = sparse.shape();
                    let read_only_vars: &[Value] = vars;
                    let spec = ExprPlanSpec {
                        dims: *dims,
                        colon_mask: *colon_mask,
                        end_mask: *end_mask,
                        range_dims,
                        range_params: &range_params,
                        range_start_exprs,
                        range_step_exprs,
                        range_end_exprs,
                        numeric: &numeric,
                        shape: &shape,
                    };
                    let vm_plan = if delete {
                        build_expr_slice_plan(spec, vars).await?
                    } else {
                        build_expr_sparse_assignment_plan(spec, |dim_len, expr| {
                            let expr = expr.clone();
                            async move {
                                resolve_range_end_value(dim_len, &expr, read_only_vars).await
                            }
                        })
                        .await?
                    };
                    stack.push(if delete {
                        idx_write_slice::delete_sparse_with_plan(sparse, &vm_plan, &rhs)?
                    } else {
                        idx_write_slice::assign_sparse_with_plan(sparse, &vm_plan, &rhs).await?
                    });
                }
                Value::Object(obj) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::Object(obj.clone()),
                        ObjectIndexOp::Subsasgn,
                    ) {
                        return Err(err);
                    }
                    let descriptor = ObjectIndexDescriptor::subsasgn_paren_from_expr_slice(
                        Value::Object(obj),
                        ObjectParenExprSelectorSpec {
                            dims: *dims,
                            colon_mask: *colon_mask,
                            end_mask: *end_mask,
                            range_dims,
                            range_params: &range_params,
                            range_start_exprs,
                            range_step_exprs,
                            range_end_exprs,
                            end_numeric_exprs,
                            numeric: &numeric,
                        },
                        rhs,
                    )?;
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::HandleObject(handle) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::HandleObject(handle.clone()),
                        ObjectIndexOp::Subsasgn,
                    ) {
                        return Err(err);
                    }
                    let descriptor = ObjectIndexDescriptor::subsasgn_paren_from_expr_slice(
                        Value::HandleObject(handle),
                        ObjectParenExprSelectorSpec {
                            dims: *dims,
                            colon_mask: *colon_mask,
                            end_mask: *end_mask,
                            range_dims,
                            range_params: &range_params,
                            range_start_exprs,
                            range_step_exprs,
                            range_end_exprs,
                            end_numeric_exprs,
                            numeric: &numeric,
                        },
                        rhs,
                    )?;
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::StringArray(mut sa) => {
                    if delete {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedSliceDeletion",
                            "Slice deletion currently supports cell arrays only",
                        ));
                    }
                    let vm_plan = build_expr_slice_plan(
                        ExprPlanSpec {
                            dims: *dims,
                            colon_mask: *colon_mask,
                            end_mask: *end_mask,
                            range_dims,
                            range_params: &range_params,
                            range_start_exprs,
                            range_step_exprs,
                            range_end_exprs,
                            numeric: &numeric,
                            shape: &sa.shape,
                        },
                        vars,
                    )
                    .await?;
                    if !vm_plan.indices.is_empty() {
                        let rhs_view = idx_write_slice::build_string_rhs_view(
                            &rhs,
                            &vm_plan.selection_lengths,
                        )?;
                        idx_write_slice::scatter_string_with_plan(&mut sa, &vm_plan, &rhs_view)?;
                    }
                    stack.push(Value::StringArray(sa));
                }
                Value::Cell(ca) => {
                    let vm_plan = build_expr_slice_plan(
                        ExprPlanSpec {
                            dims: *dims,
                            colon_mask: *colon_mask,
                            end_mask: *end_mask,
                            range_dims,
                            range_params: &range_params,
                            range_start_exprs,
                            range_step_exprs,
                            range_end_exprs,
                            numeric: &numeric,
                            shape: &ca.shape,
                        },
                        vars,
                    )
                    .await?;
                    let selected: Vec<usize> = vm_plan
                        .indices
                        .iter()
                        .map(|idx| (*idx as usize) + 1)
                        .collect();
                    stack.push(
                        runmat_runtime::object::cell::assign_cell_paren_linear_indices_with_policy(
                            ca, &selected, &rhs, delete,
                        )?,
                    );
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "SliceNonTensor",
                        "StoreSliceExpr only supports tensors, cells, and string arrays currently",
                    ));
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
