use super::*;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    vars: &mut [Value],
    function_registry: &crate::bytecode::FunctionRegistry,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::IndexSlice(dims, numeric_count, colon_mask, end_mask) => {
            let mut numeric: Vec<Value> = Vec::with_capacity(*numeric_count);
            for _ in 0..*numeric_count {
                numeric.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            numeric.reverse();
            let mut base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let mut logical_base = false;
            base = match base {
                Value::LogicalArray(la) => {
                    logical_base = true;
                    let data: Vec<f64> = la
                        .data
                        .iter()
                        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                        .collect();
                    let tensor = runmat_value::Tensor::new(data, la.shape.clone())
                        .map_err(|e| map_slice_shape_error("slice", e))?;
                    Value::Tensor(tensor)
                }
                other => other,
            };
            match base {
                Value::Object(obj) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::Object(obj.clone()),
                        ObjectIndexOp::Subsref,
                    ) {
                        return Err(err);
                    }
                    let descriptor = ObjectIndexDescriptor::subsref_paren_from_slice(
                        Value::Object(obj),
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )?;
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::HandleObject(handle) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::HandleObject(handle.clone()),
                        ObjectIndexOp::Subsref,
                    ) {
                        return Err(err);
                    }
                    let descriptor = ObjectIndexDescriptor::subsref_paren_from_slice(
                        Value::HandleObject(handle),
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )?;
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle { .. }
                | Value::Closure(_) => {
                    if *colon_mask != 0 || *end_mask != 0 {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedFunctionHandleSelector",
                            "Function handle call does not support colon or end selector syntax",
                        ));
                    }
                    let args = numeric;
                    match crate::call::feval::execute_feval(base, args, 1, function_registry)
                        .await?
                    {
                        crate::call::feval::FevalDispatch::Completed(value) => stack.push(value),
                    }
                }
                Value::Symbolic(expr) => {
                    if let Some((name, parameters)) = expr.function_reference_signature() {
                        if *colon_mask != 0 || *end_mask != 0 {
                            return Err(crate::interpreter::errors::mex(
                                "UnsupportedSymbolicFunctionSelector",
                                "symbolic function calls do not support colon or end selector syntax",
                            ));
                        }
                        stack.push(apply_symbolic_function_reference(
                            name, parameters, &numeric,
                        )?);
                    } else {
                        stack.push(
                            read_scalar_like_slice(
                                &Value::Symbolic(expr),
                                *dims,
                                *colon_mask,
                                *end_mask,
                                &numeric,
                            )
                            .await?,
                        );
                    }
                }
                Value::Tensor(t) => {
                    if *dims == 1 {
                        stack.push(
                            idx_read_slice::read_tensor_slice_1d(
                                &t,
                                *colon_mask,
                                *end_mask,
                                &numeric,
                            )
                            .await?,
                        )
                    } else {
                        stack.push(
                            idx_read_slice::read_tensor_slice_nd(
                                &t,
                                *dims,
                                *colon_mask,
                                *end_mask,
                                &numeric,
                            )
                            .await?,
                        )
                    }
                }
                Value::ComplexTensor(ct) => stack.push(
                    idx_read_slice::read_complex_slice(
                        &ct,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )
                    .await?,
                ),
                Value::SparseTensor(st) => stack.push(
                    idx_read_slice::read_sparse_slice(&st, *dims, *colon_mask, *end_mask, &numeric)
                        .await?,
                ),
                Value::GpuTensor(handle) => stack.push(
                    idx_read_slice::read_gpu_slice(
                        &handle,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )
                    .await?,
                ),
                Value::StringArray(sa) => stack.push(
                    idx_read_slice::read_string_slice(&sa, *dims, *colon_mask, *end_mask, &numeric)
                        .await?,
                ),
                Value::Cell(ca) => {
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &ca.shape)
                            .await?;
                    let plan = build_index_plan(&selectors, *dims, &ca.shape)?;
                    stack.push(gather_cell_with_plan(&ca, &plan)?);
                }
                other => {
                    stack.push(
                        read_scalar_like_slice(&other, *dims, *colon_mask, *end_mask, &numeric)
                            .await?,
                    );
                }
            }
            if logical_base {
                let result = stack.pop().ok_or(crate::interpreter::errors::mex(
                    "SliceNonTensor",
                    "logical slice missing result",
                ))?;
                let converted = match result {
                    Value::Tensor(t) => logical_value_from_tensor(t)?,
                    Value::Num(n) => Value::Bool(n != 0.0),
                    Value::Bool(_) | Value::LogicalArray(_) => result,
                    other => other,
                };
                stack.push(converted);
            }
            Ok(true)
        }
        crate::bytecode::Instr::IndexSliceExpr {
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
            let mut numeric: Vec<Value> = Vec::with_capacity(*numeric_count);
            for _ in 0..*numeric_count {
                numeric.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            numeric.reverse();
            let mut range_params: Vec<(f64, f64)> = Vec::with_capacity(range_dims.len());
            for i in (0..range_dims.len()).rev() {
                let has_step = range_has_step[i];
                let step = if has_step {
                    let v = stack.pop().ok_or(crate::interpreter::errors::mex(
                        "StackUnderflow",
                        "stack underflow",
                    ))?;
                    range_selector_scalar_to_f64(&v).await?
                } else {
                    1.0
                };
                let v = stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?;
                let start = range_selector_scalar_to_f64(&v).await?;
                range_params.push((start, step));
            }
            range_params.reverse();
            let mut base = stack.pop().ok_or(crate::interpreter::errors::mex(
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
                    Value::Int(value) => {
                        let tensor = integer_scalar_tensor(value.clone())?;
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(
                                *dims,
                                *colon_mask,
                                *end_mask,
                                range_dims,
                                &tensor.shape,
                            ),
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
            if let Value::GpuTensor(handle) = &base {
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
                        shape: &handle.shape,
                    },
                    vars,
                )
                .await?;

                if let Ok(result) = idx_read_slice::read_gpu_slice_from_plan(handle, &vm_plan) {
                    stack.push(result);
                    return Ok(true);
                }

                let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "AccelerationProviderUnavailable",
                        "No acceleration provider registered",
                    )
                })?;
                let host = provider.download(handle).await.map_err(|e| {
                    crate::interpreter::errors::mex(
                        "AccelerationOperationFailed",
                        &format!("slice: {e}"),
                    )
                })?;
                let tensor = runmat_value::Tensor::new(host.data, host.shape)
                    .map_err(|e| map_slice_shape_error("slice", e))?;
                base = Value::Tensor(tensor);
            }
            match base {
                Value::Object(obj) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::Object(obj.clone()),
                        ObjectIndexOp::Subsref,
                    ) {
                        return Err(err);
                    }
                    let descriptor = ObjectIndexDescriptor::subsref_paren_from_expr_slice(
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
                    )?;
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::HandleObject(handle) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::HandleObject(handle.clone()),
                        ObjectIndexOp::Subsref,
                    ) {
                        return Err(err);
                    }
                    let descriptor = ObjectIndexDescriptor::subsref_paren_from_expr_slice(
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
                    )?;
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
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
                    stack.push(idx_read_slice::read_complex_slice_from_plan(&t, &vm_plan)?);
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
                    stack.push(idx_read_slice::read_tensor_slice_from_plan(&t, &vm_plan)?);
                }
                Value::LogicalArray(la) => {
                    let data: Vec<f64> = la
                        .data
                        .iter()
                        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                        .collect();
                    let tensor = runmat_value::Tensor::new(data, la.shape.clone())
                        .map_err(|e| map_slice_shape_error("slice", e))?;
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
                    let sliced = idx_read_slice::read_tensor_slice_from_plan(&tensor, &vm_plan)?;
                    stack.push(match sliced {
                        Value::Tensor(t) => logical_value_from_tensor(t)?,
                        Value::Num(n) => Value::Bool(n != 0.0),
                        other => other,
                    });
                }
                Value::SparseTensor(s) => {
                    if let Some(full_range_colon_mask) =
                        sparse_full_range_exprs_as_colon_mask(SparseFullRangeExprSpec {
                            dims: *dims,
                            colon_mask: *colon_mask,
                            end_mask: *end_mask,
                            range_dims,
                            range_params: &range_params,
                            range_start_exprs,
                            range_step_exprs,
                            range_end_exprs,
                        })
                    {
                        stack.push(
                            idx_read_slice::read_sparse_slice(
                                &s,
                                *dims,
                                full_range_colon_mask,
                                *end_mask,
                                &numeric,
                            )
                            .await?,
                        );
                        return Ok(true);
                    }
                    let shape = s.shape();
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
                            shape: &shape,
                        },
                        vars,
                    )
                    .await?;
                    stack.push(idx_read_slice::read_sparse_slice_from_plan(&s, &vm_plan)?);
                }
                Value::StringArray(sa) => {
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
                    stack.push(idx_read_slice::gather_string_slice(&sa, &vm_plan)?);
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
                    stack.push(gather_cell_with_plan(&ca, &vm_plan)?);
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "SliceNonTensor",
                        "Slicing only supported on tensors",
                    ))
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
