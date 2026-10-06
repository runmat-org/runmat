use super::*;

pub(super) async fn resolve_cell_indices(values: &[Value]) -> Result<Vec<usize>, RuntimeError> {
    let selectors = build_cell_scalar_selectors(values)
        .await
        .map_err(map_cell_scalar_selector_error)?;
    let mut indices = Vec::with_capacity(selectors.len());
    for selector in selectors {
        match selector {
            SliceSelector::Scalar(index) => indices.push(index),
            _ => {
                return Err(crate::interpreter::errors::mex(
                    "CellIndexType",
                    "Unsupported cell index type",
                ))
            }
        }
    }
    Ok(indices)
}

pub(super) fn apply_cell_end_offsets_for_base(
    base: &Value,
    raw_indices: &[Value],
    end_offsets: &[(usize, isize)],
    allow_end_plus_one_growth: bool,
) -> Result<Vec<Value>, RuntimeError> {
    if end_offsets.is_empty() {
        return Ok(raw_indices.to_vec());
    }
    let Value::Cell(ca) = base else {
        return Ok(raw_indices.to_vec());
    };
    let mut seen = vec![false; raw_indices.len()];
    for (position, _) in end_offsets {
        if *position >= raw_indices.len() {
            return Err(crate::interpreter::errors::mex(
                "InvalidEndSelectorPlan",
                "cell end-selector position is out of bounds",
            ));
        }
        if std::mem::replace(&mut seen[*position], true) {
            return Err(crate::interpreter::errors::mex(
                "InvalidEndSelectorPlan",
                "cell end-selector position appears more than once",
            ));
        }
    }
    let mut adjusted = raw_indices.to_vec();
    for (position, offset) in end_offsets {
        if *position >= adjusted.len() {
            return Err(crate::interpreter::errors::mex(
                "CellIndexOutOfBounds",
                "Cell end selector position is out of bounds",
            ));
        }
        let len = if adjusted.len() == 1 {
            ca.rows * ca.cols
        } else if *position == 0 {
            ca.rows
        } else {
            ca.cols
        };
        let resolved = (len as isize) + *offset;
        if resolved < 1 || (!allow_end_plus_one_growth && (resolved as usize) > len) {
            return Err(crate::interpreter::errors::mex(
                "CellIndexOutOfBounds",
                "Cell index out of bounds",
            ));
        }
        adjusted[*position] = Value::Num(resolved as f64);
    }
    Ok(adjusted)
}

pub(super) async fn apply_cell_end_exprs_for_base(
    base: &Value,
    raw_indices: &[Value],
    end_exprs: &[(usize, EndExpr)],
    vars: &mut [Value],
    allow_end_plus_one_growth: bool,
) -> Result<Vec<Value>, RuntimeError> {
    if end_exprs.is_empty() {
        return Ok(raw_indices.to_vec());
    }
    let Value::Cell(ca) = base else {
        return Ok(raw_indices.to_vec());
    };
    let mut seen = vec![false; raw_indices.len()];
    for (position, _) in end_exprs {
        if *position >= raw_indices.len() {
            return Err(crate::interpreter::errors::mex(
                "InvalidEndSelectorPlan",
                "cell end-selector position is out of bounds",
            ));
        }
        if std::mem::replace(&mut seen[*position], true) {
            return Err(crate::interpreter::errors::mex(
                "InvalidEndSelectorPlan",
                "cell end-selector position appears more than once",
            ));
        }
    }
    let mut adjusted = raw_indices.to_vec();
    for (position, end_expr) in end_exprs {
        if *position >= adjusted.len() {
            return Err(crate::interpreter::errors::mex(
                "CellIndexOutOfBounds",
                "Cell end selector position is out of bounds",
            ));
        }
        let dim_len = if adjusted.len() == 1 {
            ca.rows * ca.cols
        } else if *position == 0 {
            ca.rows
        } else {
            ca.cols
        };
        let resolved = resolve_end_expr_index(dim_len, end_expr, vars).await?;
        if resolved < 1 || (!allow_end_plus_one_growth && (resolved as usize) > dim_len) {
            return Err(crate::interpreter::errors::mex(
                "CellIndexOutOfBounds",
                "Cell index out of bounds",
            ));
        }
        adjusted[*position] = Value::Num(resolved as f64);
    }
    Ok(adjusted)
}

pub(super) enum BraceIndexOperation {
    ReadSingle,
    Expand { out_count: usize },
    List,
    Store { rhs: Value },
}

pub(super) enum BraceIndexOutcome {
    Value(Value),
    Expanded(Vec<Value>),
}

pub(super) enum BraceOutcomeExpectation {
    SingleValue { invalid_message: &'static str },
    ExpandedValues { invalid_message: &'static str },
}

pub(super) async fn execute_brace_operation(
    base: Value,
    raw_indices: &[Value],
    operation: BraceIndexOperation,
) -> Result<BraceIndexOutcome, RuntimeError> {
    if matches!(
        base,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    ) {
        return Err(crate::interpreter::errors::mex(
            "UnsupportedFunctionHandleSelector",
            "Function handle call does not support brace selector syntax",
        ));
    }
    match operation {
        BraceIndexOperation::ReadSingle => {
            let value = match base {
                Value::Object(obj) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::Object(obj.clone()),
                        ObjectIndexOp::Subsref,
                    ) {
                        return Err(err);
                    }
                    call_object_index_descriptor_method(ObjectIndexDescriptor::subsref_brace(
                        Value::Object(obj),
                        ObjectIndexSelector::IndexValues {
                            values: raw_indices.to_vec(),
                        },
                    ))
                    .await?
                }
                Value::HandleObject(handle) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::HandleObject(handle.clone()),
                        ObjectIndexOp::Subsref,
                    ) {
                        return Err(err);
                    }
                    call_object_index_descriptor_method(ObjectIndexDescriptor::subsref_brace(
                        Value::HandleObject(handle),
                        ObjectIndexSelector::IndexValues {
                            values: raw_indices.to_vec(),
                        },
                    ))
                    .await?
                }
                Value::Cell(ca) => {
                    let indices = resolve_cell_indices(raw_indices).await?;
                    runmat_runtime::object::cell::index_cell_value(&ca, &indices)?
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "CellIndexingOnNonCell",
                        "Cell indexing on non-cell",
                    ))
                }
            };
            Ok(BraceIndexOutcome::Value(value))
        }
        BraceIndexOperation::Expand { out_count } => {
            let values = expand_brace_values(base, raw_indices, Some(out_count)).await?;
            Ok(BraceIndexOutcome::Expanded(values))
        }
        BraceIndexOperation::List => {
            if !matches!(
                base,
                Value::Cell(_) | Value::Object(_) | Value::HandleObject(_)
            ) {
                return Err(crate::interpreter::errors::mex(
                    "CellIndexingOnNonCell",
                    "Cell indexing on non-cell",
                ));
            }
            let values = expand_brace_values(base, raw_indices, None).await?;
            let value = if values.len() == 1 {
                values.into_iter().next().unwrap_or(Value::Num(0.0))
            } else {
                Value::OutputList(values)
            };
            Ok(BraceIndexOutcome::Value(value))
        }
        BraceIndexOperation::Store { rhs } => {
            let value = match base {
                Value::Object(obj) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::Object(obj.clone()),
                        ObjectIndexOp::Subsasgn,
                    ) {
                        return Err(err);
                    }
                    call_object_index_descriptor_method(ObjectIndexDescriptor::subsasgn_brace(
                        Value::Object(obj),
                        ObjectIndexSelector::IndexValues {
                            values: raw_indices.to_vec(),
                        },
                        rhs,
                    ))
                    .await?
                }
                Value::HandleObject(handle) => {
                    if let Some(err) = missing_member_index_overload_error(
                        &Value::HandleObject(handle.clone()),
                        ObjectIndexOp::Subsasgn,
                    ) {
                        return Err(err);
                    }
                    call_object_index_descriptor_method(ObjectIndexDescriptor::subsasgn_brace(
                        Value::HandleObject(handle),
                        ObjectIndexSelector::IndexValues {
                            values: raw_indices.to_vec(),
                        },
                        rhs,
                    ))
                    .await?
                }
                Value::Cell(ca) => {
                    if raw_indices.len() > 1 {
                        let indices = resolve_cell_indices(raw_indices).await?;
                        match rhs {
                            Value::OutputList(values) if values.len() == 1 => {
                                runmat_runtime::object::cell::assign_cell_value(
                                    ca,
                                    &indices,
                                    values.into_iter().next().unwrap_or(Value::Num(0.0)),
                                    |oldv, newv| {
                                        runmat_gc::gc_record_write(oldv, newv);
                                    },
                                )?
                            }
                            Value::OutputList(_) => {
                                return Err(crate::interpreter::errors::mex(
                                    "CellAssignmentArityMismatch",
                                    "Cell brace assignment target count does not match source value count",
                                ))
                            }
                            other => runmat_runtime::object::cell::assign_cell_value(
                                ca,
                                &indices,
                                other,
                                |oldv, newv| {
                                    runmat_gc::gc_record_write(oldv, newv);
                                },
                            )?,
                        }
                    } else {
                        let positions =
                            runmat_runtime::object::cell::resolve_cell_assignment_positions(
                                &ca,
                                raw_indices,
                            )?;
                        match rhs {
                            Value::OutputList(values) => runmat_runtime::object::cell::assign_cell_value_multi(
                                ca,
                                &positions,
                                &values,
                                |oldv, newv| {
                                    runmat_gc::gc_record_write(oldv, newv);
                                },
                            )?,
                            other if positions.len() == 1 => runmat_runtime::object::cell::assign_cell_value(
                                ca,
                                &positions,
                                other,
                                |oldv, newv| {
                                    runmat_gc::gc_record_write(oldv, newv);
                                },
                            )?,
                            _ => {
                                return Err(crate::interpreter::errors::mex(
                                    "CellAssignmentArityMismatch",
                                    "Cell brace assignment target count does not match source value count",
                                ))
                            }
                        }
                    }
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "CellAssignmentOnNonCell",
                        "Cell assignment on non-cell",
                    ))
                }
            };
            Ok(BraceIndexOutcome::Value(value))
        }
    }
}

pub(super) async fn execute_brace_operation_from_stack(
    stack: &mut Vec<Value>,
    num_indices: usize,
    end_offsets: &[(usize, isize)],
    end_exprs: &[(usize, EndExpr)],
    vars: &mut [Value],
    operation: BraceIndexOperation,
    expectation: BraceOutcomeExpectation,
) -> Result<(), RuntimeError> {
    let raw_indices = pop_index_values(stack, num_indices)?;
    let base = pop_index_base(stack)?;
    let allow_end_plus_one_growth = matches!(operation, BraceIndexOperation::Store { .. });
    let adjusted_end_exprs = apply_cell_end_exprs_for_base(
        &base,
        &raw_indices,
        end_exprs,
        vars,
        allow_end_plus_one_growth,
    )
    .await?;
    let adjusted_indices = apply_cell_end_offsets_for_base(
        &base,
        &adjusted_end_exprs,
        end_offsets,
        allow_end_plus_one_growth,
    )?;
    let outcome = execute_brace_operation(base, &adjusted_indices, operation).await?;
    match (outcome, expectation) {
        (BraceIndexOutcome::Value(value), BraceOutcomeExpectation::SingleValue { .. }) => {
            stack.push(value)
        }
        (BraceIndexOutcome::Expanded(values), BraceOutcomeExpectation::ExpandedValues { .. }) => {
            for value in values {
                stack.push(value);
            }
        }
        (
            BraceIndexOutcome::Value(_),
            BraceOutcomeExpectation::ExpandedValues { invalid_message },
        )
        | (
            BraceIndexOutcome::Expanded(_),
            BraceOutcomeExpectation::SingleValue { invalid_message },
        ) => {
            return Err(crate::interpreter::errors::mex(
                "InvalidBraceIndexOutcome",
                invalid_message,
            ))
        }
    }
    Ok(())
}
