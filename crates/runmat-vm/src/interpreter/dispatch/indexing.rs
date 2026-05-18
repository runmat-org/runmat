use crate::bytecode::EndExpr;
use crate::call::descriptor::{execute_callable_descriptor, CallableCallKind, CallableDescriptor};
use crate::call::shared::{
    call_object_index_descriptor_method, expand_brace_values, ObjectIndexDescriptor,
};
use crate::indexing::end_expr as idx_end_expr;
use crate::indexing::plan::{build_expr_index_plan, build_index_plan, ExprPlanSpec};
use crate::indexing::read_linear as idx_read_linear;
use crate::indexing::read_slice as idx_read_slice;
use crate::indexing::selectors::{build_slice_selectors, index_scalar_from_value, SliceSelector};
use crate::indexing::write_linear as idx_write_linear;
use crate::indexing::write_slice as idx_write_slice;
use runmat_builtins::{CellArray, Value};
use runmat_runtime::RuntimeError;
use std::future::Future;
use std::pin::Pin;

fn map_slice_plan_error(context: &str, err: RuntimeError) -> RuntimeError {
    format!("{context}: {}", err.message()).into()
}

async fn linear_index_values_to_f64(values: &[Value]) -> Result<Vec<f64>, RuntimeError> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let mut index_value = value.clone();
        if matches!(index_value, Value::GpuTensor(_)) {
            index_value = runmat_runtime::dispatcher::gather_if_needed_async(&index_value).await?;
        }
        let index_val = index_scalar_from_value(&index_value)
            .await?
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "UnsupportedIndexType",
                    &format!("Unsupported index type: expected numeric scalar, got {value:?}"),
                )
            })?;
        out.push(index_val as f64);
    }
    Ok(out)
}

fn assign_scalar_struct_index(
    _base: runmat_builtins::StructValue,
    indices: &[usize],
    rhs: Value,
) -> Result<Value, RuntimeError> {
    match indices {
        [1] | [1, 1] => Ok(rhs),
        _ => Err(crate::interpreter::errors::mex(
            "IndexOutOfBounds",
            "Struct subscript out of bounds",
        )),
    }
}

fn resolve_cell_indices(values: &[Value]) -> Result<Vec<usize>, RuntimeError> {
    values
        .iter()
        .map(|value| match value {
            Value::Num(index) => Ok(*index as usize),
            _ => {
                let index: f64 = value.try_into()?;
                Ok(index as usize)
            }
        })
        .collect()
}

fn apply_cell_end_offsets_for_base(
    base: &Value,
    raw_indices: &[Value],
    end_offsets: &[(usize, isize)],
) -> Result<Vec<Value>, RuntimeError> {
    if end_offsets.is_empty() {
        return Ok(raw_indices.to_vec());
    }
    let Value::Cell(ca) = base else {
        return Ok(raw_indices.to_vec());
    };
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
        if resolved < 1 || (resolved as usize) > len {
            return Err(crate::interpreter::errors::mex(
                "CellIndexOutOfBounds",
                "Cell index out of bounds",
            ));
        }
        adjusted[*position] = Value::Num(resolved as f64);
    }
    Ok(adjusted)
}

fn gather_cell_with_plan(
    ca: &CellArray,
    plan: &crate::indexing::plan::IndexPlan,
) -> Result<Value, RuntimeError> {
    let indices: Vec<usize> = plan.indices.iter().map(|idx| (*idx as usize) + 1).collect();
    crate::ops::cells::gather_cell_paren_linear_indices(ca, &indices, &plan.output_shape)
}

async fn build_cell_scalar_selectors(
    raw_indices: &[Value],
) -> Result<Vec<SliceSelector>, RuntimeError> {
    let mut selectors = Vec::with_capacity(raw_indices.len());
    for value in raw_indices {
        let idx_val = index_scalar_from_value(value).await?.ok_or_else(|| {
            crate::interpreter::errors::mex(
                "ScalarIndexRequired",
                "Cell indexing requires scalar numeric indices",
            )
        })?;
        selectors.push(SliceSelector::Scalar(if idx_val <= 0 {
            0
        } else {
            idx_val as usize
        }));
    }
    Ok(selectors)
}

fn pop_index_values(stack: &mut Vec<Value>, count: usize) -> Result<Vec<Value>, RuntimeError> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let value = stack.pop().ok_or(crate::interpreter::errors::mex(
            "StackUnderflow",
            "stack underflow",
        ))?;
        values.push(value);
    }
    values.reverse();
    Ok(values)
}

fn pop_index_base(stack: &mut Vec<Value>) -> Result<Value, RuntimeError> {
    stack.pop().ok_or(crate::interpreter::errors::mex(
        "StackUnderflow",
        "stack underflow",
    ))
}

enum BraceIndexOperation {
    ReadSingle,
    Expand { out_count: usize },
    List,
    Store { rhs: Value },
}

enum BraceIndexOutcome {
    Value(Value),
    Expanded(Vec<Value>),
}

async fn execute_brace_operation(
    base: Value,
    raw_indices: &[Value],
    operation: BraceIndexOperation,
) -> Result<BraceIndexOutcome, RuntimeError> {
    match operation {
        BraceIndexOperation::ReadSingle => {
            let value = match base {
                Value::Object(obj) => {
                    call_object_index_descriptor_method(ObjectIndexDescriptor::subsref_brace(
                        Value::Object(obj),
                        crate::call::shared::ObjectIndexSelector::IndexValues {
                            values: raw_indices.to_vec(),
                        },
                    ))
                    .await?
                }
                Value::HandleObject(handle) => {
                    call_object_index_descriptor_method(ObjectIndexDescriptor::subsref_brace(
                        Value::HandleObject(handle),
                        crate::call::shared::ObjectIndexSelector::IndexValues {
                            values: raw_indices.to_vec(),
                        },
                    ))
                    .await?
                }
                Value::Cell(ca) => {
                    let indices = resolve_cell_indices(raw_indices)?;
                    crate::ops::cells::index_cell_value(&ca, &indices)?
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
                    call_object_index_descriptor_method(ObjectIndexDescriptor::subsasgn_brace(
                        Value::Object(obj),
                        crate::call::shared::ObjectIndexSelector::IndexValues {
                            values: raw_indices.to_vec(),
                        },
                        rhs,
                    ))
                    .await?
                }
                Value::HandleObject(handle) => {
                    call_object_index_descriptor_method(ObjectIndexDescriptor::subsasgn_brace(
                        Value::HandleObject(handle),
                        crate::call::shared::ObjectIndexSelector::IndexValues {
                            values: raw_indices.to_vec(),
                        },
                        rhs,
                    ))
                    .await?
                }
                Value::Cell(ca) => {
                    let indices = resolve_cell_indices(raw_indices)?;
                    crate::ops::cells::assign_cell_value(ca, &indices, rhs, |oldv, newv| {
                        runmat_gc::gc_record_write(oldv, newv);
                    })?
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

#[derive(Clone, Copy)]
struct IndexContext<'a> {
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    range_dims: &'a [usize],
    base_shape: &'a [usize],
}

impl<'a> IndexContext<'a> {
    fn new(
        dims: usize,
        colon_mask: u32,
        end_mask: u32,
        range_dims: &'a [usize],
        base_shape: &'a [usize],
    ) -> Self {
        Self {
            dims,
            colon_mask,
            end_mask,
            range_dims,
            base_shape,
        }
    }

    fn dim_len_for_numeric_position(&self, numeric_position: usize) -> usize {
        let mut seen_numeric = 0usize;
        let mut dim_for_pos = 0usize;
        for d in 0..self.dims {
            let is_colon = (self.colon_mask & (1u32 << d)) != 0;
            let is_end = (self.end_mask & (1u32 << d)) != 0;
            let is_range = self.range_dims.contains(&d);
            if is_colon || is_end || is_range {
                continue;
            }
            if seen_numeric == numeric_position {
                dim_for_pos = d;
                break;
            }
            seen_numeric += 1;
        }
        if self.dims == 1 {
            let n = self.base_shape.iter().copied().product::<usize>();
            n.max(1)
        } else {
            self.base_shape.get(dim_for_pos).copied().unwrap_or(1)
        }
    }
}

async fn apply_end_offsets_to_numeric(
    numeric: &[Value],
    ctx: IndexContext<'_>,
    end_offsets: &[(usize, EndExpr)],
    vars: &mut [Value],
) -> Result<Vec<Value>, RuntimeError> {
    let mut adjusted = numeric.to_vec();
    for (position, end_expr) in end_offsets {
        if let Some(value) = adjusted.get_mut(*position) {
            let dim_len = ctx.dim_len_for_numeric_position(*position);
            let idx_val = resolve_range_end_index(dim_len, end_expr, vars).await?;
            *value = Value::Num(idx_val as f64);
        }
    }
    Ok(adjusted)
}

async fn resolve_range_end_index(
    dim_len: usize,
    end_expr: &EndExpr,
    vars: &[Value],
) -> Result<i64, RuntimeError> {
    fn eval_end_expr_value<'a>(
        expr: &'a EndExpr,
        end_value: f64,
        vars: &'a [Value],
    ) -> Pin<Box<dyn Future<Output = Result<f64, RuntimeError>> + 'a>> {
        Box::pin(async move {
            match expr {
                EndExpr::End => Ok(end_value),
                EndExpr::Const(v) => Ok(*v),
                EndExpr::Var(i) => idx_end_expr::value_to_f64(vars.get(*i).ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "MissingNumericIndex",
                        "missing variable for end expression",
                    )
                })?)
                .map_err(|_| {
                    crate::interpreter::errors::mex(
                        "UnsupportedIndexType",
                        "end expression must be numeric",
                    )
                }),
                EndExpr::ResolvedCall {
                    identity,
                    fallback_policy,
                    display_name,
                    args,
                } => {
                    let mut argv: Vec<Value> = Vec::with_capacity(args.len());
                    for a in args {
                        let val = eval_end_expr_value(a, end_value, vars).await?;
                        argv.push(Value::Num(val));
                    }
                    let descriptor = CallableDescriptor::resolved(
                        identity.clone(),
                        display_name.clone(),
                        argv,
                        1,
                        *fallback_policy,
                        CallableCallKind::EndExpr,
                    )
                    .with_call_kind(CallableCallKind::EndExpr);
                    let v = execute_callable_descriptor(descriptor).await?;
                    idx_end_expr::value_to_f64(&v).map_err(|_| {
                        crate::interpreter::errors::mex(
                            "UnsupportedIndexType",
                            "end call must return scalar",
                        )
                    })
                }
                EndExpr::Add(a, b) => Ok(eval_end_expr_value(a, end_value, vars).await?
                    + eval_end_expr_value(b, end_value, vars).await?),
                EndExpr::Sub(a, b) => Ok(eval_end_expr_value(a, end_value, vars).await?
                    - eval_end_expr_value(b, end_value, vars).await?),
                EndExpr::Mul(a, b) => Ok(eval_end_expr_value(a, end_value, vars).await?
                    * eval_end_expr_value(b, end_value, vars).await?),
                EndExpr::Div(a, b) => {
                    let denom = eval_end_expr_value(b, end_value, vars).await?;
                    if denom == 0.0 {
                        return Err(crate::interpreter::errors::mex(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                    Ok(eval_end_expr_value(a, end_value, vars).await? / denom)
                }
                EndExpr::LeftDiv(a, b) => {
                    let denom = eval_end_expr_value(a, end_value, vars).await?;
                    if denom == 0.0 {
                        return Err(crate::interpreter::errors::mex(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                    Ok(eval_end_expr_value(b, end_value, vars).await? / denom)
                }
                EndExpr::Pow(a, b) => Ok(eval_end_expr_value(a, end_value, vars)
                    .await?
                    .powf(eval_end_expr_value(b, end_value, vars).await?)),
                EndExpr::Neg(a) => Ok(-eval_end_expr_value(a, end_value, vars).await?),
                EndExpr::Pos(a) => Ok(eval_end_expr_value(a, end_value, vars).await?),
                EndExpr::Floor(a) => Ok(eval_end_expr_value(a, end_value, vars).await?.floor()),
                EndExpr::Ceil(a) => Ok(eval_end_expr_value(a, end_value, vars).await?.ceil()),
                EndExpr::Round(a) => Ok(eval_end_expr_value(a, end_value, vars).await?.round()),
                EndExpr::Fix(a) => {
                    let v = eval_end_expr_value(a, end_value, vars).await?;
                    Ok(if v >= 0.0 { v.floor() } else { v.ceil() })
                }
            }
        })
    }

    Ok(eval_end_expr_value(end_expr, dim_len as f64, vars)
        .await?
        .floor() as i64)
}

pub async fn dispatch_indexing(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    vars: &mut Vec<Value>,
    semantic_registry: &crate::bytecode::SemanticFunctionRegistry,
    pc: usize,
    mut clear_value_residency: impl FnMut(&Value),
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::Index(num_indices) => {
            let mut raw_indices = Vec::with_capacity(*num_indices);
            for _ in 0..*num_indices {
                raw_indices.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            raw_indices.reverse();
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            match &base {
                Value::Object(_) | Value::HandleObject(_) => {
                    let descriptor = ObjectIndexDescriptor::subsref_paren(
                        base,
                        crate::call::shared::ObjectIndexSelector::IndexValues {
                            values: raw_indices.clone(),
                        },
                    );
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::Cell(ca) => {
                    let selectors = build_cell_scalar_selectors(&raw_indices).await?;
                    let plan = build_index_plan(&selectors, raw_indices.len(), &ca.shape)?;
                    stack.push(gather_cell_with_plan(ca, &plan)?);
                }
                Value::FunctionHandle(_)
                | Value::SemanticFunctionHandle { .. }
                | Value::Closure(_) => {
                    let args = raw_indices;
                    match crate::call::feval::execute_feval(base, args, 1, semantic_registry)
                        .await?
                    {
                        crate::call::feval::FevalDispatch::Completed(value) => stack.push(value),
                    }
                }
                _ => {
                    let numeric = linear_index_values_to_f64(&raw_indices).await?;
                    stack.push(idx_read_linear::generic_index(&base, &numeric).await?);
                }
            }
            Ok(true)
        }
        crate::bytecode::Instr::IndexCell {
            num_indices,
            end_offsets,
        } => {
            let raw_indices = pop_index_values(stack, *num_indices)?;
            let base = pop_index_base(stack)?;
            let adjusted_indices =
                apply_cell_end_offsets_for_base(&base, &raw_indices, end_offsets)?;
            let outcome =
                execute_brace_operation(base, &adjusted_indices, BraceIndexOperation::ReadSingle)
                    .await?;
            match outcome {
                BraceIndexOutcome::Value(value) => stack.push(value),
                BraceIndexOutcome::Expanded(_) => {
                    return Err(crate::interpreter::errors::mex(
                        "InvalidBraceIndexOutcome",
                        "IndexCell expected a single value outcome",
                    ))
                }
            }
            Ok(true)
        }
        crate::bytecode::Instr::IndexCellExpand {
            num_indices,
            out_count,
            end_offsets,
        } => {
            let raw_indices = pop_index_values(stack, *num_indices)?;
            let base = pop_index_base(stack)?;
            let adjusted_indices =
                apply_cell_end_offsets_for_base(&base, &raw_indices, end_offsets)?;
            let outcome = execute_brace_operation(
                base,
                &adjusted_indices,
                BraceIndexOperation::Expand {
                    out_count: *out_count,
                },
            )
            .await?;
            match outcome {
                BraceIndexOutcome::Expanded(values) => {
                    for value in values {
                        stack.push(value);
                    }
                }
                BraceIndexOutcome::Value(_) => {
                    return Err(crate::interpreter::errors::mex(
                        "InvalidBraceIndexOutcome",
                        "IndexCellExpand expected an expanded value list",
                    ))
                }
            }
            Ok(true)
        }
        crate::bytecode::Instr::IndexCellList {
            num_indices,
            end_offsets,
        } => {
            let raw_indices = pop_index_values(stack, *num_indices)?;
            let base = pop_index_base(stack)?;
            let adjusted_indices =
                apply_cell_end_offsets_for_base(&base, &raw_indices, end_offsets)?;
            let outcome =
                execute_brace_operation(base, &adjusted_indices, BraceIndexOperation::List).await?;
            match outcome {
                BraceIndexOutcome::Value(value) => stack.push(value),
                BraceIndexOutcome::Expanded(_) => {
                    return Err(crate::interpreter::errors::mex(
                        "InvalidBraceIndexOutcome",
                        "IndexCellList expected a single list value",
                    ))
                }
            }
            Ok(true)
        }
        crate::bytecode::Instr::StoreIndexCell {
            num_indices,
            end_offsets,
        }
        | crate::bytecode::Instr::StoreIndexCellDelete {
            num_indices,
            end_offsets,
        } => {
            let delete = matches!(instr, crate::bytecode::Instr::StoreIndexCellDelete { .. });
            if delete {
                return Err(crate::interpreter::errors::mex(
                    "UnsupportedCellBraceDeletion",
                    "Cell brace assignment does not support deletion",
                ));
            }
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let raw_indices = pop_index_values(stack, *num_indices)?;
            let base = pop_index_base(stack)?;
            let adjusted_indices =
                apply_cell_end_offsets_for_base(&base, &raw_indices, end_offsets)?;
            let outcome = execute_brace_operation(
                base,
                &adjusted_indices,
                BraceIndexOperation::Store { rhs },
            )
            .await?;
            match outcome {
                BraceIndexOutcome::Value(value) => stack.push(value),
                BraceIndexOutcome::Expanded(_) => {
                    return Err(crate::interpreter::errors::mex(
                        "InvalidBraceIndexOutcome",
                        "StoreIndexCell expected a single base value",
                    ))
                }
            }
            Ok(true)
        }
        crate::bytecode::Instr::StoreIndex(num_indices)
        | crate::bytecode::Instr::StoreIndexDelete(num_indices) => {
            let delete = matches!(instr, crate::bytecode::Instr::StoreIndexDelete(_));
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
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
                indices.push(if idx_val <= 0 { 0 } else { idx_val as usize });
            }
            indices.reverse();
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            clear_value_residency(&base);
            match base {
                Value::Object(obj) => {
                    let descriptor = ObjectIndexDescriptor::subsasgn_paren(
                        Value::Object(obj),
                        crate::call::shared::ObjectIndexSelector::ScalarIndices { indices },
                        rhs,
                    );
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::HandleObject(handle) => {
                    let descriptor = ObjectIndexDescriptor::subsasgn_paren(
                        Value::HandleObject(handle),
                        crate::call::shared::ObjectIndexSelector::ScalarIndices { indices },
                        rhs,
                    );
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::Tensor(t) => stack
                    .push(idx_write_linear::assign_tensor_scalar(t, &indices, &rhs, delete).await?),
                Value::ComplexTensor(t) => stack.push(
                    idx_write_linear::assign_complex_scalar(t, &indices, &rhs, delete).await?,
                ),
                Value::Cell(ca) => stack.push(crate::ops::cells::assign_cell_paren_with_policy(
                    ca, &indices, &rhs, delete,
                )?),
                Value::Struct(st) => stack.push(assign_scalar_struct_index(st, &indices, rhs)?),
                Value::GpuTensor(h) => stack
                    .push(idx_write_linear::assign_gpu_scalar(&h, &indices, &rhs, delete).await?),
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
                    let tensor = runmat_builtins::Tensor::new(data, la.shape.clone())
                        .map_err(|e| format!("slice: {e}"))?;
                    Value::Tensor(tensor)
                }
                other => other,
            };
            match base {
                Value::Object(obj) => {
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
                | Value::SemanticFunctionHandle { .. }
                | Value::Closure(_) => {
                    if *colon_mask != 0 || *end_mask != 0 {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedFunctionHandleSelector",
                            "Function handle call does not support colon or end selector syntax",
                        ));
                    }
                    let args = numeric;
                    match crate::call::feval::execute_feval(base, args, 1, semantic_registry)
                        .await?
                    {
                        crate::call::feval::FevalDispatch::Completed(value) => stack.push(value),
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
                    .await
                    .map_err(|e| format!("slice: {e}"))?,
                ),
                Value::GpuTensor(handle) => stack.push(
                    idx_read_slice::read_gpu_slice(
                        &handle,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )
                    .await
                    .map_err(|e| format!("slice: {e}"))?,
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
                    if *dims == 1 {
                        if (*colon_mask & 1u32) != 0 {
                            return Err(crate::interpreter::errors::mex(
                                "SliceNonTensor",
                                "Slicing only supported on tensors",
                            ));
                        }
                        let selectors =
                            build_slice_selectors(1, *colon_mask, *end_mask, &numeric, &[1usize])
                                .await?;
                        let linear_indices: Vec<f64> = match selectors.first() {
                            Some(SliceSelector::Scalar(index)) => vec![*index as f64],
                            Some(SliceSelector::Indices(indices)) => {
                                indices.iter().map(|&index| index as f64).collect()
                            }
                            Some(SliceSelector::LinearIndices { values, .. }) => {
                                values.iter().map(|&index| index as f64).collect()
                            }
                            Some(SliceSelector::Colon) | None => {
                                return Err(crate::interpreter::errors::mex(
                                    "SliceNonTensor",
                                    "Slicing only supported on tensors",
                                ));
                            }
                        };
                        let v = runmat_runtime::perform_indexing(&other, &linear_indices)
                            .await
                            .map_err(|_| {
                                crate::interpreter::errors::mex(
                                    "SliceNonTensor",
                                    "Slicing only supported on tensors",
                                )
                            })?;
                        stack.push(v);
                    } else {
                        return Err(crate::interpreter::errors::mex(
                            "SliceNonTensor",
                            "Slicing only supported on tensors",
                        ));
                    }
                }
            }
            if logical_base {
                let result = stack.pop().ok_or(crate::interpreter::errors::mex(
                    "SliceNonTensor",
                    "logical slice missing result",
                ))?;
                let converted = match result {
                    Value::Tensor(t) => {
                        let logical_data: Vec<u8> = t
                            .data
                            .iter()
                            .map(|&v| if v != 0.0 { 1 } else { 0 })
                            .collect();
                        if logical_data.len() <= 1 {
                            Value::Bool(logical_data.first().copied().unwrap_or(0) != 0)
                        } else {
                            let logical =
                                runmat_builtins::LogicalArray::new(logical_data, t.shape.clone())
                                    .map_err(|e| {
                                    crate::interpreter::errors::mex(
                                        "SliceNonTensor",
                                        &format!("slice: {e}"),
                                    )
                                })?;
                            Value::LogicalArray(logical)
                        }
                    }
                    Value::Num(n) => Value::Bool(n != 0.0),
                    Value::Bool(_) | Value::LogicalArray(_) => result,
                    other => other,
                };
                stack.push(converted);
            }
            Ok(true)
        }
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
                Value::ComplexTensor(mut ct) => {
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &ct.shape)
                            .await
                            .map_err(|e| format!("slice assign: {e}"))?;
                    let plan = build_index_plan(&selectors, *dims, &ct.shape)
                        .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    if delete {
                        stack.push(idx_write_slice::delete_complex_with_plan(ct, &plan, &rhs)?);
                        return Ok(true);
                    }
                    if plan.indices.is_empty() {
                        stack.push(Value::ComplexTensor(ct));
                        return Ok(true);
                    }
                    let rhs_view =
                        idx_write_slice::build_complex_rhs_view(&rhs, &plan.selection_lengths)
                            .map_err(|e| format!("slice assign: {e}"))?;
                    idx_write_slice::scatter_complex_with_plan(&mut ct, &plan, &rhs_view)
                        .map_err(|e| format!("slice assign: {e}"))?;
                    stack.push(Value::ComplexTensor(ct));
                }
                Value::Cell(ca) => {
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &ca.shape)
                            .await
                            .map_err(|e| format!("cell slice assign: {e}"))?;
                    let plan = build_index_plan(&selectors, *dims, &ca.shape)
                        .map_err(|e| map_slice_plan_error("cell slice assign", e))?;
                    let selected: Vec<usize> =
                        plan.indices.iter().map(|idx| (*idx as usize) + 1).collect();
                    stack.push(
                        crate::ops::cells::assign_cell_paren_linear_indices_with_policy(
                            ca, &selected, &rhs, delete,
                        )?,
                    );
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
                            .map_err(|e| format!("slice assign: {e}"))?;
                    let plan = build_index_plan(&selectors, *dims, &sa.shape)
                        .map_err(|e| map_slice_plan_error("slice assign", e))?;
                    if plan.indices.is_empty() {
                        stack.push(Value::StringArray(sa));
                        return Ok(true);
                    }
                    let rhs_view =
                        idx_write_slice::build_string_rhs_view(&rhs, &plan.selection_lengths)
                            .map_err(|e| format!("slice assign: {e}"))?;
                    idx_write_slice::scatter_string_with_plan(&mut sa, &plan, &rhs_view)
                        .map_err(|e| format!("slice assign: {e}"))?;
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
                    return Err(
                        "Slicing assignment only supported on tensors or string arrays"
                            .to_string()
                            .into(),
                    );
                }
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
                    match v {
                        Value::Num(n) => n,
                        Value::Int(i) => i.to_f64(),
                        Value::Tensor(t) if !t.data.is_empty() => t.data[0],
                        _ => 1.0,
                    }
                } else {
                    1.0
                };
                let v = stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?;
                let start = match v {
                    Value::Num(n) => n,
                    Value::Int(i) => i.to_f64(),
                    Value::Tensor(t) if !t.data.is_empty() => t.data[0],
                    _ => 1.0,
                };
                range_params.push((start, step));
            }
            range_params.reverse();
            let mut base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            clear_value_residency(&base);
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
                let vm_plan = build_expr_index_plan(
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
                    |dim_len, expr| {
                        let expr = expr.clone();
                        let vars_ref = &*vars;
                        async move { resolve_range_end_index(dim_len, &expr, vars_ref).await }
                    },
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
                let host = provider
                    .download(handle)
                    .await
                    .map_err(|e| format!("slice: {e}"))?;
                let tensor = runmat_builtins::Tensor::new(host.data, host.shape)
                    .map_err(|e| format!("slice: {e}"))?;
                base = Value::Tensor(tensor);
            }
            match base {
                Value::ComplexTensor(t) => {
                    let vm_plan = build_expr_index_plan(
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
                        |dim_len, expr| {
                            let expr = expr.clone();
                            let vars_ref = &*vars;
                            async move { resolve_range_end_index(dim_len, &expr, vars_ref).await }
                        },
                    )
                    .await?;
                    stack.push(
                        idx_read_slice::read_complex_slice_from_plan(&t, &vm_plan)
                            .map_err(|e| format!("Slice error: {e}"))?,
                    );
                }
                Value::Tensor(t) => {
                    let vm_plan = build_expr_index_plan(
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
                        |dim_len, expr| {
                            let expr = expr.clone();
                            let vars_ref = &*vars;
                            async move { resolve_range_end_index(dim_len, &expr, vars_ref).await }
                        },
                    )
                    .await?;
                    stack.push(
                        idx_read_slice::read_tensor_slice_from_plan(&t, &vm_plan)
                            .map_err(|e| format!("Slice error: {e}"))?,
                    );
                }
                Value::StringArray(sa) => {
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &sa.shape)
                            .await
                            .map_err(|e| format!("slice: {e}"))?;
                    let plan = build_index_plan(&selectors, *dims, &sa.shape)
                        .map_err(|e| map_slice_plan_error("slice", e))?;
                    stack.push(
                        idx_read_slice::gather_string_slice(&sa, &plan)
                            .map_err(|e| format!("slice: {e}"))?,
                    );
                }
                Value::Cell(ca) => {
                    let vm_plan = build_expr_index_plan(
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
                        |dim_len, expr| {
                            let expr = expr.clone();
                            let vars_ref = &*vars;
                            async move { resolve_range_end_index(dim_len, &expr, vars_ref).await }
                        },
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
            let delete = matches!(instr, crate::bytecode::Instr::StoreSliceExprDelete { .. });
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let mut range_params: Vec<(f64, f64)> = Vec::with_capacity(range_dims.len());
            for i in (0..range_dims.len()).rev() {
                let has = range_has_step[i];
                let step = if has {
                    let v: f64 = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                        "StackUnderflow",
                        "stack underflow",
                    ))?)
                        .try_into()?;
                    v
                } else {
                    1.0
                };
                let st: f64 = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?)
                    .try_into()?;
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
            clear_value_residency(&base);
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
                Value::ComplexTensor(mut t) => {
                    let vm_plan = build_expr_index_plan(
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
                        |dim_len, expr| {
                            let expr = expr.clone();
                            let vars_ref = &*vars;
                            async move { resolve_range_end_index(dim_len, &expr, vars_ref).await }
                        },
                    )
                    .await?;
                    if delete {
                        stack.push(idx_write_slice::delete_complex_with_plan(
                            t, &vm_plan, &rhs,
                        )?);
                        return Ok(true);
                    }
                    if !vm_plan.indices.is_empty() {
                        let rhs_view = idx_write_slice::build_complex_rhs_view(
                            &rhs,
                            &vm_plan.selection_lengths,
                        )
                        .map_err(|e| format!("slice assign: {e}"))?;
                        idx_write_slice::scatter_complex_with_plan(&mut t, &vm_plan, &rhs_view)
                            .map_err(|e| format!("slice assign: {e}"))?;
                    }
                    stack.push(Value::ComplexTensor(t));
                }
                Value::Tensor(t) => {
                    let vm_plan = build_expr_index_plan(
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
                        |dim_len, expr| {
                            let expr = expr.clone();
                            let vars_ref = &*vars;
                            async move { resolve_range_end_index(dim_len, &expr, vars_ref).await }
                        },
                    )
                    .await?;
                    stack.push(if delete {
                        idx_write_slice::delete_tensor_with_plan(t, &vm_plan, &rhs)?
                    } else {
                        idx_write_slice::assign_tensor_with_plan(t, &vm_plan, &rhs).await?
                    });
                }
                Value::GpuTensor(h) => {
                    let vm_plan = build_expr_index_plan(
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
                        |dim_len, expr| {
                            let expr = expr.clone();
                            let vars_ref = &*vars;
                            async move { resolve_range_end_index(dim_len, &expr, vars_ref).await }
                        },
                    )
                    .await?;
                    let updated = if delete {
                        idx_write_slice::delete_gpu_slice_with_plan(&h, &vm_plan, &rhs).await?
                    } else {
                        idx_write_slice::assign_gpu_slice_with_plan(&h, &vm_plan, &rhs).await?
                    };
                    stack.push(updated);
                }
                Value::Object(obj) => {
                    let descriptor = ObjectIndexDescriptor::subsasgn_paren_from_expr_slice(
                        Value::Object(obj),
                        *dims,
                        *colon_mask,
                        *end_mask,
                        range_dims,
                        &range_params,
                        range_start_exprs,
                        range_step_exprs,
                        range_end_exprs,
                        &numeric,
                        rhs,
                    )?;
                    stack.push(call_object_index_descriptor_method(descriptor).await?);
                }
                Value::HandleObject(handle) => {
                    let descriptor = ObjectIndexDescriptor::subsasgn_paren_from_expr_slice(
                        Value::HandleObject(handle),
                        *dims,
                        *colon_mask,
                        *end_mask,
                        range_dims,
                        &range_params,
                        range_start_exprs,
                        range_step_exprs,
                        range_end_exprs,
                        &numeric,
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
                    let vm_plan = build_expr_index_plan(
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
                        |dim_len, expr| {
                            let expr = expr.clone();
                            let vars_ref = &*vars;
                            async move { resolve_range_end_index(dim_len, &expr, vars_ref).await }
                        },
                    )
                    .await?;
                    if !vm_plan.indices.is_empty() {
                        let rhs_view = idx_write_slice::build_string_rhs_view(
                            &rhs,
                            &vm_plan.selection_lengths,
                        )
                        .map_err(|e| format!("slice assign: {e}"))?;
                        idx_write_slice::scatter_string_with_plan(&mut sa, &vm_plan, &rhs_view)
                            .map_err(|e| format!("slice assign: {e}"))?;
                    }
                    stack.push(Value::StringArray(sa));
                }
                Value::Cell(ca) => {
                    let vm_plan = build_expr_index_plan(
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
                        |dim_len, expr| {
                            let expr = expr.clone();
                            let vars_ref = &*vars;
                            async move { resolve_range_end_index(dim_len, &expr, vars_ref).await }
                        },
                    )
                    .await?;
                    let selected: Vec<usize> = vm_plan
                        .indices
                        .iter()
                        .map(|idx| (*idx as usize) + 1)
                        .collect();
                    stack.push(
                        crate::ops::cells::assign_cell_paren_linear_indices_with_policy(
                            ca, &selected, &rhs, delete,
                        )?,
                    );
                }
                _ => {
                    return Err(
                        "StoreSliceExpr only supports tensors, cells, and string arrays currently"
                            .to_string()
                            .into(),
                    )
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
