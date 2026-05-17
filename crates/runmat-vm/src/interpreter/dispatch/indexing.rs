use crate::bytecode::EndExpr;
use crate::call::descriptor::{
    try_execute_callable_descriptor, CallableCallKind, CallableDescriptor,
};
use crate::call::shared::{
    build_object_paren_expr_selector_values, build_object_paren_selector_values,
    call_object_subsasgn_brace_values, call_object_subsasgn_paren_scalar_indices,
    call_object_subsasgn_paren_values, call_object_subsref_brace_values,
    call_object_subsref_paren_values,
};
use crate::indexing::end_expr as idx_end_expr;
use crate::indexing::plan::{build_expr_index_plan, build_index_plan, ExprPlanSpec};
use crate::indexing::read_linear as idx_read_linear;
use crate::indexing::read_slice as idx_read_slice;
use crate::indexing::selectors::{
    build_slice_selectors, index_scalar_from_value, indices_from_value_linear,
};
use crate::indexing::write_linear as idx_write_linear;
use crate::indexing::write_slice as idx_write_slice;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
use std::future::Future;
use std::pin::Pin;

fn map_slice_plan_error(context: &str, err: RuntimeError) -> RuntimeError {
    format!("{context}: {}", err.message()).into()
}

const CELL_END_PLUS_TAG_MASK: u64 = 0x7ff8_0000_0000_0000;
const CELL_END_PLUS_TAG_VALUE: u64 = 0x7ff8_c311_0000_0000;
const CELL_END_PLUS_OFFSET_MASK: u64 = 0x0000_0000_ffff_ffff;

fn decode_cell_end_plus(value: f64) -> Option<usize> {
    if !value.is_nan() {
        return None;
    }
    let bits = value.to_bits();
    if (bits & CELL_END_PLUS_TAG_MASK) != CELL_END_PLUS_TAG_VALUE {
        return None;
    }
    Some((bits & CELL_END_PLUS_OFFSET_MASK) as usize)
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

fn resolve_cell_indices(
    values: &[Value],
    rows: usize,
    cols: usize,
) -> Result<Vec<usize>, RuntimeError> {
    values
        .iter()
        .enumerate()
        .map(|(dim, value)| match value {
            Value::Num(index) => {
                let len = if values.len() == 1 {
                    rows * cols
                } else if dim == 0 {
                    rows
                } else {
                    cols
                };
                if let Some(offset) = decode_cell_end_plus(*index) {
                    let resolved = len + offset;
                    if resolved < 1 || resolved > len {
                        return Err(crate::interpreter::errors::mex(
                            "CellIndexOutOfBounds",
                            "Cell index out of bounds",
                        ));
                    }
                    return Ok(resolved);
                }
                if *index == 0.0 && index.is_sign_negative() {
                    return Ok(len);
                }
                if *index < 0.0 {
                    let resolved = len as isize + *index as isize;
                    if resolved < 1 || resolved as usize > len {
                        return Err(crate::interpreter::errors::mex(
                            "CellIndexOutOfBounds",
                            "Cell index out of bounds",
                        ));
                    }
                    return Ok(resolved as usize);
                }
                Ok(*index as usize)
            }
            _ => {
                let index: f64 = value.try_into()?;
                Ok(index as usize)
            }
        })
        .collect()
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

async fn execute_brace_read_single(
    base: Value,
    raw_indices: &[Value],
) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => {
            call_object_subsref_brace_values(Value::Object(obj), raw_indices.to_vec()).await
        }
        Value::HandleObject(handle) => {
            call_object_subsref_brace_values(Value::HandleObject(handle), raw_indices.to_vec())
                .await
        }
        Value::Cell(ca) => {
            let indices = resolve_cell_indices(raw_indices, ca.rows, ca.cols)?;
            crate::ops::cells::index_cell_value(&ca, &indices)
        }
        _ => Err(crate::interpreter::errors::mex(
            "CellIndexingOnNonCell",
            "Cell indexing on non-cell",
        )),
    }
}

async fn execute_brace_expand(
    base: Value,
    raw_indices: &[Value],
    out_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    match base {
        Value::Cell(ca) => {
            let mut values = if raw_indices.is_empty() {
                crate::ops::cells::expand_cell_values(&ca, &[], out_count)?
            } else {
                crate::call::shared::expand_cell_indices(&ca, raw_indices)?
            };
            if values.len() > out_count {
                values.truncate(out_count);
            } else {
                values.resize(out_count, Value::Num(0.0));
            }
            Ok(values)
        }
        Value::Object(obj) => {
            let value =
                call_object_subsref_brace_values(Value::Object(obj), raw_indices.to_vec()).await?;
            let mut out = vec![value];
            out.resize(out_count, Value::Num(0.0));
            Ok(out)
        }
        Value::HandleObject(handle) => {
            let value =
                call_object_subsref_brace_values(Value::HandleObject(handle), raw_indices.to_vec())
                    .await?;
            let mut out = vec![value];
            out.resize(out_count, Value::Num(0.0));
            Ok(out)
        }
        _ => Err(crate::interpreter::errors::mex(
            "CellExpansionOnNonCell",
            "Cell expansion on non-cell",
        )),
    }
}

async fn execute_brace_list(base: Value, raw_indices: &[Value]) -> Result<Value, RuntimeError> {
    match base {
        Value::Cell(ca) => {
            let values = if raw_indices.is_empty() {
                crate::ops::cells::expand_all_cell_values(&ca)?
            } else {
                crate::call::shared::expand_cell_indices(&ca, raw_indices)?
            };
            if values.len() == 1 {
                Ok(values.into_iter().next().unwrap_or(Value::Num(0.0)))
            } else {
                Ok(Value::OutputList(values))
            }
        }
        Value::Object(obj) => {
            let value =
                call_object_subsref_brace_values(Value::Object(obj), raw_indices.to_vec()).await?;
            Ok(Value::OutputList(vec![value]))
        }
        Value::HandleObject(handle) => {
            let value =
                call_object_subsref_brace_values(Value::HandleObject(handle), raw_indices.to_vec())
                    .await?;
            Ok(Value::OutputList(vec![value]))
        }
        _ => Err(crate::interpreter::errors::mex(
            "CellExpansionOnNonCell",
            "Cell expansion on non-cell",
        )),
    }
}

async fn execute_brace_store(
    base: Value,
    raw_indices: &[Value],
    rhs: Value,
) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => {
            call_object_subsasgn_brace_values(Value::Object(obj), raw_indices.to_vec(), rhs).await
        }
        Value::HandleObject(handle) => {
            call_object_subsasgn_brace_values(
                Value::HandleObject(handle),
                raw_indices.to_vec(),
                rhs,
            )
            .await
        }
        Value::Cell(ca) => {
            let indices = resolve_cell_indices(raw_indices, ca.rows, ca.cols)?;
            crate::ops::cells::assign_cell_value(ca, &indices, rhs, |oldv, newv| {
                runmat_gc::gc_record_write(oldv, newv);
            })
        }
        _ => Err(crate::interpreter::errors::mex(
            "CellAssignmentOnNonCell",
            "Cell assignment on non-cell",
        )),
    }
}

#[derive(Clone, Copy)]
struct IndexContext<'a> {
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    base_shape: &'a [usize],
}

impl<'a> IndexContext<'a> {
    fn new(dims: usize, colon_mask: u32, end_mask: u32, base_shape: &'a [usize]) -> Self {
        Self {
            dims,
            colon_mask,
            end_mask,
            base_shape,
        }
    }

    fn dim_len_for_numeric_position(&self, numeric_position: usize) -> usize {
        let mut seen_numeric = 0usize;
        let mut dim_for_pos = 0usize;
        for d in 0..self.dims {
            let is_colon = (self.colon_mask & (1u32 << d)) != 0;
            let is_end = (self.end_mask & (1u32 << d)) != 0;
            if is_colon || is_end {
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

fn apply_end_offsets_to_numeric<'a, F>(
    numeric: &'a [Value],
    ctx: IndexContext<'a>,
    end_offsets: &'a [(usize, EndExpr)],
    vars: &'a mut [Value],
    call_user: F,
) -> Pin<Box<dyn Future<Output = Result<Vec<Value>, RuntimeError>> + 'a>>
where
    F: Fn(
            &'a str,
            Vec<Value>,
            &'a [Value],
        ) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>>
        + Copy
        + 'a,
{
    Box::pin(async move {
        let mut adjusted = numeric.to_vec();
        for (position, end_expr) in end_offsets {
            if let Some(value) = adjusted.get_mut(*position) {
                let dim_len = ctx.dim_len_for_numeric_position(*position);
                let idx_val = resolve_range_end_index(dim_len, end_expr, vars, call_user).await?;
                *value = Value::Num(idx_val as f64);
            }
        }
        Ok(adjusted)
    })
}

async fn resolve_range_end_index<'a, F>(
    dim_len: usize,
    end_expr: &'a EndExpr,
    vars: &'a [Value],
    call_user: F,
) -> Result<i64, RuntimeError>
where
    F: Fn(
            &'a str,
            Vec<Value>,
            &'a [Value],
        ) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>>
        + Copy
        + 'a,
{
    fn eval_end_expr_value<'a, F>(
        expr: &'a EndExpr,
        end_value: f64,
        vars: &'a [Value],
        call_user: F,
    ) -> Pin<Box<dyn Future<Output = Result<f64, RuntimeError>> + 'a>>
    where
        F: Fn(
                &'a str,
                Vec<Value>,
                &'a [Value],
            ) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>>
            + Copy
            + 'a,
    {
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
                EndExpr::Call(name, args) => {
                    let mut argv: Vec<Value> = Vec::with_capacity(args.len());
                    for a in args {
                        let val = eval_end_expr_value(a, end_value, vars, call_user).await?;
                        argv.push(Value::Num(val));
                    }
                    let v = match runmat_runtime::call_builtin_async(name, &argv).await {
                        Ok(v) => v,
                        Err(_) => call_user(name, argv, vars).await?,
                    };
                    idx_end_expr::value_to_f64(&v).map_err(|_| {
                        crate::interpreter::errors::mex(
                            "UnsupportedIndexType",
                            "end call must return scalar",
                        )
                    })
                }
                EndExpr::SemanticCall(function, name, args) => {
                    let mut argv: Vec<Value> = Vec::with_capacity(args.len());
                    for a in args {
                        let val = eval_end_expr_value(a, end_value, vars, call_user).await?;
                        argv.push(Value::Num(val));
                    }
                    let descriptor = CallableDescriptor::semantic_named(
                        *function,
                        name.clone(),
                        argv.clone(),
                        1,
                    )
                    .with_call_kind(CallableCallKind::EndExpr);
                    let v = match try_execute_callable_descriptor(descriptor).await? {
                        Some(value) => value,
                        None => call_user(name, argv, vars).await?,
                    };
                    idx_end_expr::value_to_f64(&v).map_err(|_| {
                        crate::interpreter::errors::mex(
                            "UnsupportedIndexType",
                            "end call must return scalar",
                        )
                    })
                }
                EndExpr::Add(a, b) => Ok(eval_end_expr_value(a, end_value, vars, call_user)
                    .await?
                    + eval_end_expr_value(b, end_value, vars, call_user).await?),
                EndExpr::Sub(a, b) => Ok(eval_end_expr_value(a, end_value, vars, call_user)
                    .await?
                    - eval_end_expr_value(b, end_value, vars, call_user).await?),
                EndExpr::Mul(a, b) => Ok(eval_end_expr_value(a, end_value, vars, call_user)
                    .await?
                    * eval_end_expr_value(b, end_value, vars, call_user).await?),
                EndExpr::Div(a, b) => {
                    let denom = eval_end_expr_value(b, end_value, vars, call_user).await?;
                    if denom == 0.0 {
                        return Err(crate::interpreter::errors::mex(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                    Ok(eval_end_expr_value(a, end_value, vars, call_user).await? / denom)
                }
                EndExpr::LeftDiv(a, b) => {
                    let denom = eval_end_expr_value(a, end_value, vars, call_user).await?;
                    if denom == 0.0 {
                        return Err(crate::interpreter::errors::mex(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                    Ok(eval_end_expr_value(b, end_value, vars, call_user).await? / denom)
                }
                EndExpr::Pow(a, b) => Ok(eval_end_expr_value(a, end_value, vars, call_user)
                    .await?
                    .powf(eval_end_expr_value(b, end_value, vars, call_user).await?)),
                EndExpr::Neg(a) => Ok(-eval_end_expr_value(a, end_value, vars, call_user).await?),
                EndExpr::Pos(a) => Ok(eval_end_expr_value(a, end_value, vars, call_user).await?),
                EndExpr::Floor(a) => Ok(eval_end_expr_value(a, end_value, vars, call_user)
                    .await?
                    .floor()),
                EndExpr::Ceil(a) => Ok(eval_end_expr_value(a, end_value, vars, call_user)
                    .await?
                    .ceil()),
                EndExpr::Round(a) => Ok(eval_end_expr_value(a, end_value, vars, call_user)
                    .await?
                    .round()),
                EndExpr::Fix(a) => {
                    let v = eval_end_expr_value(a, end_value, vars, call_user).await?;
                    Ok(if v >= 0.0 { v.floor() } else { v.ceil() })
                }
            }
        })
    }

    Ok(
        eval_end_expr_value(end_expr, dim_len as f64, vars, call_user)
            .await?
            .floor() as i64,
    )
}

pub async fn dispatch_indexing<F>(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    vars: &mut Vec<Value>,
    semantic_registry: &crate::bytecode::SemanticFunctionRegistry,
    pc: usize,
    mut clear_value_residency: impl FnMut(&Value),
    call_user: F,
) -> Result<bool, RuntimeError>
where
    F: for<'b> Fn(
            &'b str,
            Vec<Value>,
            &'b [Value],
        ) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'b>>
        + Copy,
{
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
                    stack.push(call_object_subsref_paren_values(base, raw_indices.clone()).await?);
                }
                Value::FunctionHandle(_)
                | Value::SemanticFunctionHandle { .. }
                | Value::Closure(_) => {
                    let numeric = linear_index_values_to_f64(&raw_indices).await?;
                    let args = numeric.into_iter().map(Value::Num).collect::<Vec<_>>();
                    match crate::call::feval::execute_feval(base, args, 1, semantic_registry)
                        .await?
                    {
                        crate::call::feval::FevalDispatch::Completed(value) => stack.push(value),
                    }
                }
                Value::Tensor(t)
                    if raw_indices.len() == 1
                        && index_scalar_from_value(&raw_indices[0]).await?.is_none() =>
                {
                    stack.push(idx_read_slice::read_tensor_slice_1d(t, 0, 0, &raw_indices).await?);
                }
                _ => {
                    let numeric = linear_index_values_to_f64(&raw_indices).await?;
                    stack.push(idx_read_linear::generic_index(&base, &numeric).await?);
                }
            }
            Ok(true)
        }
        crate::bytecode::Instr::IndexCell(num_indices) => {
            let raw_indices = pop_index_values(stack, *num_indices)?;
            let base = pop_index_base(stack)?;
            stack.push(execute_brace_read_single(base, &raw_indices).await?);
            Ok(true)
        }
        crate::bytecode::Instr::IndexCellExpand(num_indices, out_count) => {
            let raw_indices = pop_index_values(stack, *num_indices)?;
            let base = pop_index_base(stack)?;
            for value in execute_brace_expand(base, &raw_indices, *out_count).await? {
                stack.push(value);
            }
            Ok(true)
        }
        crate::bytecode::Instr::IndexCellList(num_indices) => {
            let raw_indices = pop_index_values(stack, *num_indices)?;
            let base = pop_index_base(stack)?;
            stack.push(execute_brace_list(base, &raw_indices).await?);
            Ok(true)
        }
        crate::bytecode::Instr::StoreIndexCell(num_indices) => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let raw_indices = pop_index_values(stack, *num_indices)?;
            let base = pop_index_base(stack)?;
            stack.push(execute_brace_store(base, &raw_indices, rhs).await?);
            Ok(true)
        }
        crate::bytecode::Instr::StoreIndex(num_indices) => {
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
                    stack.push(
                        call_object_subsasgn_paren_scalar_indices(Value::Object(obj), indices, rhs)
                            .await?,
                    );
                }
                Value::HandleObject(handle) => {
                    stack.push(
                        call_object_subsasgn_paren_scalar_indices(
                            Value::HandleObject(handle),
                            indices,
                            rhs,
                        )
                        .await?,
                    );
                }
                Value::Tensor(t) => {
                    stack.push(idx_write_linear::assign_tensor_scalar(t, &indices, &rhs).await?)
                }
                Value::ComplexTensor(t) => {
                    stack.push(idx_write_linear::assign_complex_scalar(t, &indices, &rhs).await?)
                }
                Value::Cell(ca) => {
                    stack.push(crate::ops::cells::assign_cell_paren(ca, &indices, &rhs)?)
                }
                Value::Struct(st) => stack.push(assign_scalar_struct_index(st, &indices, rhs)?),
                Value::GpuTensor(h) => {
                    stack.push(idx_write_linear::assign_gpu_scalar(&h, &indices, &rhs).await?)
                }
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
                    let selectors = build_object_paren_selector_values(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )?;
                    stack.push(
                        call_object_subsref_paren_values(Value::Object(obj), selectors).await?,
                    );
                }
                Value::HandleObject(handle) => {
                    let selectors = build_object_paren_selector_values(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )?;
                    stack.push(
                        call_object_subsref_paren_values(Value::HandleObject(handle), selectors)
                            .await?,
                    );
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
                other => {
                    if *dims == 1 {
                        let is_colon = (*colon_mask & 1u32) != 0;
                        let is_end = (*end_mask & 1u32) != 0;
                        if is_colon {
                            return Err(crate::interpreter::errors::mex(
                                "SliceNonTensor",
                                "Slicing only supported on tensors",
                            ));
                        }
                        let linear_indices: Vec<f64> = if is_end {
                            vec![1.0]
                        } else {
                            let value = numeric.first().ok_or_else(|| {
                                crate::interpreter::errors::mex(
                                    "MissingNumericIndex",
                                    "missing numeric index for linear slice",
                                )
                            })?;
                            indices_from_value_linear(value, 1)
                                .await?
                                .into_iter()
                                .map(|idx| idx as f64)
                                .collect()
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
        crate::bytecode::Instr::StoreSlice(dims, numeric_count, colon_mask, end_mask) => {
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
                    let selectors = build_object_paren_selector_values(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )?;
                    stack.push(
                        call_object_subsasgn_paren_values(Value::Object(obj), selectors, rhs)
                            .await?,
                    );
                }
                Value::HandleObject(handle) => {
                    let selectors = build_object_paren_selector_values(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )?;
                    stack.push(
                        call_object_subsasgn_paren_values(
                            Value::HandleObject(handle),
                            selectors,
                            rhs,
                        )
                        .await?,
                    );
                }
                Value::Tensor(t) => {
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &t.shape)
                            .await?;
                    let plan = build_index_plan(&selectors, *dims, &t.shape)?;
                    stack.push(idx_write_slice::assign_tensor_with_plan(t, &plan, &rhs).await?);
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
                    idx_write_slice::assign_gpu_slice_with_plan(&handle, &plan, &rhs).await?
                }),
                Value::ComplexTensor(mut ct) => {
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &ct.shape)
                            .await
                            .map_err(|e| format!("slice assign: {e}"))?;
                    let plan = build_index_plan(&selectors, *dims, &ct.shape)
                        .map_err(|e| map_slice_plan_error("slice assign", e))?;
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
                    stack.push(crate::ops::cells::assign_cell_paren_linear_indices(
                        ca, &selected, &rhs,
                    )?);
                }
                Value::StringArray(mut sa) => {
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
                            IndexContext::new(*dims, *colon_mask, *end_mask, &handle.shape),
                            end_numeric_exprs,
                            vars,
                            call_user,
                        )
                        .await?
                    }
                    Value::Tensor(t) => {
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(*dims, *colon_mask, *end_mask, &t.shape),
                            end_numeric_exprs,
                            vars,
                            call_user,
                        )
                        .await?
                    }
                    Value::ComplexTensor(t) => {
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(*dims, *colon_mask, *end_mask, &t.shape),
                            end_numeric_exprs,
                            vars,
                            call_user,
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
                        let call_user_ref = call_user;
                        async move {
                            resolve_range_end_index(dim_len, &expr, vars_ref, call_user_ref).await
                        }
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
                            let call_user_ref = call_user;
                            async move {
                                resolve_range_end_index(dim_len, &expr, vars_ref, call_user_ref)
                                    .await
                            }
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
                            let call_user_ref = call_user;
                            async move {
                                resolve_range_end_index(dim_len, &expr, vars_ref, call_user_ref)
                                    .await
                            }
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
        } => {
            let mut rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
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
            let mut base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            clear_value_residency(&base);
            let base_assignable = matches!(
                base,
                Value::Object(_) | Value::Tensor(_) | Value::ComplexTensor(_) | Value::GpuTensor(_)
            );
            if !base_assignable
                && matches!(
                    rhs,
                    Value::Object(_)
                        | Value::Tensor(_)
                        | Value::ComplexTensor(_)
                        | Value::GpuTensor(_)
                )
            {
                std::mem::swap(&mut base, &mut rhs);
            }
            if !end_numeric_exprs.is_empty() {
                numeric = match &base {
                    Value::GpuTensor(handle) => {
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(*dims, *colon_mask, *end_mask, &handle.shape),
                            end_numeric_exprs,
                            vars,
                            call_user,
                        )
                        .await?
                    }
                    Value::Tensor(t) => {
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(*dims, *colon_mask, *end_mask, &t.shape),
                            end_numeric_exprs,
                            vars,
                            call_user,
                        )
                        .await?
                    }
                    Value::ComplexTensor(t) => {
                        apply_end_offsets_to_numeric(
                            &numeric,
                            IndexContext::new(*dims, *colon_mask, *end_mask, &t.shape),
                            end_numeric_exprs,
                            vars,
                            call_user,
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
                            async move {
                                resolve_range_end_index(dim_len, &expr, vars_ref, call_user).await
                            }
                        },
                    )
                    .await?;
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
                            async move {
                                resolve_range_end_index(dim_len, &expr, vars_ref, call_user).await
                            }
                        },
                    )
                    .await?;
                    stack.push(idx_write_slice::assign_tensor_with_plan(t, &vm_plan, &rhs).await?);
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
                            async move {
                                resolve_range_end_index(dim_len, &expr, vars_ref, call_user).await
                            }
                        },
                    )
                    .await?;
                    let updated =
                        idx_write_slice::assign_gpu_slice_with_plan(&h, &vm_plan, &rhs).await?;
                    stack.push(updated);
                }
                Value::Object(obj) => {
                    let idx_values = build_object_paren_expr_selector_values(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        range_dims,
                        &range_params,
                        range_start_exprs,
                        range_step_exprs,
                        range_end_exprs,
                        &numeric,
                    )?;
                    stack.push(
                        call_object_subsasgn_paren_values(Value::Object(obj), idx_values, rhs)
                            .await?,
                    );
                }
                _ => {
                    return Err("StoreSliceExpr only supports tensors currently"
                        .to_string()
                        .into())
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
