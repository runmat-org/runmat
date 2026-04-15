use crate::bytecode::{EndExpr, UserFunction};
use crate::indexing::end_expr as idx_end_expr;
use crate::indexing::plan as idx_plan;
use crate::indexing::plan::{build_expr_index_plan, build_index_plan, ExprPlanSpec};
use crate::indexing::read_linear as idx_read_linear;
use crate::indexing::read_slice as idx_read_slice;
use crate::indexing::selectors::build_slice_selectors;
use crate::indexing::selectors::index_scalar_from_value;
use crate::indexing::write_linear as idx_write_linear;
use crate::indexing::write_slice as idx_write_slice;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

fn map_slice_plan_error(context: &str, err: RuntimeError) -> RuntimeError {
    format!("{context}: {}", err.message()).into()
}

fn total_len_from_shape(shape: &[usize]) -> usize {
    if runmat_runtime::builtins::common::shape::is_scalar_shape(shape) {
        1
    } else {
        shape.iter().copied().product()
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
    functions: &'a HashMap<String, UserFunction>,
    call_user: F,
) -> Pin<Box<dyn Future<Output = Result<Vec<Value>, RuntimeError>> + 'a>>
where
    F: Fn(
            &'a str,
            Vec<Value>,
            &'a HashMap<String, UserFunction>,
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
                let idx_val =
                    resolve_range_end_index(dim_len, end_expr, vars, functions, call_user).await?;
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
    functions: &'a HashMap<String, UserFunction>,
    call_user: F,
) -> Result<i64, RuntimeError>
where
    F: Fn(
            &'a str,
            Vec<Value>,
            &'a HashMap<String, UserFunction>,
            &'a [Value],
        ) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>>
        + Copy
        + 'a,
{
    fn eval_end_expr_value<'a, F>(
        expr: &'a EndExpr,
        end_value: f64,
        vars: &'a [Value],
        functions: &'a HashMap<String, UserFunction>,
        call_user: F,
    ) -> Pin<Box<dyn Future<Output = Result<f64, RuntimeError>> + 'a>>
    where
        F: Fn(
                &'a str,
                Vec<Value>,
                &'a HashMap<String, UserFunction>,
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
                        let val =
                            eval_end_expr_value(a, end_value, vars, functions, call_user).await?;
                        argv.push(Value::Num(val));
                    }
                    let v = match runmat_runtime::call_builtin_async(name, &argv).await {
                        Ok(v) => v,
                        Err(_) if functions.contains_key(name) => {
                            call_user(name, argv, functions, vars).await?
                        }
                        Err(_) => {
                            return Err(crate::interpreter::errors::mex(
                                "UndefinedFunction",
                                &format!("Undefined function in end expression: {name}"),
                            ))
                        }
                    };
                    idx_end_expr::value_to_f64(&v).map_err(|_| {
                        crate::interpreter::errors::mex(
                            "UnsupportedIndexType",
                            "end call must return scalar",
                        )
                    })
                }
                EndExpr::Add(a, b) => Ok(eval_end_expr_value(
                    a, end_value, vars, functions, call_user,
                )
                .await?
                    + eval_end_expr_value(b, end_value, vars, functions, call_user).await?),
                EndExpr::Sub(a, b) => Ok(eval_end_expr_value(
                    a, end_value, vars, functions, call_user,
                )
                .await?
                    - eval_end_expr_value(b, end_value, vars, functions, call_user).await?),
                EndExpr::Mul(a, b) => Ok(eval_end_expr_value(
                    a, end_value, vars, functions, call_user,
                )
                .await?
                    * eval_end_expr_value(b, end_value, vars, functions, call_user).await?),
                EndExpr::Div(a, b) => {
                    let denom =
                        eval_end_expr_value(b, end_value, vars, functions, call_user).await?;
                    if denom == 0.0 {
                        return Err(crate::interpreter::errors::mex(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                    Ok(
                        eval_end_expr_value(a, end_value, vars, functions, call_user).await?
                            / denom,
                    )
                }
                EndExpr::LeftDiv(a, b) => {
                    let denom =
                        eval_end_expr_value(a, end_value, vars, functions, call_user).await?;
                    if denom == 0.0 {
                        return Err(crate::interpreter::errors::mex(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                    Ok(
                        eval_end_expr_value(b, end_value, vars, functions, call_user).await?
                            / denom,
                    )
                }
                EndExpr::Pow(a, b) => Ok(eval_end_expr_value(
                    a, end_value, vars, functions, call_user,
                )
                .await?
                .powf(eval_end_expr_value(b, end_value, vars, functions, call_user).await?)),
                EndExpr::Neg(a) => {
                    Ok(-eval_end_expr_value(a, end_value, vars, functions, call_user).await?)
                }
                EndExpr::Pos(a) => {
                    Ok(eval_end_expr_value(a, end_value, vars, functions, call_user).await?)
                }
                EndExpr::Floor(a) => Ok(eval_end_expr_value(
                    a, end_value, vars, functions, call_user,
                )
                .await?
                .floor()),
                EndExpr::Ceil(a) => Ok(eval_end_expr_value(
                    a, end_value, vars, functions, call_user,
                )
                .await?
                .ceil()),
                EndExpr::Round(a) => Ok(eval_end_expr_value(
                    a, end_value, vars, functions, call_user,
                )
                .await?
                .round()),
                EndExpr::Fix(a) => {
                    let v = eval_end_expr_value(a, end_value, vars, functions, call_user).await?;
                    Ok(if v >= 0.0 { v.floor() } else { v.ceil() })
                }
            }
        })
    }

    Ok(
        eval_end_expr_value(end_expr, dim_len as f64, vars, functions, call_user)
            .await?
            .floor() as i64,
    )
}

fn encode_end_expr_value(expr: &EndExpr) -> Result<Value, RuntimeError> {
    fn mk_cell(items: Vec<Value>) -> Result<Value, RuntimeError> {
        let cols = items.len();
        let cell = runmat_builtins::CellArray::new(items, 1, cols)
            .map_err(|e| format!("end expression encoding: {e}"))?;
        Ok(Value::Cell(cell))
    }

    match expr {
        EndExpr::End => Ok(Value::String("end".to_string())),
        EndExpr::Const(v) => Ok(Value::Num(*v)),
        EndExpr::Var(i) => Ok(Value::String(format!("var:{i}"))),
        EndExpr::Call(name, args) => {
            let mut items = vec![
                Value::String("call".to_string()),
                Value::String(name.clone()),
            ];
            for a in args {
                items.push(encode_end_expr_value(a)?);
            }
            mk_cell(items)
        }
        EndExpr::Add(a, b) => mk_cell(vec![
            Value::String("+".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Sub(a, b) => mk_cell(vec![
            Value::String("-".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Mul(a, b) => mk_cell(vec![
            Value::String("*".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Div(a, b) => mk_cell(vec![
            Value::String("/".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::LeftDiv(a, b) => mk_cell(vec![
            Value::String("\\".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Pow(a, b) => mk_cell(vec![
            Value::String("^".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Neg(a) => mk_cell(vec![
            Value::String("neg".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Pos(a) => mk_cell(vec![
            Value::String("pos".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Floor(a) => mk_cell(vec![
            Value::String("floor".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Ceil(a) => mk_cell(vec![
            Value::String("ceil".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Round(a) => mk_cell(vec![
            Value::String("round".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Fix(a) => mk_cell(vec![
            Value::String("fix".to_string()),
            encode_end_expr_value(a)?,
        ]),
    }
}

fn build_end_range_descriptor(
    start: Value,
    step: Value,
    end_expr: &EndExpr,
) -> Result<Value, RuntimeError> {
    let encoded_end = encode_end_expr_value(end_expr)?;
    let cell = runmat_builtins::CellArray::new(
        vec![
            start,
            step,
            Value::String("end_expr".to_string()),
            encoded_end,
        ],
        1,
        4,
    )
    .map_err(|e| format!("obj range: {e}"))?;
    Ok(Value::Cell(cell))
}

pub async fn dispatch_indexing<F>(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    vars: &mut Vec<Value>,
    functions: &HashMap<String, UserFunction>,
    pc: usize,
    mut clear_value_residency: impl FnMut(&Value),
    call_user: F,
) -> Result<bool, RuntimeError>
where
    F: for<'b> Fn(
            &'b str,
            Vec<Value>,
            &'b HashMap<String, UserFunction>,
            &'b [Value],
        ) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'b>>
        + Copy,
{
    match instr {
        crate::bytecode::Instr::Index(num_indices) => {
            let numeric = idx_read_linear::collect_linear_indices(stack, *num_indices).await?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            match &base {
                Value::Object(_) | Value::HandleObject(_) => {
                    let cell = idx_read_linear::build_object_subsref_cell(&numeric)?;
                    let args = vec![
                        base,
                        Value::String("subsref".to_string()),
                        Value::String("()".to_string()),
                        cell,
                    ];
                    stack.push(runmat_runtime::call_builtin_async("call_method", &args).await?);
                }
                _ => {
                    stack.push(idx_read_linear::generic_index(&base, &numeric).await?);
                }
            }
            Ok(true)
        }
        crate::bytecode::Instr::IndexCell(num_indices) => {
            let mut indices = Vec::with_capacity(*num_indices);
            for _ in 0..*num_indices {
                let v: f64 = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?)
                    .try_into()?;
                indices.push(v as usize);
            }
            indices.reverse();
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            match base {
                Value::Object(obj) => {
                    let cell = runmat_runtime::call_builtin_async(
                        "__make_cell",
                        &indices
                            .iter()
                            .map(|n| Value::Num(*n as f64))
                            .collect::<Vec<_>>(),
                    )
                    .await?;
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsref".to_string()),
                        Value::String("{}".to_string()),
                        cell,
                    ];
                    stack.push(runmat_runtime::call_builtin_async("call_method", &args).await?);
                }
                Value::HandleObject(handle) => {
                    let cell = runmat_runtime::call_builtin_async(
                        "__make_cell",
                        &indices
                            .iter()
                            .map(|n| Value::Num(*n as f64))
                            .collect::<Vec<_>>(),
                    )
                    .await?;
                    let args = vec![
                        Value::HandleObject(handle),
                        Value::String("subsref".to_string()),
                        Value::String("{}".to_string()),
                        cell,
                    ];
                    stack.push(runmat_runtime::call_builtin_async("call_method", &args).await?);
                }
                Value::Cell(ca) => stack.push(crate::ops::cells::index_cell_value(&ca, &indices)?),
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "CellIndexingOnNonCell",
                        "Cell indexing on non-cell",
                    ))
                }
            }
            Ok(true)
        }
        crate::bytecode::Instr::IndexCellExpand(num_indices, out_count) => {
            let mut indices = Vec::with_capacity(*num_indices);
            if *num_indices > 0 {
                for _ in 0..*num_indices {
                    let v: f64 = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                        "StackUnderflow",
                        "stack underflow",
                    ))?)
                        .try_into()?;
                    indices.push(v as usize);
                }
                indices.reverse();
            }
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            match base {
                Value::Cell(ca) => {
                    let values = crate::ops::cells::expand_cell_values(&ca, &indices, *out_count)?;
                    for v in values {
                        stack.push(v);
                    }
                }
                Value::Object(obj) => {
                    let cell = runmat_runtime::call_builtin_async(
                        "__make_cell",
                        &indices
                            .iter()
                            .map(|n| Value::Num(*n as f64))
                            .collect::<Vec<_>>(),
                    )
                    .await?;
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsref".to_string()),
                        Value::String("{}".to_string()),
                        cell,
                    ];
                    let v = runmat_runtime::call_builtin_async("call_method", &args).await?;
                    stack.push(v);
                    for _ in 1..*out_count {
                        stack.push(Value::Num(0.0));
                    }
                }
                Value::HandleObject(handle) => {
                    let cell = runmat_runtime::call_builtin_async(
                        "__make_cell",
                        &indices
                            .iter()
                            .map(|n| Value::Num(*n as f64))
                            .collect::<Vec<_>>(),
                    )
                    .await?;
                    let args = vec![
                        Value::HandleObject(handle),
                        Value::String("subsref".to_string()),
                        Value::String("{}".to_string()),
                        cell,
                    ];
                    let v = runmat_runtime::call_builtin_async("call_method", &args).await?;
                    stack.push(v);
                    for _ in 1..*out_count {
                        stack.push(Value::Num(0.0));
                    }
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "CellExpansionOnNonCell",
                        "Cell expansion on non-cell",
                    ))
                }
            }
            Ok(true)
        }
        crate::bytecode::Instr::StoreIndexCell(num_indices) => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let mut indices = Vec::new();
            for _ in 0..*num_indices {
                let v: f64 = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?)
                    .try_into()?;
                indices.push(v as usize);
            }
            indices.reverse();
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            match base {
                Value::Object(obj) => {
                    let cell = runmat_builtins::CellArray::new(
                        indices.iter().map(|n| Value::Num(*n as f64)).collect(),
                        1,
                        indices.len(),
                    )
                    .map_err(|e| format!("subsasgn build error: {e}"))?;
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsasgn".to_string()),
                        Value::String("{}".to_string()),
                        Value::Cell(cell),
                        rhs,
                    ];
                    stack.push(runmat_runtime::call_builtin_async("call_method", &args).await?);
                }
                Value::HandleObject(handle) => {
                    let cell = runmat_builtins::CellArray::new(
                        indices.iter().map(|n| Value::Num(*n as f64)).collect(),
                        1,
                        indices.len(),
                    )
                    .map_err(|e| format!("subsasgn build error: {e}"))?;
                    let args = vec![
                        Value::HandleObject(handle),
                        Value::String("subsasgn".to_string()),
                        Value::String("{}".to_string()),
                        Value::Cell(cell),
                        rhs,
                    ];
                    stack.push(runmat_runtime::call_builtin_async("call_method", &args).await?);
                }
                Value::Cell(ca) => {
                    let updated =
                        crate::ops::cells::assign_cell_value(ca, &indices, rhs, |oldv, newv| {
                            runmat_gc::gc_record_write(oldv, newv);
                        })?;
                    stack.push(updated);
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "CellAssignmentOnNonCell",
                        "Cell assignment on non-cell",
                    ))
                }
            }
            Ok(true)
        }
        crate::bytecode::Instr::StoreIndex(num_indices) => {
            if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                let snap = stack
                    .iter()
                    .rev()
                    .take(6)
                    .map(|v| match v {
                        Value::Object(_) => "Object",
                        Value::HandleObject(_) => "HandleObject",
                        Value::Tensor(t) => {
                            log::debug!("[vm] StoreIndex pre-snap tensor shape={:?}", t.shape);
                            "Tensor"
                        }
                        Value::GpuTensor(h) => {
                            log::debug!("[vm] StoreIndex pre-snap GPU tensor shape={:?}", h.shape);
                            "GpuTensor"
                        }
                        Value::Num(_) => "Num",
                        Value::Int(_) => "Int",
                        Value::String(_) => "String",
                        Value::Cell(_) => "Cell",
                        _ => "Other",
                    })
                    .collect::<Vec<_>>();
                log::debug!(
                    "[vm] StoreIndex pre-snap pc={} stack_top_types={:?}",
                    pc,
                    snap
                );
            }
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let assignable = |v: &Value| {
                matches!(
                    v,
                    Value::Object(_)
                        | Value::HandleObject(_)
                        | Value::Tensor(_)
                        | Value::ComplexTensor(_)
                        | Value::GpuTensor(_)
                )
            };
            let base_idx_opt = (0..stack.len()).rev().find(|&j| assignable(&stack[j]));
            let base_pos = if let Some(j) = base_idx_opt {
                j
            } else {
                return Err(crate::interpreter::errors::mex(
                    "IndexAssignmentUnsupportedBase",
                    "Index assignment only for tensors",
                ));
            };
            let base = stack.remove(base_pos);
            clear_value_residency(&base);
            let mut indices: Vec<usize> = Vec::new();
            if *num_indices > 0 {
                let mut contiguous_ok = true;
                if base_pos + *num_indices > stack.len() {
                    contiguous_ok = false;
                } else {
                    for k in 0..*num_indices {
                        let idx_pos = base_pos + k;
                        let idx_val = match index_scalar_from_value(&stack[idx_pos]).await {
                            Ok(Some(val)) => val,
                            Ok(None) => {
                                contiguous_ok = false;
                                indices.clear();
                                break;
                            }
                            Err(err) => return Err(err),
                        };
                        let idx_val = if idx_val <= 0 { 0 } else { idx_val as usize };
                        indices.push(idx_val);
                    }
                }
                if contiguous_ok {
                    for k in (0..*num_indices).rev() {
                        stack.remove(base_pos + k);
                    }
                } else {
                    indices.clear();
                }
            }
            let (rows_opt, cols_opt) = match &base {
                Value::Tensor(t) => (Some(t.rows()), Some(t.cols())),
                Value::GpuTensor(h) => (
                    Some(h.shape.first().copied().unwrap_or(1).max(1)),
                    Some(h.shape.get(1).copied().unwrap_or(1).max(1)),
                ),
                _ => (None, None),
            };
            if indices.is_empty() {
                let mut numeric_above: Vec<(usize, usize)> = Vec::new();
                let mut scan_limit = 12usize;
                let mut kk = stack.len();
                while kk > 0 && scan_limit > 0 {
                    let idx = kk - 1;
                    if assignable(&stack[idx]) {
                        break;
                    }
                    if let Some(v) = index_scalar_from_value(&stack[idx]).await? {
                        let v = if v <= 0 { 0 } else { v as usize };
                        numeric_above.push((idx, v));
                    }
                    kk -= 1;
                    scan_limit -= 1;
                }
                if numeric_above.len() >= 2 {
                    let mut picked: Option<((usize, usize), (usize, usize))> = None;
                    for w in (1..numeric_above.len()).rev() {
                        let (j_idx, j_val) = numeric_above[w];
                        let (i_idx, i_val) = numeric_above[w - 1];
                        let fits = match (rows_opt, cols_opt) {
                            (Some(r), Some(c)) => {
                                i_val >= 1 && i_val <= r && j_val >= 1 && j_val <= c
                            }
                            _ => true,
                        };
                        if fits {
                            picked = Some(((i_idx, i_val), (j_idx, j_val)));
                            break;
                        }
                    }
                    if let Some(((i_idx, i_val), (j_idx, j_val))) = picked {
                        let mut to_remove = [i_idx, j_idx];
                        to_remove.sort_unstable();
                        stack.remove(to_remove[1]);
                        stack.remove(to_remove[0]);
                        indices = vec![i_val, j_val];
                    }
                } else if numeric_above.len() == 1 {
                    let (k_idx, k_val) = numeric_above[0];
                    stack.remove(k_idx);
                    indices = vec![k_val];
                }
            }
            if indices.is_empty() {
                return Err(crate::interpreter::errors::mex(
                    "IndexAssignmentUnsupportedBase",
                    "Index assignment only for tensors",
                ));
            }
            match base {
                Value::Object(obj) => {
                    let cell = runmat_runtime::call_builtin_async(
                        "__make_cell",
                        &indices
                            .iter()
                            .map(|n| Value::Num(*n as f64))
                            .collect::<Vec<_>>(),
                    )
                    .await?;
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsasgn".to_string()),
                        Value::String("()".to_string()),
                        cell,
                        rhs,
                    ];
                    stack.push(runmat_runtime::call_builtin_async("call_method", &args).await?);
                }
                Value::HandleObject(handle) => {
                    let cell = runmat_runtime::call_builtin_async(
                        "__make_cell",
                        &indices
                            .iter()
                            .map(|n| Value::Num(*n as f64))
                            .collect::<Vec<_>>(),
                    )
                    .await?;
                    let args = vec![
                        Value::HandleObject(handle),
                        Value::String("subsasgn".to_string()),
                        Value::String("()".to_string()),
                        cell,
                        rhs,
                    ];
                    stack.push(runmat_runtime::call_builtin_async("call_method", &args).await?);
                }
                Value::Tensor(t) => {
                    stack.push(idx_write_linear::assign_tensor_scalar(t, &indices, &rhs).await?)
                }
                Value::ComplexTensor(t) => {
                    stack.push(idx_write_linear::assign_complex_scalar(t, &indices, &rhs).await?)
                }
                Value::GpuTensor(h) => {
                    stack.push(idx_write_linear::assign_gpu_scalar(&h, &indices, &rhs).await?)
                }
                _ => {
                    if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                        let kind = |v: &Value| match v {
                            Value::Object(_) => "Object",
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
                Value::Object(obj) => stack.push(
                    idx_read_slice::object_subsref_paren(Value::Object(obj), &numeric).await?,
                ),
                Value::HandleObject(handle) => stack.push(
                    idx_read_slice::object_subsref_paren(Value::HandleObject(handle), &numeric)
                        .await?,
                ),
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
                        let idx_val: f64 = if is_end {
                            1.0
                        } else {
                            match numeric.first() {
                                Some(Value::Num(n)) => *n,
                                Some(Value::Int(i)) => i.to_f64(),
                                _ => 1.0,
                            }
                        };
                        let v = runmat_runtime::perform_indexing(&other, &[idx_val])
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
                    match idx_write_slice::object_subsasgn_paren(
                        Value::Object(obj.clone()),
                        &numeric,
                        rhs.clone(),
                    )
                    .await
                    {
                        Ok(v) => stack.push(v),
                        Err(_e) => {
                            let qualified = format!("{}.subsasgn", obj.class_name);
                            let cell = idx_write_slice::build_subsasgn_paren_cell(&numeric)?;
                            let args = vec![
                                Value::Object(obj),
                                Value::String("()".to_string()),
                                cell,
                                rhs,
                            ];
                            stack
                                .push(runmat_runtime::call_builtin_async(&qualified, &args).await?);
                        }
                    }
                }
                Value::HandleObject(handle) => stack.push(
                    idx_write_slice::object_subsasgn_paren(
                        Value::HandleObject(handle),
                        &numeric,
                        rhs,
                    )
                    .await?,
                ),
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
                            functions,
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
                            functions,
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
                            functions,
                            call_user,
                        )
                        .await?
                    }
                    _ => numeric,
                };
            }
            if let Value::GpuTensor(handle) = &base {
                let attempt = async {
                    let rank = handle.shape.len();
                    #[derive(Clone)]
                    enum Sel {
                        Colon,
                        Scalar(usize),
                        Indices(Vec<usize>),
                        Range {
                            start: i64,
                            step: i64,
                            end_off: EndExpr,
                        },
                    }
                    let full_shape: Vec<usize> = if *dims == 1 {
                        vec![total_len_from_shape(&handle.shape)]
                    } else if rank < *dims {
                        let mut s = handle.shape.clone();
                        s.resize(*dims, 1);
                        s
                    } else {
                        handle.shape.clone()
                    };
                    let mut selectors: Vec<Sel> = Vec::with_capacity(*dims);
                    let mut num_iter = 0usize;
                    let mut rp_iter = 0usize;
                    for d in 0..*dims {
                        let is_colon = (*colon_mask & (1u32 << d)) != 0;
                        let is_end = (*end_mask & (1u32 << d)) != 0;
                        if is_colon {
                            selectors.push(Sel::Colon);
                        } else if is_end {
                            selectors.push(Sel::Scalar(*full_shape.get(d).unwrap_or(&1)));
                        } else if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                            let (raw_st, raw_sp) = range_params[rp_iter];
                            let dim_len = *full_shape.get(d).unwrap_or(&1);
                            let st = if let Some(expr) = &range_start_exprs[rp_iter] {
                                resolve_range_end_index(dim_len, expr, &*vars, functions, call_user)
                                    .await? as f64
                            } else {
                                raw_st
                            };
                            let sp = if let Some(expr) = &range_step_exprs[rp_iter] {
                                resolve_range_end_index(dim_len, expr, &*vars, functions, call_user)
                                    .await? as f64
                            } else {
                                raw_sp
                            };
                            rp_iter += 1;
                            let off = range_end_exprs[pos].clone();
                            selectors.push(Sel::Range {
                                start: st as i64,
                                step: if sp >= 0.0 {
                                    sp as i64
                                } else {
                                    -(sp.abs() as i64)
                                },
                                end_off: off,
                            });
                        } else {
                            let v =
                                numeric
                                    .get(num_iter)
                                    .ok_or(crate::interpreter::errors::mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                            num_iter += 1;
                            if let Value::Int(idx_val) = v {
                                let idx = idx_val.to_i64();
                                if idx < 1 {
                                    return Err(crate::interpreter::errors::mex(
                                        "IndexOutOfBounds",
                                        "Index out of bounds",
                                    ));
                                }
                                selectors.push(Sel::Scalar(idx as usize));
                            } else {
                                match v {
                                    Value::Tensor(idx_t) => {
                                        let dim_len = *full_shape.get(d).unwrap_or(&1);
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == dim_len {
                                            let mut vv = Vec::new();
                                            for (i, &val) in idx_t.data.iter().enumerate() {
                                                if val != 0.0 {
                                                    vv.push(i + 1);
                                                }
                                            }
                                            selectors.push(Sel::Indices(vv));
                                        } else {
                                            let mut vv = Vec::with_capacity(len);
                                            for &val in &idx_t.data {
                                                let idx = val as isize;
                                                if idx < 1 {
                                                    return Err(crate::interpreter::errors::mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds",
                                                    ));
                                                }
                                                vv.push(idx as usize);
                                            }
                                            selectors.push(Sel::Indices(vv));
                                        }
                                    }
                                    _ => {
                                        return Err(crate::interpreter::errors::mex(
                                            "UnsupportedIndexType",
                                            "Unsupported index type",
                                        ))
                                    }
                                }
                            }
                        }
                    }
                    let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(*dims);
                    for (d, sel) in selectors.iter().enumerate().take(*dims) {
                        let dim_len = *full_shape.get(d).unwrap_or(&1) as i64;
                        let idxs: Vec<usize> = match sel {
                            Sel::Colon => (1..=dim_len as usize).collect(),
                            Sel::Scalar(i) => vec![*i],
                            Sel::Indices(v) => v.clone(),
                            Sel::Range {
                                start,
                                step,
                                end_off,
                            } => {
                                let mut vv = Vec::new();
                                let mut cur = *start;
                                let end_i = resolve_range_end_index(
                                    dim_len as usize,
                                    end_off,
                                    &*vars,
                                    functions,
                                    call_user,
                                )
                                .await?;
                                if *step == 0 {
                                    return Err(crate::interpreter::errors::mex(
                                        "IndexStepZero",
                                        "Index step cannot be zero",
                                    ));
                                }
                                if *step > 0 {
                                    while cur <= end_i {
                                        if cur < 1 || cur > dim_len {
                                            break;
                                        }
                                        vv.push(cur as usize);
                                        cur += *step;
                                    }
                                } else {
                                    while cur >= end_i {
                                        if cur < 1 || cur > dim_len {
                                            break;
                                        }
                                        vv.push(cur as usize);
                                        cur += *step;
                                    }
                                }
                                vv
                            }
                        };
                        if idxs.iter().any(|&i| i == 0 || i > dim_len as usize) {
                            return Err(crate::interpreter::errors::mex(
                                "IndexOutOfBounds",
                                "Index out of bounds",
                            ));
                        }
                        per_dim_indices.push(idxs);
                    }
                    let total_out: usize = per_dim_indices.iter().map(|v| v.len()).product();
                    if total_out == 0 {
                        return Ok((Vec::new(), vec![0, 0]));
                    }
                    let mut strides: Vec<usize> = vec![0; *dims];
                    let mut acc = 1usize;
                    for (d, stride) in strides.iter_mut().enumerate().take(*dims) {
                        *stride = acc;
                        acc *= full_shape[d];
                    }
                    let mut indices: Vec<u32> = Vec::with_capacity(total_out);
                    let mut idx = vec![0usize; *dims];
                    loop {
                        let mut lin = 0usize;
                        for d in 0..*dims {
                            let i0 = per_dim_indices[d][idx[d]] - 1;
                            lin += i0 * strides[d];
                        }
                        indices.push(lin as u32);
                        let mut d = 0usize;
                        while d < *dims {
                            idx[d] += 1;
                            if idx[d] < per_dim_indices[d].len() {
                                break;
                            }
                            idx[d] = 0;
                            d += 1;
                        }
                        if d == *dims {
                            break;
                        }
                    }
                    let output_shape = if *dims == 1 {
                        if total_out <= 1 {
                            vec![1, 1]
                        } else {
                            vec![total_out, 1]
                        }
                    } else {
                        per_dim_indices.iter().map(|v| v.len().max(1)).collect()
                    };
                    Ok((indices, output_shape))
                }
                .await;

                if let Ok((indices, output_shape)) = attempt {
                    let vm_plan = idx_plan::IndexPlan::new(
                        indices,
                        output_shape,
                        Vec::new(),
                        *dims,
                        handle.shape.clone(),
                    );
                    if let Ok(result) = idx_read_slice::read_gpu_slice_from_plan(handle, &vm_plan) {
                        stack.push(result);
                        return Ok(true);
                    }
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
                            let functions_ref = functions;
                            let call_user_ref = call_user;
                            async move {
                                resolve_range_end_index(
                                    dim_len,
                                    &expr,
                                    vars_ref,
                                    functions_ref,
                                    call_user_ref,
                                )
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
                            let functions_ref = functions;
                            let call_user_ref = call_user;
                            async move {
                                resolve_range_end_index(
                                    dim_len,
                                    &expr,
                                    vars_ref,
                                    functions_ref,
                                    call_user_ref,
                                )
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
                            functions,
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
                            functions,
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
                            functions,
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
                            let functions_ref = functions;
                            async move {
                                resolve_range_end_index(
                                    dim_len,
                                    &expr,
                                    vars_ref,
                                    functions_ref,
                                    call_user,
                                )
                                .await
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
                            let functions_ref = functions;
                            async move {
                                resolve_range_end_index(
                                    dim_len,
                                    &expr,
                                    vars_ref,
                                    functions_ref,
                                    call_user,
                                )
                                .await
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
                            let functions_ref = functions;
                            async move {
                                resolve_range_end_index(
                                    dim_len,
                                    &expr,
                                    vars_ref,
                                    functions_ref,
                                    call_user,
                                )
                                .await
                            }
                        },
                    )
                    .await?;
                    let updated =
                        idx_write_slice::assign_gpu_slice_with_plan(&h, &vm_plan, &rhs).await?;
                    stack.push(updated);
                }
                Value::Object(obj) => {
                    let mut idx_values: Vec<Value> = Vec::with_capacity(*dims);
                    let mut num_iter = 0usize;
                    let mut rp_iter = 0usize;
                    for d in 0..*dims {
                        let is_colon = (*colon_mask & (1u32 << d)) != 0;
                        let is_end = (*end_mask & (1u32 << d)) != 0;
                        if is_colon {
                            idx_values.push(Value::String(":".to_string()));
                            continue;
                        }
                        if is_end {
                            idx_values.push(Value::String("end".to_string()));
                            continue;
                        }
                        if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                            let (raw_st, raw_sp) = range_params[rp_iter];
                            let st = if let Some(expr) = &range_start_exprs[rp_iter] {
                                encode_end_expr_value(expr)?
                            } else {
                                Value::Num(raw_st)
                            };
                            let sp = if let Some(expr) = &range_step_exprs[rp_iter] {
                                encode_end_expr_value(expr)?
                            } else {
                                Value::Num(raw_sp)
                            };
                            rp_iter += 1;
                            let off = range_end_exprs[pos].clone();
                            idx_values.push(build_end_range_descriptor(st, sp, &off)?);
                        } else {
                            let v =
                                numeric
                                    .get(num_iter)
                                    .ok_or(crate::interpreter::errors::mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                            num_iter += 1;
                            match v {
                                Value::Num(n) => idx_values.push(Value::Num(*n)),
                                Value::Int(i) => idx_values.push(Value::Num(i.to_f64())),
                                Value::Tensor(t) => idx_values.push(Value::Tensor(t.clone())),
                                other => {
                                    return Err(format!(
                                        "Unsupported index type for object: {other:?}"
                                    )
                                    .into())
                                }
                            }
                        }
                    }
                    let cell = runmat_builtins::CellArray::new(idx_values, 1, *dims)
                        .map_err(|e| format!("subsasgn build error: {e}"))?;
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsasgn".to_string()),
                        Value::String("()".to_string()),
                        Value::Cell(cell),
                        rhs,
                    ];
                    stack.push(runmat_runtime::call_builtin_async("call_method", &args).await?);
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
