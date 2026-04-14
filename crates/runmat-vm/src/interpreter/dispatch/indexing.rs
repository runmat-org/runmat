use crate::indexing::read_linear as idx_read_linear;
use crate::indexing::selectors::index_scalar_from_value;
use crate::indexing::write_linear as idx_write_linear;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub async fn dispatch_indexing(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    pc: usize,
    mut clear_value_residency: impl FnMut(&Value),
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::Index(num_indices) => {
            let numeric = idx_read_linear::collect_linear_indices(stack, *num_indices).await?;
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
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
                let v: f64 = (&stack
                    .pop()
                    .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                indices.push(v as usize);
            }
            indices.reverse();
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            match base {
                Value::Object(obj) => {
                    let cell = runmat_runtime::call_builtin_async(
                        "__make_cell",
                        &indices.iter().map(|n| Value::Num(*n as f64)).collect::<Vec<_>>(),
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
                        &indices.iter().map(|n| Value::Num(*n as f64)).collect::<Vec<_>>(),
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
                    let v: f64 = (&stack
                        .pop()
                        .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    indices.push(v as usize);
                }
                indices.reverse();
            }
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
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
                        &indices.iter().map(|n| Value::Num(*n as f64)).collect::<Vec<_>>(),
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
                        &indices.iter().map(|n| Value::Num(*n as f64)).collect::<Vec<_>>(),
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
            let rhs = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let mut indices = Vec::new();
            for _ in 0..*num_indices {
                let v: f64 = (&stack
                    .pop()
                    .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                indices.push(v as usize);
            }
            indices.reverse();
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
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
                    let updated = crate::ops::cells::assign_cell_value(ca, &indices, rhs, |oldv, newv| {
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
                log::debug!("[vm] StoreIndex pre-snap pc={} stack_top_types={:?}", pc, snap);
            }
            let rhs = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
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
                            (Some(r), Some(c)) => i_val >= 1 && i_val <= r && j_val >= 1 && j_val <= c,
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
                        &indices.iter().map(|n| Value::Num(*n as f64)).collect::<Vec<_>>(),
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
                        &indices.iter().map(|n| Value::Num(*n as f64)).collect::<Vec<_>>(),
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
                Value::Tensor(t) => stack.push(idx_write_linear::assign_tensor_scalar(t, &indices, &rhs).await?),
                Value::ComplexTensor(t) => stack.push(idx_write_linear::assign_complex_scalar(t, &indices, &rhs).await?),
                Value::GpuTensor(h) => stack.push(idx_write_linear::assign_gpu_scalar(&h, &indices, &rhs).await?),
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
        _ => Ok(false),
    }
}
