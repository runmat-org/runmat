use crate::indexing::read_linear as idx_read_linear;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub async fn dispatch_indexing(instr: &crate::bytecode::Instr, stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
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
        _ => Ok(false),
    }
}
