use crate::functions::{Bytecode, ExecutionContext, UserFunction};
use crate::gc_roots::InterpretContext;
use crate::instr::Instr;
use runmat_builtins::Value;
use runmat_runtime::call_builtin;
use std::convert::TryInto;

pub fn interpret_with_vars(bytecode: &Bytecode, initial_vars: &mut [Value]) -> Result<Vec<Value>, String> {
    let mut stack: Vec<Value> = Vec::new();
    let mut vars = initial_vars.to_vec();
    if vars.len() < bytecode.var_count { vars.resize(bytecode.var_count, Value::Num(0.0)); }
    let mut pc: usize = 0;
    let mut context = ExecutionContext { call_stack: Vec::new(), locals: Vec::new(), instruction_pointer: 0, functions: bytecode.functions.clone() };
    let _gc_context = InterpretContext::new(&stack, &vars)?;
    // Stack of (catch_pc, catch_var_global_index)
    let mut try_stack: Vec<(usize, Option<usize>)> = Vec::new();
    macro_rules! vm_bail {
        ($err:expr) => {{
            let e: String = $err;
            if let Some((catch_pc, catch_var)) = try_stack.pop() {
                if let Some(var_idx) = catch_var {
                    if var_idx >= vars.len() { vars.resize(var_idx + 1, Value::Num(0.0)); }
                    vars[var_idx] = Value::MException(parse_exception(&e));
                }
                pc = catch_pc;
                continue;
            } else {
                return Err(e);
            }
        }};
    }
    while pc < bytecode.instructions.len() {
        match bytecode.instructions[pc].clone() {
            Instr::LoadConst(c) => stack.push(Value::Num(c)),
            Instr::LoadString(s) => stack.push(Value::String(s)),
            Instr::LoadVar(i) => stack.push(vars[i].clone()),
            Instr::StoreVar(i) => {
                let val = stack.pop().ok_or("stack underflow")?;
                if i >= vars.len() { vars.resize(i + 1, Value::Num(0.0)); }
                vars[i] = val;
            }
            Instr::LoadLocal(offset) => {
                if let Some(current_frame) = context.call_stack.last() {
                    let local_index = current_frame.locals_start + offset;
                    if local_index >= context.locals.len() { vm_bail!("Local variable index out of bounds".to_string()); }
                    stack.push(context.locals[local_index].clone());
                } else {
                    if offset < vars.len() { stack.push(vars[offset].clone()); } else { stack.push(Value::Num(0.0)); }
                }
            }
            Instr::StoreLocal(offset) => {
                let val = stack.pop().ok_or("stack underflow")?;
                if let Some(current_frame) = context.call_stack.last() {
                    let local_index = current_frame.locals_start + offset;
                    while context.locals.len() <= local_index { context.locals.push(Value::Num(0.0)); }
                    context.locals[local_index] = val;
                } else {
                    if offset >= vars.len() { vars.resize(offset + 1, Value::Num(0.0)); }
                    vars[offset] = val;
                }
            }
            Instr::EnterScope(local_count) => { for _ in 0..local_count { context.locals.push(Value::Num(0.0)); } }
            Instr::ExitScope(local_count) => { for _ in 0..local_count { context.locals.pop(); } }
            Instr::Add => element_binary(&mut stack, runmat_runtime::elementwise_add)?,
            Instr::Sub => element_binary(&mut stack, runmat_runtime::elementwise_sub)?,
            Instr::Mul => element_binary(&mut stack, runmat_runtime::elementwise_mul)?,
            Instr::Div => element_binary(&mut stack, runmat_runtime::elementwise_div)?,
            Instr::Pow => element_binary(&mut stack, runmat_runtime::power)?,
            Instr::Neg => { let value = stack.pop().ok_or("stack underflow")?; let result = runmat_runtime::elementwise_neg(&value)?; stack.push(result); }
            Instr::Transpose => { let value = stack.pop().ok_or("stack underflow")?; let result = runmat_runtime::transpose(value)?; stack.push(result); }
            Instr::ElemMul => element_binary(&mut stack, runmat_runtime::elementwise_mul)?,
            Instr::ElemDiv => element_binary(&mut stack, runmat_runtime::elementwise_div)?,
            Instr::ElemPow => element_binary(&mut stack, runmat_runtime::elementwise_pow)?,
            Instr::ElemLeftDiv => { element_binary(&mut stack, |a, b| runmat_runtime::elementwise_div(b, a))? }
            Instr::LessEqual => { let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; stack.push(Value::Num(if a <= b { 1.0 } else { 0.0 })); }
            Instr::Less => { let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; stack.push(Value::Num(if a < b { 1.0 } else { 0.0 })); }
            Instr::Greater => { let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; stack.push(Value::Num(if a > b { 1.0 } else { 0.0 })); }
            Instr::GreaterEqual => { let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; stack.push(Value::Num(if a >= b { 1.0 } else { 0.0 })); }
            Instr::Equal => { let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; stack.push(Value::Num(if a == b { 1.0 } else { 0.0 })); }
            Instr::NotEqual => { let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; stack.push(Value::Num(if a != b { 1.0 } else { 0.0 })); }
            Instr::JumpIfFalse(target) => { let cond: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; if cond == 0.0 { pc = target; continue; } }
            Instr::Jump(target) => { pc = target; continue; }
            Instr::CallBuiltin(name, arg_count) => {
                let mut args = Vec::new(); for _ in 0..arg_count { args.push(stack.pop().ok_or("stack underflow")?); } args.reverse();
                match call_builtin(&name, &args) {
                    Ok(result) => stack.push(result),
                    Err(e) => {
                        if let Some((catch_pc, catch_var)) = try_stack.pop() {
                            // Set catch var (global) if present
                            if let Some(var_idx) = catch_var {
                                if var_idx >= vars.len() { vars.resize(var_idx + 1, Value::Num(0.0)); }
                                vars[var_idx] = Value::MException(parse_exception(&e));
                            }
                            pc = catch_pc; // Jump to catch
                            continue;
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
            Instr::CallFunction(name, arg_count) => {
                let func: UserFunction = match bytecode.functions.get(&name) { Some(f) => f.clone(), None => vm_bail!(format!("undefined function: {name}")) };
                if arg_count != func.params.len() { vm_bail!(format!("Function '{}' expects {} arguments, got {} - Not enough input arguments", name, func.params.len(), arg_count)); }
                let mut args = Vec::new(); for _ in 0..arg_count { args.push(stack.pop().ok_or("stack underflow")?); } args.reverse();
                let var_map = runmat_hir::remapping::create_complete_function_var_map(&func.params, &func.outputs, &func.body);
                let local_var_count = var_map.len();
                let remapped_body = runmat_hir::remapping::remap_function_body(&func.body, &var_map);
                let func_vars_count = local_var_count.max(func.params.len());
                let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                for (i, _param_id) in func.params.iter().enumerate() { if i < args.len() && i < func_vars.len() { func_vars[i] = args[i].clone(); } }
                // Copy referenced globals into local frame
                for (original_var_id, local_var_id) in &var_map {
                    let local_index = local_var_id.0; let global_index = original_var_id.0;
                    if local_index < func_vars.len() && global_index < vars.len() {
                        let is_parameter = func.params.iter().any(|param_id| param_id == original_var_id);
                        if !is_parameter { func_vars[local_index] = vars[global_index].clone(); }
                    }
                }
                let func_program = runmat_hir::HirProgram { body: remapped_body };
                let func_bytecode = crate::compile_with_functions(&func_program, &bytecode.functions)?;
                let func_result_vars = match interpret_function(&func_bytecode, func_vars) {
                    Ok(v) => v,
                    Err(e) => {
                        if let Some((catch_pc, catch_var)) = try_stack.pop() {
                            if let Some(var_idx) = catch_var {
                                if var_idx >= vars.len() { vars.resize(var_idx + 1, Value::Num(0.0)); }
                                vars[var_idx] = Value::MException(parse_exception(&e));
                            }
                            pc = catch_pc; continue;
                        } else { vm_bail!(e); }
                    }
                };
                if let Some(output_var_id) = func.outputs.first() {
                    let local_output_index = var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);
                    if local_output_index < func_result_vars.len() { stack.push(func_result_vars[local_output_index].clone()); } else { stack.push(Value::Num(0.0)); }
                } else { stack.push(Value::Num(0.0)); }
            }
            Instr::CallFunctionMulti(name, arg_count, out_count) => {
                let func: UserFunction = match bytecode.functions.get(&name) { Some(f) => f.clone(), None => vm_bail!(format!("undefined function: {name}")) };
                if arg_count != func.params.len() { vm_bail!(format!("Function '{}' expects {} arguments, got {} - Not enough input arguments", name, func.params.len(), arg_count)); }
                let mut args = Vec::new(); for _ in 0..arg_count { args.push(stack.pop().ok_or("stack underflow")?); } args.reverse();
                let var_map = runmat_hir::remapping::create_complete_function_var_map(&func.params, &func.outputs, &func.body);
                let local_var_count = var_map.len();
                let remapped_body = runmat_hir::remapping::remap_function_body(&func.body, &var_map);
                let func_vars_count = local_var_count.max(func.params.len());
                let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                for (i, _param_id) in func.params.iter().enumerate() { if i < args.len() && i < func_vars.len() { func_vars[i] = args[i].clone(); } }
                for (original_var_id, local_var_id) in &var_map {
                    let local_index = local_var_id.0; let global_index = original_var_id.0;
                    if local_index < func_vars.len() && global_index < vars.len() {
                        let is_parameter = func.params.iter().any(|param_id| param_id == original_var_id);
                        if !is_parameter { func_vars[local_index] = vars[global_index].clone(); }
                    }
                }
                let func_program = runmat_hir::HirProgram { body: remapped_body };
                let func_bytecode = crate::compile_with_functions(&func_program, &bytecode.functions)?;
                let func_result_vars = match interpret_function(&func_bytecode, func_vars) {
                    Ok(v) => v,
                    Err(e) => {
                        if let Some((catch_pc, catch_var)) = try_stack.pop() {
                            if let Some(var_idx) = catch_var {
                                if var_idx >= vars.len() { vars.resize(var_idx + 1, Value::Num(0.0)); }
                                vars[var_idx] = Value::MException(parse_exception(&e));
                            }
                            pc = catch_pc; continue;
                        } else { vm_bail!(e); }
                    }
                };
                // Push out_count values, left-to-right; if missing, push 0.0
                for i in 0..out_count {
                    let v = func.outputs.get(i).and_then(|oid| var_map.get(oid)).map(|lid| lid.0)
                        .and_then(|idx| func_result_vars.get(idx)).cloned().unwrap_or(Value::Num(0.0));
                    stack.push(v);
                }
            }
            Instr::EnterTry(catch_pc, catch_var) => { try_stack.push((catch_pc, catch_var)); }
            Instr::PopTry => { try_stack.pop(); }
            Instr::CreateMatrix(rows, cols) => {
                let total_elements = rows * cols; let mut row_major = Vec::with_capacity(total_elements);
                for _ in 0..total_elements { let val: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; row_major.push(val); }
                row_major.reverse();
                // Reorder to column-major storage: cm[r + c*rows] = rm[r*cols + c]
                let mut data = vec![0.0; total_elements];
                for r in 0..rows { for c in 0..cols { data[r + c * rows] = row_major[r * cols + c]; } }
                let matrix = runmat_builtins::Tensor::new_2d(data, rows, cols).map_err(|e| format!("Matrix creation error: {e}"))?;
                stack.push(Value::Tensor(matrix));
            }
            Instr::CreateMatrixDynamic(num_rows) => {
                let mut row_lengths = Vec::new();
                for _ in 0..num_rows { let row_len: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; row_lengths.push(row_len as usize); }
                row_lengths.reverse();
                let mut rows_data = Vec::new();
                for &row_len in row_lengths.iter().rev() { let mut row_values = Vec::new(); for _ in 0..row_len { row_values.push(stack.pop().ok_or("stack underflow")?); } row_values.reverse(); rows_data.push(row_values); }
                rows_data.reverse();
                let result = runmat_runtime::create_matrix_from_values(&rows_data)?; stack.push(result);
            }
            Instr::CreateRange(has_step) => {
                if has_step {
                    let end: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let step: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let start: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let range_result = runmat_runtime::create_range(start, Some(step), end)?; stack.push(range_result);
                } else {
                    let end: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let start: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let range_result = runmat_runtime::create_range(start, None, end)?; stack.push(range_result);
                }
            }
            Instr::Index(num_indices) => {
                let mut indices = Vec::new(); let count = num_indices; for _ in 0..count { let index_val: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; indices.push(index_val); } indices.reverse();
                let base = stack.pop().ok_or("stack underflow")?;
                match base {
                    Value::Object(obj) => {
                        let cell = runmat_builtins::CellArray::new(indices.iter().map(|n| Value::Num(*n)).collect(), 1, indices.len()).map_err(|e| format!("subsref build error: {e}"))?;
                        match runmat_runtime::call_builtin("call_method", &[
                            Value::Object(obj),
                            Value::String("subsref".to_string()),
                            Value::String("()".to_string()),
                            Value::Cell(cell),
                        ]) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    other => {
                        let result = match runmat_runtime::perform_indexing(&other, &indices) { Ok(v) => v, Err(e) => vm_bail!(e) };
                        stack.push(result);
                    }
                }
            }
            Instr::IndexSlice(dims, numeric_count, colon_mask, end_mask) => {
                // Pop numeric indices in reverse order (they were pushed in order), then base
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count { numeric.push(stack.pop().ok_or("stack underflow")?); }
                numeric.reverse();
                let base = stack.pop().ok_or("stack underflow")?;
                match base {
                    Value::Object(obj) => {
                        let cell = runmat_builtins::CellArray::new(numeric.iter().cloned().collect(), 1, numeric.len()).map_err(|e| format!("subsref build error: {e}"))?;
                        match runmat_runtime::call_builtin("call_method", &[
                            Value::Object(obj),
                            Value::String("subsref".to_string()),
                            Value::String("()".to_string()),
                            Value::Cell(cell),
                        ]) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::Tensor(t) => {
                        let rank = t.shape.len();
                        // Build per-dimension selectors
                        #[derive(Clone)]
                        enum Sel { Colon, Scalar(usize), Indices(Vec<usize>) }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        if dims == 1 {
                            let total = t.data.len();
                            let mut idxs: Vec<usize> = Vec::new();
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon { idxs = (1..=total).collect(); }
                            else if is_end { idxs = vec![total]; }
                            else if let Some(v) = numeric.get(0) {
                                match v {
                                    Value::Num(n) => { let i = *n as isize; if i < 1 { vm_bail!("Index out of bounds".into()); } idxs = vec![i as usize]; }
                                    Value::Tensor(idx_t) => {
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == total { for (i, &val) in idx_t.data.iter().enumerate() { if val != 0.0 { idxs.push(i+1); } } }
                                        else { for &val in &idx_t.data { let i = val as isize; if i < 1 { vm_bail!("Index out of bounds".into()); } idxs.push(i as usize); } }
                                    }
                                    _ => vm_bail!("Unsupported index type".into()),
                                }
                            } else { vm_bail!("missing numeric index".into()); }
                            if idxs.iter().any(|&i| i == 0 || i > total) { vm_bail!("Index out of bounds".into()); }
                            if idxs.len() == 1 { stack.push(Value::Num(t.data[idxs[0] - 1])); }
                            else {
                                let mut out = Vec::with_capacity(idxs.len()); for &i in &idxs { out.push(t.data[i - 1]); }
                                let tens = runmat_builtins::Tensor::new(out, vec![idxs.len(), 1]).map_err(|e| format!("Slice error: {e}"))?;
                                stack.push(Value::Tensor(tens));
                            }
                        } else {
                            for d in 0..dims {
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon {
                                selectors.push(Sel::Colon);
                            } else if is_end {
                                // Plain 'end' -> scalar size of this dim
                                let dim_len = *t.shape.get(d).unwrap_or(&1);
                                selectors.push(Sel::Scalar(dim_len));
                            } else {
                                let v = numeric.get(num_iter).ok_or("missing numeric index")?;
                                num_iter += 1;
                                match v {
                                    Value::Num(n) => {
                                        let idx = *n as isize;
                                        if idx < 1 { return Err("Index out of bounds".to_string()); }
                                        selectors.push(Sel::Scalar(idx as usize));
                                    }
                                    Value::Tensor(idx_t) => {
                                        // Logical mask if length matches dimension
                                        let dim_len = *t.shape.get(d).unwrap_or(&1);
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == dim_len {
                                            let mut indices = Vec::new();
                                            for (i, &val) in idx_t.data.iter().enumerate() { if val != 0.0 { indices.push(i+1); } }
                                            selectors.push(Sel::Indices(indices));
                                        } else {
                                            // Treat as explicit indices (1-based)
                                            let mut indices = Vec::with_capacity(len);
                                            for &val in &idx_t.data { let idx = val as isize; if idx<1 { return Err("Index out of bounds".into()); } indices.push(idx as usize); }
                                            selectors.push(Sel::Indices(indices));
                                        }
                                    }
                                    _ => return Err("Unsupported index type".into()),
                                }
                            }
                        }
                        {
                            // Compute output shape and gather
                            let mut out_dims: Vec<usize> = Vec::new();
                            let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                            for d in 0..dims {
                                let dim_len = *t.shape.get(d).unwrap_or(&1);
                                let idxs = match &selectors[d] {
                                    Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                                    Sel::Scalar(i) => vec![*i],
                                    Sel::Indices(v) => v.clone(),
                                };
                                if idxs.iter().any(|&i| i == 0 || i > dim_len) { return Err("Index out of bounds".into()); }
                                if idxs.len() > 1 { out_dims.push(idxs.len()); } else { out_dims.push(1); }
                                per_dim_indices.push(idxs);
                            }
                            // 2D mixed selectors shape correction to match MATLAB:
                            // (I, scalar) => column vector [len(I), 1]; (scalar, J) => row vector [1, len(J)]
                            if dims == 2 {
                                match (&per_dim_indices[0].as_slice(), &per_dim_indices[1].as_slice()) {
                                    // I (len>1), scalar
                                    (i_list, j_list) if i_list.len() > 1 && j_list.len() == 1 => {
                                        out_dims = vec![i_list.len(), 1];
                                    }
                                    // scalar, J (len>1)
                                    (i_list, j_list) if i_list.len() == 1 && j_list.len() > 1 => {
                                        out_dims = vec![1, j_list.len()];
                                    }
                                    _ => {}
                                }
                            }
                            // Strides for column-major order (first dimension fastest)
                            let mut strides: Vec<usize> = vec![0; dims];
                            let full_shape: Vec<usize> = if rank < dims { let mut s = t.shape.clone(); s.resize(dims, 1); s } else { t.shape.clone() };
                            let mut acc = 1usize;
                            for d in 0..dims { strides[d] = acc; acc *= full_shape[d]; }
                            // Cartesian product gather
                            let total_out: usize = out_dims.iter().product();
                            let mut out_data: Vec<f64> = Vec::with_capacity(total_out);
                            if out_dims.iter().any(|&d| d == 0) {
                                // Empty selection on some dimension -> empty tensor
                                let out_tensor = runmat_builtins::Tensor::new(out_data, out_dims)
                                    .map_err(|e| format!("Slice error: {e}"))?;
                                stack.push(Value::Tensor(out_tensor));
                            } else {
                                fn cartesian<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
                                    let dims = lists.len();
                                    let mut idx = vec![0usize; dims];
                                    loop {
                                        let current: Vec<usize> = (0..dims).map(|d| lists[d][idx[d]]).collect();
                                        f(&current);
                                        // Increment first dimension fastest (column-major order)
                                        let mut d = 0usize;
                                        while d < dims {
                                            idx[d] += 1;
                                            if idx[d] < lists[d].len() { break; }
                                            idx[d] = 0;
                                            d += 1;
                                        }
                                        if d == dims { break; }
                                    }
                                }
                                cartesian(&per_dim_indices, |multi| {
                                    let mut lin = 0usize;
                                    for d in 0..dims { let i0 = multi[d] - 1; lin += i0 * strides[d]; }
                                    out_data.push(t.data[lin]);
                                });
                                if out_data.len() == 1 { stack.push(Value::Num(out_data[0])); }
                                else {
                                    let out_tensor = runmat_builtins::Tensor::new(out_data, out_dims).map_err(|e| format!("Slice error: {e}"))?;
                                    stack.push(Value::Tensor(out_tensor));
                                }
                            }
                        }
                        }
                    }
                    other => {
                        // Support 1-D linear indexing and scalar(1) on non-tensors
                        if dims == 1 {
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon { vm_bail!("Colon selection not supported on non-tensors".into()); }
                            let idx_val: f64 = if is_end {
                                1.0
                            } else {
                                match numeric.get(0) {
                                    Some(Value::Num(n)) => *n,
                                    Some(Value::Int(i)) => *i as f64,
                                    _ => 1.0,
                                }
                            };
                            let v = match runmat_runtime::perform_indexing(&other, &[idx_val]) { Ok(v) => v, Err(e) => vm_bail!(e) };
                            stack.push(v);
                        }
                        vm_bail!("Slicing only supported on tensors".to_string());
                    }
                }
            }
            Instr::StoreSlice(dims, numeric_count, colon_mask, end_mask) => {
                // RHS value to scatter, then numeric indices, then base
                let rhs = stack.pop().ok_or("stack underflow")?;
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count { numeric.push(stack.pop().ok_or("stack underflow")?); }
                numeric.reverse();
                let base = stack.pop().ok_or("stack underflow")?;
                match base {
                    Value::Object(obj) => {
                        let cell = runmat_builtins::CellArray::new(numeric.clone(), 1, numeric.len()).map_err(|e| format!("subsasgn build error: {e}"))?;
                        match runmat_runtime::call_builtin("call_method", &[
                            Value::Object(obj),
                            Value::String("subsasgn".to_string()),
                            Value::String("()".to_string()),
                            Value::Cell(cell),
                            rhs,
                        ]) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::Tensor(mut t) => {
                        let rank = t.shape.len();
                        #[derive(Clone)] enum Sel { Colon, Scalar(usize), Indices(Vec<usize>) }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        for d in 0..dims {
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon { selectors.push(Sel::Colon); }
                            else if is_end { selectors.push(Sel::Scalar(*t.shape.get(d).unwrap_or(&1))); }
                            else {
                                let v = numeric.get(num_iter).ok_or("missing numeric index")?; num_iter+=1;
                                match v {
                                    Value::Num(n) => { let idx = *n as isize; if idx<1 { vm_bail!("Index out of bounds".into()); } selectors.push(Sel::Scalar(idx as usize)); }
                                    Value::Tensor(idx_t) => {
                                        let dim_len = *t.shape.get(d).unwrap_or(&1);
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == dim_len { let mut v = Vec::new(); for (i, &val) in idx_t.data.iter().enumerate() { if val!=0.0 { v.push(i+1); } } selectors.push(Sel::Indices(v)); }
                                        else { let mut v = Vec::with_capacity(len); for &val in &idx_t.data { let idx = val as isize; if idx<1 { vm_bail!("Index out of bounds".into()); } v.push(idx as usize); } selectors.push(Sel::Indices(v)); }
                                    }
                                    _ => vm_bail!("Unsupported index type".into()),
                                }
                            }
                        }
                        // Build per-dim index lists and strides
                        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                        let full_shape: Vec<usize> = if rank < dims { let mut s = t.shape.clone(); s.resize(dims, 1); s } else { t.shape.clone() };
                        for d in 0..dims {
                            let dim_len = full_shape[d];
                            let idxs = match &selectors[d] { Sel::Colon => (1..=dim_len).collect(), Sel::Scalar(i) => vec![*i], Sel::Indices(v) => v.clone() };
                            if idxs.iter().any(|&i| i==0 || i>dim_len) { vm_bail!("Index out of bounds".into()); }
                            per_dim_indices.push(idxs);
                        }
                        // Column-major strides (first dimension fastest)
                        let mut strides: Vec<usize> = vec![0; dims];
                        let mut acc = 1usize; for d in 0..dims { strides[d] = acc; acc *= full_shape[d]; }
                        let total_out: usize = per_dim_indices.iter().map(|v| v.len()).product();
                        // Prepare RHS values
                        enum RhsView { Scalar(f64), Tensor{ data: Vec<f64>, shape: Vec<usize>, strides: Vec<usize> } }
                        let rhs_view = match rhs {
                            Value::Num(n) => RhsView::Scalar(n),
                            Value::Tensor(rt) => {
                                // Allow exact match or N-D broadcasting where rhs_dim is 1 or equals out_dim
                                let mut shape = rt.shape.clone();
                                if shape.len() < dims { shape.resize(dims, 1); }
                                if shape.len() > dims { if shape.iter().skip(dims).any(|&s| s != 1) { vm_bail!("shape mismatch for slice assign".into()); } shape.truncate(dims); }
                                let mut ok = true;
                                for d in 0..dims { let out_len = per_dim_indices[d].len(); let rhs_len = shape[d]; if !(rhs_len == 1 || rhs_len == out_len) { ok = false; break; } }
                                if !ok { vm_bail!("shape mismatch for slice assign".into()); }
                                let mut rstrides = vec![0usize; dims];
                                let mut racc = 1usize; for d in 0..dims { rstrides[d] = racc; racc *= shape[d]; }
                                RhsView::Tensor{ data: rt.data, shape, strides: rstrides }
                            }
                            _ => vm_bail!("rhs must be numeric or tensor".into()),
                        };
                        // Iterate and scatter
                        let mut _k = 0usize;
                        let mut idx = vec![0usize; dims];
                        if total_out == 0 { stack.push(Value::Tensor(t)); continue; }
                        loop {
                            let mut lin = 0usize; for d in 0..dims { let i0 = per_dim_indices[d][idx[d]] - 1; lin += i0 * strides[d]; }
                            match &rhs_view {
                                RhsView::Scalar(val) => t.data[lin] = *val,
                                RhsView::Tensor{ data, shape, strides } => {
                                    let mut rlin = 0usize;
                                    for d in 0..dims { let rhs_len = shape[d]; let pos = if rhs_len == 1 { 0 } else { idx[d] }; rlin += pos * strides[d]; }
                                    t.data[lin] = data[rlin];
                                }
                            }
                            _k += 1;
                            // Increment first dim fastest
                            let mut d = 0usize; while d < dims { idx[d]+=1; if idx[d] < per_dim_indices[d].len() { break; } idx[d]=0; d+=1; }
                            if d==dims { break; }
                        }
                        stack.push(Value::Tensor(t));
                    }
                    _ => vm_bail!("Slicing assignment only supported on tensors".into()),
                }
            }
            Instr::CreateCell2D(rows, cols) => {
                let mut elems = Vec::with_capacity(rows*cols);
                for _ in 0..rows*cols { elems.push(stack.pop().ok_or("stack underflow")?); }
                elems.reverse();
                let ca = runmat_builtins::CellArray::new(elems, rows, cols).map_err(|e| format!("Cell creation error: {e}"))?;
                stack.push(Value::Cell(ca));
            }
            Instr::IndexCell(num_indices) => {
                // Pop indices first (in reverse), then base
                let mut indices = Vec::with_capacity(num_indices);
                for _ in 0..num_indices { let v: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; indices.push(v as usize); }
                indices.reverse();
                let base = stack.pop().ok_or("stack underflow")?;
                match base {
                    Value::Object(obj) => {
                        // Route to subsref(obj, '{}', {indices})
                        let cell = runmat_builtins::CellArray::new(indices.iter().map(|n| Value::Num(*n as f64)).collect(), 1, indices.len())
                            .map_err(|e| format!("subsref build error: {e}"))?;
                        match runmat_runtime::call_builtin("call_method", &[
                            Value::Object(obj),
                            Value::String("subsref".to_string()),
                            Value::String("{}".to_string()),
                            Value::Cell(cell),
                        ]) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::Cell(ca) => {
                        match indices.len() {
                            1 => {
                                let i = indices[0];
                                if i == 0 || i > ca.data.len() { return Err("Cell index out of bounds".to_string()); }
                                stack.push(ca.data[i-1].clone());
                            }
                            2 => {
                                let r = indices[0]; let c = indices[1];
                                if r==0 || r>ca.rows || c==0 || c>ca.cols { return Err("Cell subscript out of bounds".to_string()); }
                                stack.push(ca.data[(r-1)*ca.cols + (c-1)].clone());
                            }
                            _ => return Err("Unsupported number of cell indices".to_string()),
                        }
                    }
                    _ => return Err("Cell indexing on non-cell".to_string()),
                }
            }
            Instr::Pop => { stack.pop(); }
            Instr::Return => { break; }
            Instr::ReturnValue => { let return_value = stack.pop().ok_or("stack underflow")?; stack.push(return_value); break; }
            Instr::StoreIndex(num_indices) => {
                // RHS to assign, then indices, then base
                let rhs = stack.pop().ok_or("stack underflow")?;
                let mut indices = Vec::new();
                for _ in 0..num_indices { let v: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; indices.push(v as usize); }
                indices.reverse();
                let base = stack.pop().ok_or("stack underflow")?;
                match base {
                    Value::Object(obj) => {
                        // subsasgn(obj, '()', {indices...}, rhs)
                        let cell = runmat_builtins::CellArray::new(indices.iter().map(|n| Value::Num(*n as f64)).collect(), 1, indices.len())
                            .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match runmat_runtime::call_builtin("call_method", &[
                            Value::Object(obj),
                            Value::String("subsasgn".to_string()),
                            Value::String("()".to_string()),
                            Value::Cell(cell),
                            rhs,
                        ]) { Ok(v) => stack.push(v), Err(e) => vm_bail!(e) }
                    }
                    Value::Tensor(mut t) => {
                        // 1D linear or 2D scalar assignment only for now
                        if indices.len() == 1 {
                            let total = t.rows() * t.cols();
                            let idx = indices[0];
                            if idx == 0 || idx > total { return Err("Index out of bounds".to_string()); }
                            let val: f64 = (&rhs).try_into()?;
                            t.data[idx - 1] = val;
                            stack.push(Value::Tensor(t));
                        } else if indices.len() == 2 {
                            let i = indices[0]; let j = indices[1];
                            let rows = t.rows(); let cols = t.cols();
                            if i == 0 || i > rows || j == 0 || j > cols { return Err("Subscript out of bounds".to_string()); }
                            let val: f64 = (&rhs).try_into()?;
                            let idx = (i - 1) + (j - 1) * rows;
                            t.data[idx] = val;
                            stack.push(Value::Tensor(t));
                        } else {
                            return Err("Only 1D/2D scalar assignment supported".to_string());
                        }
                    }
                    _ => return Err("Index assignment only for tensors".to_string()),
                }
            }
            Instr::StoreIndexCell(num_indices) => {
                // RHS, then indices, then base cell
                let rhs = stack.pop().ok_or("stack underflow")?;
                let mut indices = Vec::new();
                for _ in 0..num_indices { let v: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?; indices.push(v as usize); }
                indices.reverse();
                let base = stack.pop().ok_or("stack underflow")?;
                match base {
                    Value::Object(obj) => {
                        // subsasgn(obj, '{}', {indices}, rhs)
                        let cell = runmat_builtins::CellArray::new(indices.iter().map(|n| Value::Num(*n as f64)).collect(), 1, indices.len())
                            .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match runmat_runtime::call_builtin("call_method", &[
                            Value::Object(obj),
                            Value::String("subsasgn".to_string()),
                            Value::String("{}".to_string()),
                            Value::Cell(cell),
                            rhs,
                        ]) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::Cell(mut ca) => {
                        match indices.len() {
                            1 => {
                                let i = indices[0];
                                if i == 0 || i > ca.data.len() { return Err("Cell index out of bounds".to_string()); }
                                ca.data[i - 1] = rhs;
                                stack.push(Value::Cell(ca));
                            }
                            2 => {
                                let i = indices[0]; let j = indices[1];
                                if i == 0 || i > ca.rows || j == 0 || j > ca.cols { return Err("Cell subscript out of bounds".to_string()); }
                                ca.data[(i - 1) * ca.cols + (j - 1)] = rhs;
                                stack.push(Value::Cell(ca));
                            }
                            _ => return Err("Unsupported number of cell indices".to_string()),
                        }
                    }
                    _ => return Err("Cell assignment on non-cell".to_string()),
                }
            }
            Instr::LoadMember(field) => {
                let base = stack.pop().ok_or("stack underflow")?;
                match base {
                    Value::Object(obj) => {
                        if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                            if let Some(p) = cls.properties.get(&field) {
                                if p.is_static { vm_bail!(format!("Property '{}' is static; use classref('{}').{}", field, obj.class_name, field)); }
                                match p.access { runmat_builtins::Access::Private => vm_bail!(format!("Property '{}' is private", field)), _ => {} }
                            }
                            if let Some(v) = obj.properties.get(&field) { stack.push(v.clone()); }
                            else if cls.methods.contains_key("subsref") {
                                match runmat_runtime::call_builtin("call_method", &[
                                    Value::Object(obj),
                                    Value::String("subsref".to_string()),
                                    Value::String(".".to_string()),
                                    Value::String(field),
                                ]) { Ok(v) => stack.push(v), Err(e) => vm_bail!(e) }
                            } else { vm_bail!(format!("Undefined property '{}' for class {}", field, obj.class_name)); }
                        } else { vm_bail!(format!("Unknown class {}", obj.class_name)); }
                    }
                    _ => vm_bail!("LoadMember on non-object".into()),
                }
            }
            Instr::StoreMember(field) => {
                let rhs = stack.pop().ok_or("stack underflow")?;
                let base = stack.pop().ok_or("stack underflow")?;
                match base {
                    Value::Object(mut obj) => {
                        if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                            if let Some(p) = cls.properties.get(&field) {
                                if p.is_static { vm_bail!(format!("Property '{}' is static; use classref('{}').{}", field, obj.class_name, field)); }
                                match p.access { runmat_builtins::Access::Private => vm_bail!(format!("Property '{}' is private", field)), _ => {} }
                                obj.properties.insert(field, rhs); stack.push(Value::Object(obj));
                            } else if cls.methods.contains_key("subsasgn") {
                                match runmat_runtime::call_builtin("call_method", &[
                                    Value::Object(obj),
                                    Value::String("subsasgn".to_string()),
                                    Value::String(".".to_string()),
                                    Value::String(field),
                                    rhs,
                                ]) { Ok(v) => stack.push(v), Err(e) => vm_bail!(e) }
                            } else { vm_bail!(format!("Undefined property '{}' for class {}", field, obj.class_name)); }
                        } else { vm_bail!(format!("Unknown class {}", obj.class_name)); }
                    }
                    _ => vm_bail!("StoreMember on non-object".into()),
                }
            }
            Instr::CallMethod(name, arg_count) => {
                // base, then args are on stack in order: [..., base, a1, a2, ...]
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count { args.push(stack.pop().ok_or("stack underflow")?); }
                args.reverse();
                let base = stack.pop().ok_or("stack underflow")?;
                match base {
                    Value::Object(obj) => {
                        // Compose qualified and try runtime builtin dispatch, passing receiver first
                        if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                            if let Some(m) = cls.methods.get(&name) {
                                if m.is_static { vm_bail!(format!("Method '{}' is static; use classref({}).{}", name, obj.class_name, name)); }
                                match m.access { runmat_builtins::Access::Private => vm_bail!(format!("Method '{}' is private", name)), _ => {} }
                                let mut full_args = Vec::with_capacity(1 + args.len());
                                full_args.push(Value::Object(obj));
                                full_args.extend(args.into_iter());
                                let v = runmat_runtime::call_builtin(&m.function_name, &full_args)?; stack.push(v); continue;
                            }
                        }
                        let qualified = format!("{}.{}", obj.class_name, name);
                        let mut full_args = Vec::with_capacity(1 + args.len());
                        full_args.push(Value::Object(obj));
                        full_args.extend(args.into_iter());
                        if let Ok(v) = runmat_runtime::call_builtin(&qualified, &full_args) { stack.push(v); }
                        else {
                            match runmat_runtime::call_builtin(&name, &full_args) {
                                Ok(v) => { stack.push(v); }
                                Err(e) => { vm_bail!(e); }
                            }
                        }
                    }
                    _ => vm_bail!("CallMethod on non-object".into()),
                }
            }
            Instr::LoadMethod(name) => {
                // Base object on stack; return a closure that calls the method with receiver as first captured arg
                let base = stack.pop().ok_or("stack underflow")?;
                match base {
                    Value::Object(obj) => {
                        let func_qual = format!("{}.{}", obj.class_name, name);
                        stack.push(Value::Closure(runmat_builtins::Closure { function_name: func_qual, captures: vec![Value::Object(obj)] }));
                    }
                    Value::ClassRef(cls) => {
                        // Bound static method handle (no receiver capture)
                        let func_qual = format!("{}.{}", cls, name);
                        stack.push(Value::Closure(runmat_builtins::Closure { function_name: func_qual, captures: vec![] }));
                    }
                    _ => vm_bail!("LoadMethod requires object or classref".into()),
                }
            }
            Instr::CreateClosure(func_name, capture_count) => {
                let mut captures = Vec::with_capacity(capture_count);
                for _ in 0..capture_count { captures.push(stack.pop().ok_or("stack underflow")?); }
                captures.reverse();
                stack.push(Value::Closure(runmat_builtins::Closure { function_name: func_name, captures }));
            }
            Instr::LoadStaticProperty(class_name, prop) => {
                // Enforce access and static-ness via registry
                if let Some(cls) = runmat_builtins::get_class(&class_name) {
                    if let Some(p) = cls.properties.get(&prop) {
                        if !p.is_static { vm_bail!(format!("Property '{}' is not static", prop)); }
                        match p.access { runmat_builtins::Access::Private => vm_bail!(format!("Property '{}' is private", prop)), _ => {} }
                        if let Some(v) = runmat_builtins::get_static_property_value(&class_name, &prop) { stack.push(v); } else if let Some(v) = &p.default_value { stack.push(v.clone()); } else { stack.push(Value::Num(0.0)); }
                    } else { vm_bail!(format!("Unknown property '{}' on class {}", prop, class_name)); }
                } else { vm_bail!(format!("Unknown class {}", class_name)); }
            }
            Instr::CallStaticMethod(class_name, method, arg_count) => {
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count { args.push(stack.pop().ok_or("stack underflow")?); }
                args.reverse();
                if let Some(cls) = runmat_builtins::get_class(&class_name) {
                    if let Some(m) = cls.methods.get(&method) {
                        if !m.is_static { vm_bail!(format!("Method '{}' is not static", method)); }
                        match m.access { runmat_builtins::Access::Private => vm_bail!(format!("Method '{}' is private", method)), _ => {} }
                        let v = match runmat_runtime::call_builtin(&m.function_name, &args) { Ok(v) => v, Err(e) => vm_bail!(e) };
                        stack.push(v);
                    } else { vm_bail!(format!("Unknown static method '{}' on class {}", method, class_name)); }
                } else { vm_bail!(format!("Unknown class {}", class_name)); }
            }
            Instr::CallFeval(arg_count) => {
                // Stack layout: [..., f, a1, a2, ...]
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count { args.push(stack.pop().ok_or("stack underflow")?); }
                args.reverse();
                let func_val = stack.pop().ok_or("stack underflow")?;
                match func_val {
                    Value::Closure(c) => {
                        // First try runtime builtin dispatch with captures prepended
                        let mut full_args = c.captures.clone();
                        full_args.extend(args.into_iter());
                        match runmat_runtime::call_builtin(&c.function_name, &full_args) {
                            Ok(v) => stack.push(v),
                            Err(e) => {
                                // If not a builtin, try user-defined function
                                if let Some(func) = context.functions.get(&c.function_name).cloned() {
                                    let argc = full_args.len();
                                    if argc != func.params.len() { vm_bail!(format!("Function '{}' expects {} arguments, got {} - Not enough input arguments", c.function_name, func.params.len(), argc)); }
                                    let var_map = runmat_hir::remapping::create_complete_function_var_map(&func.params, &func.outputs, &func.body);
                                    let local_var_count = var_map.len();
                                    let remapped_body = runmat_hir::remapping::remap_function_body(&func.body, &var_map);
                                    let func_vars_count = local_var_count.max(func.params.len());
                                    let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                                    for (i, _param_id) in func.params.iter().enumerate() { if i < full_args.len() && i < func_vars.len() { func_vars[i] = full_args[i].clone(); } }
                                    // Copy referenced globals into local frame
                                    for (original_var_id, local_var_id) in &var_map {
                                        let local_index = local_var_id.0; let global_index = original_var_id.0;
                                        if local_index < func_vars.len() && global_index < vars.len() {
                                            let is_parameter = func.params.iter().any(|param_id| param_id == original_var_id);
                                            if !is_parameter { func_vars[local_index] = vars[global_index].clone(); }
                                        }
                                    }
                                    let func_program = runmat_hir::HirProgram { body: remapped_body };
                                    let func_bytecode = crate::compile_with_functions(&func_program, &context.functions)?;
                                    // Merge any newly synthesized nested functions into current context
                                    for (k, v) in func_bytecode.functions.iter() { context.functions.insert(k.clone(), v.clone()); }
                                    let func_result_vars = match interpret_function(&func_bytecode, func_vars) { Ok(v) => v, Err(e) => vm_bail!(e) };
                                    if let Some(output_var_id) = func.outputs.first() {
                                        let local_output_index = var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);
                                        if local_output_index < func_result_vars.len() { stack.push(func_result_vars[local_output_index].clone()); } else { stack.push(Value::Num(0.0)); }
                                    } else { stack.push(Value::Num(0.0)); }
                                } else {
                                    // Not found -> raise error (will be caught by try/catch if present)
                                    vm_bail!(e);
                                }
                            }
                        }
                    }
                    Value::String(s) => {
                        if let Some(name) = s.strip_prefix('@') {
                            match runmat_runtime::call_builtin(name, &args) {
                                Ok(v) => { stack.push(v); }
                                Err(e) => { vm_bail!(e); }
                            }
                        } else { vm_bail!(format!("feval: expected function handle string starting with '@', got {s}")); }
                    }
                    Value::FunctionHandle(name) => {
                        match runmat_runtime::call_builtin(&name, &args) {
                            Ok(v) => { stack.push(v); }
                            Err(e) => { vm_bail!(e); }
                        }
                    }
                    other => vm_bail!(format!("feval: unsupported function value {other:?}")),
                }
            }
            Instr::Swap => {
                let a = stack.pop().ok_or("stack underflow")?;
                let b = stack.pop().ok_or("stack underflow")?;
                stack.push(a);
                stack.push(b);
            }
        }
        pc += 1;
    }
    for (i, var) in vars.iter().enumerate() { if i < initial_vars.len() { initial_vars[i] = var.clone(); } }
    Ok(vars)
}

fn parse_exception(err: &str) -> runmat_builtins::MException {
    // Expect "Identifier: message"; if multiple ':', use last as separator
    if let Some(idx) = err.rfind(':') {
        let (id, msg) = err.split_at(idx);
        let message = msg.trim_start_matches(':').trim().to_string();
        runmat_builtins::MException::new(id.trim().to_string(), message)
    } else {
        runmat_builtins::MException::new("MATLAB:Error".to_string(), err.to_string())
    }
}

fn element_binary<F>(stack: &mut Vec<Value>, f: F) -> Result<(), String>
where
    F: Fn(&Value, &Value) -> Result<Value, String>,
{
    let b = stack.pop().ok_or("stack underflow")?;
    let a = stack.pop().ok_or("stack underflow")?;
    let result = f(&a, &b)?;
    stack.push(result);
    Ok(())
}

/// Interpret bytecode with default variable initialization
pub fn interpret(bytecode: &Bytecode) -> Result<Vec<Value>, String> {
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    interpret_with_vars(bytecode, &mut vars)
}

pub fn interpret_function(bytecode: &Bytecode, mut vars: Vec<Value>) -> Result<Vec<Value>, String> {
    interpret_with_vars(bytecode, &mut vars)
}


