use crate::functions::{Bytecode, ExecutionContext, UserFunction};
use crate::gc_roots::InterpretContext;
use crate::instr::Instr;
#[cfg(feature = "native-accel")]
use runmat_accelerate::fusion_exec::{execute_elementwise, execute_reduction, FusionExecutionRequest};
#[cfg(feature = "native-accel")]
use runmat_accelerate::{
    activate_fusion_plan, deactivate_fusion_plan, fusion_residency, prepare_fusion_plan,
    set_current_pc,
};
#[cfg(feature = "native-accel")]
use runmat_accelerate::{active_group_plan_clone, ValueOrigin, VarKind};
use runmat_builtins::{Type, Value};
use runmat_runtime::call_builtin;
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryInto;

#[cfg(feature = "native-accel")]
struct FusionPlanGuard;

#[cfg(feature = "native-accel")]
impl Drop for FusionPlanGuard {
    fn drop(&mut self) {
        deactivate_fusion_plan();
    }
}

#[derive(Clone, Copy)]
enum AutoBinaryOp {
    Elementwise,
    MatMul,
}

#[derive(Clone, Copy)]
enum AutoUnaryOp {
    Transpose,
}

#[cfg(feature = "native-accel")]
fn accel_promote_binary(op: AutoBinaryOp, a: &Value, b: &Value) -> Result<(Value, Value), String> {
    use runmat_accelerate::{promote_binary, BinaryOp};
    let mapped = match op {
        AutoBinaryOp::Elementwise => BinaryOp::Elementwise,
        AutoBinaryOp::MatMul => BinaryOp::MatMul,
    };
    promote_binary(mapped, a, b).map_err(|e| e.to_string())
}

#[cfg(not(feature = "native-accel"))]
fn accel_promote_binary(_op: AutoBinaryOp, a: &Value, b: &Value) -> Result<(Value, Value), String> {
    Ok((a.clone(), b.clone()))
}

#[cfg(feature = "native-accel")]
fn accel_promote_unary(op: AutoUnaryOp, value: &Value) -> Result<Value, String> {
    use runmat_accelerate::{promote_unary, UnaryOp};
    let mapped = match op {
        AutoUnaryOp::Transpose => UnaryOp::Transpose,
    };
    promote_unary(mapped, value).map_err(|e| e.to_string())
}

#[cfg(not(feature = "native-accel"))]
fn accel_promote_unary(_op: AutoUnaryOp, value: &Value) -> Result<Value, String> {
    Ok(value.clone())
}

#[cfg(feature = "native-accel")]
fn accel_prepare_args(name: &str, args: &[Value]) -> Result<Vec<Value>, String> {
    runmat_accelerate::prepare_builtin_args(name, args).map_err(|e| e.to_string())
}

#[cfg(not(feature = "native-accel"))]
fn accel_prepare_args(_name: &str, args: &[Value]) -> Result<Vec<Value>, String> {
    Ok(args.to_vec())
}

fn call_builtin_auto(name: &str, args: &[Value]) -> Result<Value, String> {
    let prepared = accel_prepare_args(name, args)?;
    runmat_runtime::call_builtin(name, &prepared)
}

// Namespace used for error identifiers (e.g., "MATLAB:..." or "RunMat:...")
const ERROR_NAMESPACE: &str = "MATLAB";

#[inline]
fn mex(id: &str, msg: &str) -> String {
    // Normalize identifier to always use the configured namespace prefix.
    // If caller passes "Namespace:suffix", strip the namespace and re-prefix with ERROR_NAMESPACE.
    let suffix = match id.find(':') {
        Some(pos) => &id[pos + 1..],
        None => id,
    };
    let ident = format!("{ERROR_NAMESPACE}:{suffix}");
    format!("{ident}: {msg}")
}

thread_local! {
    static GLOBALS: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
}

thread_local! {
    static PERSISTENTS: RefCell<HashMap<(String, usize), Value>> = RefCell::new(HashMap::new());
}

thread_local! {
    static PERSISTENTS_BY_NAME: RefCell<HashMap<(String, String), Value>> = RefCell::new(HashMap::new());
}

thread_local! {
    // (nargin, nargout) for current call
    static CALL_COUNTS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

macro_rules! handle_rel_binary { ($op:tt, $name:literal, $stack:ident) => {{
    let b = $stack.pop().ok_or(mex("StackUnderflow","stack underflow"))?; let a = $stack.pop().ok_or(mex("StackUnderflow","stack underflow"))?;
    match (&a, &b) {
        (Value::Object(obj), _) => { let args = vec![Value::Object(obj.clone()), Value::String($name.to_string()), b.clone()]; match call_builtin("call_method", &args) { Ok(v) => $stack.push(v), Err(_) => { let aa: f64 = (&a).try_into()?; let bb: f64 = (&b).try_into()?; $stack.push(Value::Num(if aa $op bb {1.0}else{0.0})) } } }
        (_, Value::Object(obj)) => { let rev = match $name { "lt" => "gt", "le" => "ge", "gt" => "lt", "ge" => "le", other => other };
            let args = vec![Value::Object(obj.clone()), Value::String(rev.to_string()), a.clone()]; match call_builtin("call_method", &args) { Ok(v) => $stack.push(v), Err(_) => { let aa: f64 = (&a).try_into()?; let bb: f64 = (&b).try_into()?; $stack.push(Value::Num(if aa $op bb {1.0}else{0.0})) } } }
        _ => { let bb: f64 = (&b).try_into()?; let aa: f64 = (&a).try_into()?; $stack.push(Value::Num(if aa $op bb {1.0}else{0.0})) }
    }
}}; }

pub fn interpret_with_vars(
    bytecode: &Bytecode,
    initial_vars: &mut [Value],
    current_function_name: Option<&str>,
) -> Result<Vec<Value>, String> {
    #[cfg(feature = "native-accel")]
    let fusion_plan = prepare_fusion_plan(bytecode.accel_graph.as_ref(), &bytecode.fusion_groups);
    #[cfg(feature = "native-accel")]
    activate_fusion_plan(fusion_plan.clone());
    #[cfg(feature = "native-accel")]
    let _fusion_guard = FusionPlanGuard;
    let mut stack: Vec<Value> = Vec::new();
    let mut vars = initial_vars.to_vec();
    if vars.len() < bytecode.var_count {
        vars.resize(bytecode.var_count, Value::Num(0.0));
    }
    let mut pc: usize = 0;
    let mut context = ExecutionContext {
        call_stack: Vec::new(),
        locals: Vec::new(),
        instruction_pointer: 0,
        functions: bytecode.functions.clone(),
    };
    let mut _gc_context = InterpretContext::new(&stack, &vars)?;
    // Register thread-local globals/persistents as GC roots for the duration of this execution
    let mut thread_roots: Vec<Value> = Vec::new();
    GLOBALS.with(|g| {
        for v in g.borrow().values() {
            thread_roots.push(v.clone());
        }
    });
    PERSISTENTS.with(|p| {
        for v in p.borrow().values() {
            thread_roots.push(v.clone());
        }
    });
    // Name-based table may duplicate persistents; harmless if included
    PERSISTENTS_BY_NAME.with(|p| {
        for v in p.borrow().values() {
            thread_roots.push(v.clone());
        }
    });
    let _ = _gc_context.register_global_values(thread_roots, "thread_globals_persistents");
    let current_func_name_str: String = current_function_name
        .map(|s| s.to_string())
        .unwrap_or_else(|| "<main>".to_string());
    // Track per-execution alias maps for globals/persistents
    let mut global_aliases: HashMap<usize, String> = HashMap::new();
    let mut persistent_aliases: HashMap<usize, String> = HashMap::new();
    // Stack of (catch_pc, catch_var_global_index)
    let mut try_stack: Vec<(usize, Option<usize>)> = Vec::new();
    // Track last caught exception for possible rethrow handling
    let mut last_exception: Option<runmat_builtins::MException> = None;
    // Runtime import registry for this execution
    let mut imports: Vec<(Vec<String>, bool)> = Vec::new();
    // Helper to resolve unqualified static accesses if Class.* is imported
    let _resolve_static =
        |imports: &Vec<(Vec<String>, bool)>, name: &str| -> Option<(String, String)> {
            // Return (class_name, member) for unqualified 'member' where Class.* imported
            for (path, wildcard) in imports {
                if !*wildcard {
                    continue;
                }
                if path.len() == 1 {
                    // Class.* style
                    let class_name = path[0].clone();
                    // We cannot know member names here; VM paths for LoadMember/CallMethod will enforce static
                    return Some((class_name, name.to_string()));
                }
            }
            None
        };
    #[inline]
    fn bench_start() -> Option<std::time::Instant> {
        None
    }
    #[inline]
    fn bench_end(_label: &str, _start: Option<std::time::Instant>) {}
    macro_rules! vm_bail {
        ($err:expr) => {{
            let e: String = $err;
            if let Some((catch_pc, catch_var)) = try_stack.pop() {
                if let Some(var_idx) = catch_var {
                    if var_idx >= vars.len() {
                        vars.resize(var_idx + 1, Value::Num(0.0));
                    }
                    let mex = parse_exception(&e);
                    last_exception = Some(mex.clone());
                    vars[var_idx] = Value::MException(mex);
                }
                pc = catch_pc;
                continue;
            } else {
                return Err(e);
            }
        }};
    }
    while pc < bytecode.instructions.len() {
        #[cfg(feature = "native-accel")]
        set_current_pc(pc);
        #[cfg(feature = "native-accel")]
        if let (Some(plan), Some(graph)) =
            (active_group_plan_clone(), bytecode.accel_graph.as_ref())
        {
            if plan.group.span.start == pc {
                match try_execute_fusion_group(&plan, graph, &mut stack, &mut vars, &context) {
                    Ok(result) => {
                        stack.push(result);
                        pc = plan.group.span.end + 1;
                        continue;
                    }
                    Err(err) => {
                        log::debug!("fusion fallback at pc {}: {}", pc, err);
                    }
                }
            }
        }
        match bytecode.instructions[pc].clone() {
            Instr::LoadConst(c) => stack.push(Value::Num(c)),
            Instr::LoadBool(b) => stack.push(Value::Bool(b)),
            Instr::LoadString(s) => stack.push(Value::String(s)),
            Instr::LoadCharRow(s) => {
                let ca = runmat_builtins::CharArray::new(s.chars().collect(), 1, s.chars().count())
                    .map_err(|e| mex("CharError", &e))?;
                stack.push(Value::CharArray(ca));
            }
            Instr::LoadVar(i) => stack.push(vars[i].clone()),
            Instr::StoreVar(i) => {
                let val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                if i < vars.len() {
                    #[cfg(feature = "native-accel")]
                    clear_residency(&vars[i]);
                }
                if i >= vars.len() {
                    vars.resize(i + 1, Value::Num(0.0));
                }
                vars[i] = val;
                // If this var is declared global, update the global table entry
                // We optimistically write-through whenever StoreVar happens and a global exists for this name
                let key = format!("var_{i}");
                GLOBALS.with(|g| {
                    let mut m = g.borrow_mut();
                    if m.contains_key(&key) {
                        m.insert(key, vars[i].clone());
                    }
                });
                if let Some(name) = global_aliases.get(&i) {
                    GLOBALS.with(|g| {
                        g.borrow_mut().insert(name.clone(), vars[i].clone());
                    });
                }
            }
            Instr::LoadLocal(offset) => {
                if let Some(current_frame) = context.call_stack.last() {
                    let local_index = current_frame.locals_start + offset;
                    if local_index >= context.locals.len() {
                        vm_bail!("Local variable index out of bounds".to_string());
                    }
                    stack.push(context.locals[local_index].clone());
                } else if offset < vars.len() {
                    stack.push(vars[offset].clone());
                } else {
                    stack.push(Value::Num(0.0));
                }
            }
            Instr::StoreLocal(offset) => {
                let val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                if let Some(current_frame) = context.call_stack.last() {
                    let local_index = current_frame.locals_start + offset;
                    while context.locals.len() <= local_index {
                        context.locals.push(Value::Num(0.0));
                    }
                    #[cfg(feature = "native-accel")]
                    if local_index < context.locals.len() {
                        clear_residency(&context.locals[local_index]);
                    }
                    context.locals[local_index] = val;
                } else {
                    if offset >= vars.len() {
                        vars.resize(offset + 1, Value::Num(0.0));
                    }
                    #[cfg(feature = "native-accel")]
                    if offset < vars.len() {
                        clear_residency(&vars[offset]);
                    }
                    vars[offset] = val;
                    // write-through to persistents if this local is a declared persistent for current function
                    let func_name = context
                        .call_stack
                        .last()
                        .map(|f| f.function_name.clone())
                        .unwrap_or_else(|| "<main>".to_string());
                    let key = (func_name, offset);
                    PERSISTENTS.with(|p| {
                        let mut m = p.borrow_mut();
                        if m.contains_key(&key) {
                            m.insert(key, vars[offset].clone());
                        }
                    });
                }
            }
            Instr::EnterScope(local_count) => {
                for _ in 0..local_count {
                    context.locals.push(Value::Num(0.0));
                }
            }
            Instr::ExitScope(local_count) => {
                for _ in 0..local_count {
                    if let Some(val) = context.locals.pop() {
                        #[cfg(feature = "native-accel")]
                        clear_residency(&val);
                    }
                }
            }
            Instr::RegisterImport { path, wildcard } => {
                imports.push((path, wildcard));
            }
            Instr::DeclareGlobal(indices) => {
                // Bind local var slots to global table entries by name (var_N)
                for i in indices.into_iter() {
                    let key = format!("var_{i}");
                    let val_opt = GLOBALS.with(|g| g.borrow().get(&key).cloned());
                    if let Some(v) = val_opt {
                        if i >= vars.len() {
                            vars.resize(i + 1, Value::Num(0.0));
                        }
                        vars[i] = v;
                    }
                }
            }
            Instr::DeclareGlobalNamed(indices, names) => {
                for (pos, i) in indices.into_iter().enumerate() {
                    let name = names
                        .get(pos)
                        .cloned()
                        .unwrap_or_else(|| format!("var_{i}"));
                    let val_opt = GLOBALS.with(|g| g.borrow().get(&name).cloned());
                    if let Some(v) = val_opt {
                        if i >= vars.len() {
                            vars.resize(i + 1, Value::Num(0.0));
                        }
                        vars[i] = v;
                    }
                    GLOBALS.with(|g| {
                        let mut m = g.borrow_mut();
                        if let Some(v) = m.get(&name).cloned() {
                            m.insert(format!("var_{i}"), v);
                        }
                    });
                    global_aliases.insert(i, name);
                }
            }
            Instr::DeclarePersistent(indices) => {
                // Initialize locals from persistent table if present
                let func_name = current_func_name_str.clone();
                for i in indices.into_iter() {
                    let key = (func_name.clone(), i);
                    let val_opt = PERSISTENTS.with(|p| p.borrow().get(&key).cloned());
                    if let Some(v) = val_opt {
                        if i >= vars.len() {
                            vars.resize(i + 1, Value::Num(0.0));
                        }
                        vars[i] = v;
                    }
                }
            }
            Instr::DeclarePersistentNamed(indices, names) => {
                let func_name = current_func_name_str.clone();
                for (pos, i) in indices.into_iter().enumerate() {
                    let name = names
                        .get(pos)
                        .cloned()
                        .unwrap_or_else(|| format!("var_{i}"));
                    let key = (func_name.clone(), i);
                    let val_opt = PERSISTENTS_BY_NAME
                        .with(|p| p.borrow().get(&(func_name.clone(), name.clone())).cloned())
                        .or_else(|| PERSISTENTS.with(|p| p.borrow().get(&key).cloned()));
                    if let Some(v) = val_opt {
                        if i >= vars.len() {
                            vars.resize(i + 1, Value::Num(0.0));
                        }
                        vars[i] = v;
                    }
                    persistent_aliases.insert(i, name);
                }
            }
            Instr::Add => {
                // If either operand is an object, try operator overloading
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("plus".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::elementwise_add(&a, &b)?;
                                stack.push(v)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("plus".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::elementwise_add(&a, &b)?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        let v = runmat_runtime::elementwise_add(&a_acc, &b_acc)?;
                        stack.push(v)
                    }
                }
            }
            Instr::Sub => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![Value::Object(obj.clone()), b.clone()];
                        match call_builtin("minus", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::elementwise_sub(&a, &b)?;
                                stack.push(v)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![Value::Object(obj.clone()), a.clone()];
                        match call_builtin("uminus", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::elementwise_sub(&a, &b)?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        let v = runmat_runtime::elementwise_sub(&a_acc, &b_acc)?;
                        stack.push(v)
                    }
                }
            }
            Instr::Mul => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("mtimes".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::matrix::value_matmul(&a, &b)?;
                                stack.push(v)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("mtimes".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::matrix::value_matmul(&a, &b)?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::MatMul, &a, &b)?;
                        let v = runmat_runtime::matrix::value_matmul(&a_acc, &b_acc)?;
                        stack.push(v)
                    }
                }
            }
            Instr::Div => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("mrdivide".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                let v = runmat_runtime::elementwise_div(&a_acc, &b_acc)?;
                                stack.push(v)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("mrdivide".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                let v = runmat_runtime::elementwise_div(&a_acc, &b_acc)?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        let v = runmat_runtime::elementwise_div(&a_acc, &b_acc)?;
                        stack.push(v)
                    }
                }
            }
            Instr::Pow => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) | (_, Value::Object(obj)) => {
                        let arg_val = if matches!(&a, Value::Object(_)) {
                            b.clone()
                        } else {
                            a.clone()
                        };
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("power".to_string()),
                            arg_val,
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::power(&a, &b)?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        let v = runmat_runtime::power(&a_acc, &b_acc)?;
                        stack.push(v)
                    }
                }
            }
            Instr::Neg => {
                let value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match &value {
                    Value::Object(obj) => {
                        let args = vec![Value::Object(obj.clone())];
                        match call_builtin("uminus", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let result = runmat_runtime::elementwise_neg(&value)?;
                                stack.push(result)
                            }
                        }
                    }
                    _ => {
                        let result = runmat_runtime::elementwise_neg(&value)?;
                        stack.push(result);
                    }
                }
            }
            Instr::UPlus => {
                let value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match &value {
                    Value::Object(obj) => {
                        let args = vec![Value::Object(obj.clone())];
                        match call_builtin("uplus", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => stack.push(value),
                        }
                    }
                    _ => stack.push(value),
                }
            }
            Instr::Transpose => {
                let value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value)?;
                let result = runmat_runtime::transpose(promoted)?;
                stack.push(result);
            }
            Instr::ElemMul => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("times".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                stack.push(runmat_runtime::elementwise_mul(&a_acc, &b_acc)?)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("times".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                stack.push(runmat_runtime::elementwise_mul(&a_acc, &b_acc)?)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        stack.push(runmat_runtime::elementwise_mul(&a_acc, &b_acc)?)
                    }
                }
            }
            Instr::ElemDiv => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("rdivide".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                stack.push(runmat_runtime::elementwise_div(&a_acc, &b_acc)?)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("rdivide".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                stack.push(runmat_runtime::elementwise_div(&a_acc, &b_acc)?)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        stack.push(runmat_runtime::elementwise_div(&a_acc, &b_acc)?)
                    }
                }
            }
            Instr::ElemPow => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) | (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            if matches!(&a, Value::Object(_)) {
                                b.clone()
                            } else {
                                a.clone()
                            },
                        ];
                        match call_builtin("power", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                stack.push(runmat_runtime::elementwise_pow(&a_acc, &b_acc)?)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        stack.push(runmat_runtime::elementwise_pow(&a_acc, &b_acc)?)
                    }
                }
            }
            Instr::ElemLeftDiv => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ldivide".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (b_acc, a_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a)?;
                                stack.push(runmat_runtime::elementwise_div(&b_acc, &a_acc)?)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ldivide".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (b_acc, a_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a)?;
                                stack.push(runmat_runtime::elementwise_div(&b_acc, &a_acc)?)
                            }
                        }
                    }
                    _ => {
                        let (b_acc, a_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a)?;
                        stack.push(runmat_runtime::elementwise_div(&b_acc, &a_acc)?)
                    }
                }
            }
            Instr::LessEqual => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("le".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: le(a,b) = ~gt(a,b)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("gt".to_string()),
                                    b.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa <= bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ge".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ge(b,a) = ~lt(b,a) hence le(a,b) = ge(b,a)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("lt".to_string()),
                                    a.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa <= bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        let bb: f64 = (&b).try_into()?;
                        let aa: f64 = (&a).try_into()?;
                        stack.push(Value::Num(if aa <= bb { 1.0 } else { 0.0 }));
                    }
                }
            }
            Instr::Less => {
                handle_rel_binary!(<, "lt", stack);
            }
            Instr::Greater => {
                handle_rel_binary!(>, "gt", stack);
            }
            Instr::GreaterEqual => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ge".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ge(a,b) = ~lt(a,b)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("lt".to_string()),
                                    b.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa >= bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("le".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: le(b,a) = ~gt(b,a); hence ge(a,b) = le(b,a)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("gt".to_string()),
                                    a.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa >= bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        let bb: f64 = (&b).try_into()?;
                        let aa: f64 = (&a).try_into()?;
                        stack.push(Value::Num(if aa >= bb { 1.0 } else { 0.0 }));
                    }
                }
            }
            Instr::Equal => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("eq".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let aa: f64 = (&a).try_into()?;
                                let bb: f64 = (&b).try_into()?;
                                stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }))
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("eq".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let aa: f64 = (&a).try_into()?;
                                let bb: f64 = (&b).try_into()?;
                                stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }))
                            }
                        }
                    }
                    (Value::HandleObject(_), _) | (_, Value::HandleObject(_)) => {
                        // Delegate to runtime eq builtin which implements identity semantics
                        let v = runmat_runtime::call_builtin("eq", &[a.clone(), b.clone()])?;
                        stack.push(v);
                    }
                    (Value::Tensor(ta), Value::Tensor(tb)) => {
                        // Element-wise eq; shapes must match
                        if ta.shape != tb.shape {
                            return Err(mex(
                                "ShapeMismatch",
                                "shape mismatch for element-wise comparison",
                            ));
                        }
                        let mut out = Vec::with_capacity(ta.data.len());
                        for i in 0..ta.data.len() {
                            out.push(if (ta.data[i] - tb.data[i]).abs() < 1e-12 {
                                1.0
                            } else {
                                0.0
                            });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, ta.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::Tensor(t), Value::Num(_)) | (Value::Tensor(t), Value::Int(_)) => {
                        let s = match &b {
                            Value::Num(n) => *n,
                            Value::Int(i) => i.to_f64(),
                            _ => 0.0,
                        };
                        let out: Vec<f64> = t
                            .data
                            .iter()
                            .map(|x| if (*x - s).abs() < 1e-12 { 1.0 } else { 0.0 })
                            .collect();
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, t.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::Num(_), Value::Tensor(t)) | (Value::Int(_), Value::Tensor(t)) => {
                        let s = match &a {
                            Value::Num(n) => *n,
                            Value::Int(i) => i.to_f64(),
                            _ => 0.0,
                        };
                        let out: Vec<f64> = t
                            .data
                            .iter()
                            .map(|x| if (s - *x).abs() < 1e-12 { 1.0 } else { 0.0 })
                            .collect();
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, t.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::StringArray(sa), Value::StringArray(sb)) => {
                        if sa.shape != sb.shape {
                            return Err(mex(
                                "ShapeMismatch",
                                "shape mismatch for string array comparison",
                            ));
                        }
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if sa.data[i] == sb.data[i] { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::StringArray(sa), Value::String(s)) => {
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if sa.data[i] == *s { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::String(s), Value::StringArray(sa)) => {
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if *s == sa.data[i] { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::String(a_s), Value::String(b_s)) => {
                        stack.push(Value::Num(if a_s == b_s { 1.0 } else { 0.0 }));
                    }
                    _ => {
                        let bb: f64 = (&b).try_into()?;
                        let aa: f64 = (&a).try_into()?;
                        stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }));
                    }
                }
            }
            Instr::NotEqual => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ne".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ne(a,b) = ~eq(a,b)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("eq".to_string()),
                                    b.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ne".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ne(b,a) = ~eq(b,a)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("eq".to_string()),
                                    a.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    (Value::HandleObject(_), _) | (_, Value::HandleObject(_)) => {
                        let v = runmat_runtime::call_builtin("ne", &[a.clone(), b.clone()])?;
                        stack.push(v);
                    }
                    (Value::Tensor(ta), Value::Tensor(tb)) => {
                        if ta.shape != tb.shape {
                            return Err(mex(
                                "ShapeMismatch",
                                "shape mismatch for element-wise comparison",
                            ));
                        }
                        let mut out = Vec::with_capacity(ta.data.len());
                        for i in 0..ta.data.len() {
                            out.push(if (ta.data[i] - tb.data[i]).abs() >= 1e-12 {
                                1.0
                            } else {
                                0.0
                            });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, ta.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::Tensor(t), Value::Num(_)) | (Value::Tensor(t), Value::Int(_)) => {
                        let s = match &b {
                            Value::Num(n) => *n,
                            Value::Int(i) => i.to_f64(),
                            _ => 0.0,
                        };
                        let out: Vec<f64> = t
                            .data
                            .iter()
                            .map(|x| if (*x - s).abs() >= 1e-12 { 1.0 } else { 0.0 })
                            .collect();
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, t.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::Num(_), Value::Tensor(t)) | (Value::Int(_), Value::Tensor(t)) => {
                        let s = match &a {
                            Value::Num(n) => *n,
                            Value::Int(i) => i.to_f64(),
                            _ => 0.0,
                        };
                        let out: Vec<f64> = t
                            .data
                            .iter()
                            .map(|x| if (s - *x).abs() >= 1e-12 { 1.0 } else { 0.0 })
                            .collect();
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, t.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::StringArray(sa), Value::StringArray(sb)) => {
                        if sa.shape != sb.shape {
                            return Err(mex(
                                "ShapeMismatch",
                                "shape mismatch for string array comparison",
                            ));
                        }
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if sa.data[i] != sb.data[i] { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::StringArray(sa), Value::String(s)) => {
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if sa.data[i] != *s { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::String(s), Value::StringArray(sa)) => {
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if *s != sa.data[i] { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::String(a_s), Value::String(b_s)) => {
                        stack.push(Value::Num(if a_s != b_s { 1.0 } else { 0.0 }));
                    }
                    _ => {
                        let bb: f64 = (&b).try_into()?;
                        let aa: f64 = (&a).try_into()?;
                        stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
                    }
                }
            }
            Instr::JumpIfFalse(target) => {
                let cond: f64 = (&stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                if cond == 0.0 {
                    pc = target;
                    continue;
                }
            }
            Instr::Jump(target) => {
                pc = target;
                continue;
            }
            Instr::CallBuiltin(name, arg_count) => {
                if name == "nargin" {
                    if arg_count != 0 {
                        vm_bail!(mex("TooManyInputs", "nargin takes no arguments"));
                    }
                    let (nin, _) =
                        CALL_COUNTS.with(|cc| cc.borrow().last().cloned().unwrap_or((0, 0)));
                    stack.push(Value::Num(nin as f64));
                    pc += 1;
                    continue;
                }
                if name == "nargout" {
                    if arg_count != 0 {
                        vm_bail!(mex("TooManyInputs", "nargout takes no arguments"));
                    }
                    let (_, nout) =
                        CALL_COUNTS.with(|cc| cc.borrow().last().cloned().unwrap_or((0, 0)));
                    stack.push(Value::Num(nout as f64));
                    pc += 1;
                    continue;
                }
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                let prepared_primary = accel_prepare_args(&name, &args)?;
                match runmat_runtime::call_builtin(&name, &prepared_primary) {
                    Ok(result) => stack.push(result),
                    Err(e) => {
                        // Specific-import matches: import pkg.foo; name == foo
                        let mut specific_matches: Vec<(String, Vec<Value>, Value)> = Vec::new();
                        for (path, wildcard) in &imports {
                            if *wildcard {
                                continue;
                            }
                            if path.last().map(|s| s.as_str()) == Some(name.as_str()) {
                                let qual = path.join(".");
                                let qual_args = accel_prepare_args(&qual, &prepared_primary)?;
                                match runmat_runtime::call_builtin(&qual, &qual_args) {
                                    Ok(value) => specific_matches.push((qual, qual_args, value)),
                                    Err(_) => {}
                                }
                            }
                        }
                        if specific_matches.len() > 1 {
                            let msg = specific_matches
                                .iter()
                                .map(|(q, _, _)| q.clone())
                                .collect::<Vec<_>>()
                                .join(", ");
                            vm_bail!(format!("ambiguous builtin '{}' via imports: {}", name, msg));
                        }
                        if let Some((_, _, value)) = specific_matches.pop() {
                            stack.push(value);
                        } else {
                            // Wildcard-import matches: import pkg.*; try pkg.name
                            let mut wildcard_matches: Vec<(String, Vec<Value>, Value)> = Vec::new();
                            for (path, wildcard) in &imports {
                                if !*wildcard {
                                    continue;
                                }
                                if path.is_empty() {
                                    continue;
                                }
                                let mut qual = String::new();
                                for (i, part) in path.iter().enumerate() {
                                    if i > 0 {
                                        qual.push('.');
                                    }
                                    qual.push_str(part);
                                }
                                qual.push('.');
                                qual.push_str(&name);
                                let qual_args = accel_prepare_args(&qual, &prepared_primary)?;
                                match runmat_runtime::call_builtin(&qual, &qual_args) {
                                    Ok(value) => wildcard_matches.push((qual, qual_args, value)),
                                    Err(_) => {}
                                }
                            }
                            if wildcard_matches.len() > 1 {
                                let msg = wildcard_matches
                                    .iter()
                                    .map(|(q, _, _)| q.clone())
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                vm_bail!(format!(
                                    "ambiguous builtin '{}' via wildcard imports: {}",
                                    name, msg
                                ));
                            }
                            if let Some((_, _, value)) = wildcard_matches.pop() {
                                stack.push(value);
                            } else {
                                // Special-case: rethrow() without explicit e uses last caught
                                if name == "rethrow" && args.is_empty() {
                                    if let Some(le) = &last_exception {
                                        vm_bail!(format!("{}: {}", le.identifier, le.message));
                                    }
                                }
                                if let Some((catch_pc, catch_var)) = try_stack.pop() {
                                    if let Some(var_idx) = catch_var {
                                        if var_idx >= vars.len() {
                                            vars.resize(var_idx + 1, Value::Num(0.0));
                                        }
                                        let mex = parse_exception(&e);
                                        last_exception = Some(mex.clone());
                                        vars[var_idx] = Value::MException(mex);
                                    }
                                    pc = catch_pc;
                                    continue;
                                } else {
                                    return Err(e);
                                }
                            }
                        }
                    }
                }
            }
            Instr::CallBuiltinExpandLast(name, fixed_argc, num_indices) => {
                // Stack layout: [..., a1, a2, ..., a_fixed, base_for_cell, idx1, idx2, ...]
                // Build args vector by first collecting fixed args, then expanding cell indexing into comma-list
                // Evaluate indices and base
                let mut indices = Vec::with_capacity(num_indices);
                for _ in 0..num_indices {
                    let v = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    indices.push(v);
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                // Collect fixed args
                let mut fixed = Vec::with_capacity(fixed_argc);
                for _ in 0..fixed_argc {
                    fixed.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                fixed.reverse();
                // Evaluate cell indexing, then flatten cell contents to extend args
                let expanded = match (base, indices.len()) {
                    (Value::Cell(ca), 1) => {
                        match &indices[0] {
                            Value::Num(n) => {
                                let i = *n as usize;
                                if i == 0 || i > ca.data.len() {
                                    return Err(mex(
                                        "CellIndexOutOfBounds",
                                        "Cell index out of bounds",
                                    ));
                                }
                                vec![(*ca.data[i - 1]).clone()]
                            }
                            Value::Int(i) => {
                                let iu = i.to_i64() as usize;
                                if iu == 0 || iu > ca.data.len() {
                                    return Err(mex(
                                        "CellIndexOutOfBounds",
                                        "Cell index out of bounds",
                                    ));
                                }
                                vec![(*ca.data[iu - 1]).clone()]
                            }
                            Value::Tensor(t) => {
                                // Treat as list of 1-based indices; expand each
                                let mut out: Vec<Value> = Vec::with_capacity(t.data.len());
                                for &val in &t.data {
                                    let iu = val as usize;
                                    if iu == 0 || iu > ca.data.len() {
                                        return Err(mex(
                                            "CellIndexOutOfBounds",
                                            "Cell index out of bounds",
                                        ));
                                    }
                                    out.push((*ca.data[iu - 1]).clone());
                                }
                                out
                            }
                            _ => return Err(mex("CellIndexType", "Unsupported cell index type")),
                        }
                    }
                    (Value::Cell(ca), 2) => {
                        let r: f64 = (&indices[0]).try_into()?;
                        let c: f64 = (&indices[1]).try_into()?;
                        let (ir, ic) = (r as usize, c as usize);
                        if ir == 0 || ir > ca.rows || ic == 0 || ic > ca.cols {
                            return Err(mex(
                                "CellSubscriptOutOfBounds",
                                "Cell subscript out of bounds",
                            ));
                        }
                        vec![(*ca.data[(ir - 1) * ca.cols + (ic - 1)]).clone()]
                    }
                    (other, _) => {
                        // Route to subsref(obj,'{}',{indices...}) if object
                        match other {
                            Value::Object(obj) => {
                                let cell = runmat_builtins::CellArray::new(
                                    indices.clone(),
                                    1,
                                    indices.len(),
                                )
                                .map_err(|e| format!("subsref build error: {e}"))?;
                                let v = match runmat_runtime::call_builtin(
                                    "call_method",
                                    &[
                                        Value::Object(obj),
                                        Value::String("subsref".to_string()),
                                        Value::String("{}".to_string()),
                                        Value::Cell(cell),
                                    ],
                                ) {
                                    Ok(v) => v,
                                    Err(e) => vm_bail!(e),
                                };
                                vec![v]
                            }
                            _ => {
                                return Err(mex(
                                    "ExpandError",
                                    "CallBuiltinExpandLast requires cell or object cell access",
                                ))
                            }
                        }
                    }
                };
                let mut args = fixed;
                args.extend(expanded.into_iter());
                match call_builtin_auto(&name, &args) {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CallBuiltinExpandAt(name, before_count, num_indices, after_count) => {
                // Stack layout: [..., a1..abefore, base, idx..., a_after...]
                let mut after: Vec<Value> = Vec::with_capacity(after_count);
                for _ in 0..after_count {
                    after.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                after.reverse();
                let mut indices = Vec::with_capacity(num_indices);
                for _ in 0..num_indices {
                    indices.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut before: Vec<Value> = Vec::with_capacity(before_count);
                for _ in 0..before_count {
                    before.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                before.reverse();
                let expanded = match (base, indices.len()) {
                    (Value::Cell(ca), 1) => match &indices[0] {
                        Value::Num(n) => {
                            let idx = *n as usize;
                            if idx == 0 || idx > ca.data.len() {
                                return Err(mex(
                                    "CellIndexOutOfBounds",
                                    "Cell index out of bounds",
                                ));
                            }
                            vec![(*ca.data[idx - 1]).clone()]
                        }
                        Value::Int(i) => {
                            let idx = i.to_i64() as usize;
                            if idx == 0 || idx > ca.data.len() {
                                return Err(mex(
                                    "CellIndexOutOfBounds",
                                    "Cell index out of bounds",
                                ));
                            }
                            vec![(*ca.data[idx - 1]).clone()]
                        }
                        Value::Tensor(t) => {
                            let mut out: Vec<Value> = Vec::with_capacity(t.data.len());
                            for &val in &t.data {
                                let iu = val as usize;
                                if iu == 0 || iu > ca.data.len() {
                                    return Err(mex(
                                        "CellIndexOutOfBounds",
                                        "Cell index out of bounds",
                                    ));
                                }
                                out.push((*ca.data[iu - 1]).clone());
                            }
                            out
                        }
                        _ => return Err(mex("CellIndexType", "Unsupported cell index type")),
                    },
                    (Value::Cell(ca), 2) => {
                        let r: f64 = (&indices[0]).try_into()?;
                        let c: f64 = (&indices[1]).try_into()?;
                        let (ir, ic) = (r as usize, c as usize);
                        if ir == 0 || ir > ca.rows || ic == 0 || ic > ca.cols {
                            return Err(mex(
                                "CellSubscriptOutOfBounds",
                                "Cell subscript out of bounds",
                            ));
                        }
                        vec![(*ca.data[(ir - 1) * ca.cols + (ic - 1)]).clone()]
                    }
                    (Value::Object(obj), _) => {
                        let idx_vals: Vec<Value> = indices
                            .iter()
                            .map(|v| Value::Num((v).try_into().unwrap_or(0.0)))
                            .collect();
                        let cell = runmat_runtime::call_builtin("__make_cell", &idx_vals)?;
                        let v = match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e),
                        };
                        vec![v]
                    }
                    _ => {
                        return Err(mex(
                            "ExpandError",
                            "CallBuiltinExpandAt requires cell or object cell access",
                        ))
                    }
                };
                let mut args = before;
                args.extend(expanded.into_iter());
                args.extend(after.into_iter());
                match call_builtin_auto(&name, &args) {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CallBuiltinExpandMulti(name, specs) => {
                // Build final args by walking specs left-to-right and popping from stack accordingly.
                let mut args: Vec<Value> = Vec::with_capacity(specs.len());
                // We'll reconstruct by first collecting a temporary vector and then reversing (since stack is LIFO)
                let mut temp: Vec<Value> = Vec::new();
                for spec in specs.iter().rev() {
                    if spec.is_expand {
                        let mut indices = Vec::with_capacity(spec.num_indices);
                        for _ in 0..spec.num_indices {
                            indices.push(
                                stack
                                    .pop()
                                    .ok_or(mex("StackUnderflow", "stack underflow"))?,
                            );
                        }
                        indices.reverse();
                        let base = stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?;
                        #[cfg(feature = "native-accel")]
                        clear_residency(&base);
                        let expanded = if spec.expand_all {
                            match base {
                                Value::Cell(ca) => {
                                    ca.data.iter().map(|p| (*(*p)).clone()).collect()
                                }
                                Value::Object(obj) => {
                                    // subsref(obj,'{}', {}) with empty indices; expect a cell or value
                                    let empty = runmat_builtins::CellArray::new(vec![], 1, 0)
                                        .map_err(|e| format!("subsref build error: {e}"))?;
                                    let v = match runmat_runtime::call_builtin(
                                        "call_method",
                                        &[
                                            Value::Object(obj),
                                            Value::String("subsref".to_string()),
                                            Value::String("{}".to_string()),
                                            Value::Cell(empty),
                                        ],
                                    ) {
                                        Ok(v) => v,
                                        Err(e) => vm_bail!(e),
                                    };
                                    match v {
                                        Value::Cell(ca) => {
                                            ca.data.iter().map(|p| (*(*p)).clone()).collect()
                                        }
                                        other => vec![other],
                                    }
                                }
                                _ => return Err(mex(
                                    "ExpandError",
                                    "CallBuiltinExpandMulti requires cell or object for expand_all",
                                )),
                            }
                        } else {
                            match (base, indices.len()) {
                                (Value::Cell(ca), 1) => match &indices[0] {
                                    Value::Num(n) => {
                                        let idx = *n as usize;
                                        if idx == 0 || idx > ca.data.len() {
                                            return Err(mex(
                                                "CellIndexOutOfBounds",
                                                "Cell index out of bounds",
                                            ));
                                        }
                                        vec![(*ca.data[idx - 1]).clone()]
                                    }
                                    Value::Int(i) => {
                                        let idx = i.to_i64() as usize;
                                        if idx == 0 || idx > ca.data.len() {
                                            return Err(mex(
                                                "CellIndexOutOfBounds",
                                                "Cell index out of bounds",
                                            ));
                                        }
                                        vec![(*ca.data[idx - 1]).clone()]
                                    }
                                    Value::Tensor(t) => {
                                        let mut out: Vec<Value> = Vec::with_capacity(t.data.len());
                                        for &val in &t.data {
                                            let iu = val as usize;
                                            if iu == 0 || iu > ca.data.len() {
                                                return Err(mex(
                                                    "CellIndexOutOfBounds",
                                                    "Cell index out of bounds",
                                                ));
                                            }
                                            out.push((*ca.data[iu - 1]).clone());
                                        }
                                        out
                                    }
                                    _ => {
                                        return Err(mex(
                                            "CellIndexType",
                                            "Unsupported cell index type",
                                        ))
                                    }
                                },
                                (Value::Cell(ca), 2) => {
                                    let r: f64 = (&indices[0]).try_into()?;
                                    let c: f64 = (&indices[1]).try_into()?;
                                    let (ir, ic) = (r as usize, c as usize);
                                    if ir == 0 || ir > ca.rows || ic == 0 || ic > ca.cols {
                                        return Err(mex(
                                            "CellSubscriptOutOfBounds",
                                            "Cell subscript out of bounds",
                                        ));
                                    }
                                    vec![(*ca.data[(ir - 1) * ca.cols + (ic - 1)]).clone()]
                                }
                                (Value::Object(obj), _) => {
                                    let idx_vals: Vec<Value> = indices
                                        .iter()
                                        .map(|v| Value::Num((v).try_into().unwrap_or(0.0)))
                                        .collect();
                                    let cell =
                                        runmat_runtime::call_builtin("__make_cell", &idx_vals)?;
                                    let v = match runmat_runtime::call_builtin(
                                        "call_method",
                                        &[
                                            Value::Object(obj),
                                            Value::String("subsref".to_string()),
                                            Value::String("{}".to_string()),
                                            cell,
                                        ],
                                    ) {
                                        Ok(v) => v,
                                        Err(e) => vm_bail!(e),
                                    };
                                    vec![v]
                                }
                                _ => return Err(mex(
                                    "ExpandError",
                                    "CallBuiltinExpandMulti requires cell or object cell access",
                                )),
                            }
                        };
                        for v in expanded {
                            temp.push(v);
                        }
                    } else {
                        temp.push(
                            stack
                                .pop()
                                .ok_or(mex("StackUnderflow", "stack underflow"))?,
                        );
                    }
                }
                temp.reverse();
                args.extend(temp.into_iter());
                match call_builtin_auto(&name, &args) {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::PackToRow(count) => {
                // Pop count values and build a 1xN numeric tensor (Num only; others error)
                let mut vals: Vec<f64> = Vec::with_capacity(count);
                let mut tmp: Vec<Value> = Vec::with_capacity(count);
                for _ in 0..count {
                    tmp.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                tmp.reverse();
                for v in tmp {
                    let n: f64 = (&v).try_into()?;
                    vals.push(n);
                }
                let tens = runmat_builtins::Tensor::new(vals, vec![1, count])
                    .map_err(|e| format!("PackToRow: {e}"))?;
                stack.push(Value::Tensor(tens));
            }
            Instr::PackToCol(count) => {
                let mut vals: Vec<f64> = Vec::with_capacity(count);
                let mut tmp: Vec<Value> = Vec::with_capacity(count);
                for _ in 0..count {
                    tmp.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                tmp.reverse();
                for v in tmp {
                    let n: f64 = (&v).try_into()?;
                    vals.push(n);
                }
                let tens = runmat_builtins::Tensor::new(vals, vec![count, 1])
                    .map_err(|e| format!("PackToCol: {e}"))?;
                stack.push(Value::Tensor(tens));
            }
            Instr::CallFunctionExpandMulti(name, specs) => {
                // Build args via specs, then invoke user function similar to CallFunction
                let mut temp: Vec<Value> = Vec::new();
                for spec in specs.iter().rev() {
                    if spec.is_expand {
                        let mut indices = Vec::with_capacity(spec.num_indices);
                        for _ in 0..spec.num_indices {
                            indices.push(
                                stack
                                    .pop()
                                    .ok_or(mex("StackUnderflow", "stack underflow"))?,
                            );
                        }
                        indices.reverse();
                        let base = stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?;
                        let expanded = if spec.expand_all {
                            match base {
                                Value::Cell(ca) => ca.data.iter().map(|p| (*(*p)).clone()).collect::<Vec<Value>>(),
                                Value::Object(obj) => {
                                    let empty = runmat_builtins::CellArray::new(vec![], 1, 0).map_err(|e| format!("subsref build error: {e}"))?;
                                    let v = match runmat_runtime::call_builtin("call_method", &[
                                        Value::Object(obj),
                                        Value::String("subsref".to_string()),
                                        Value::String("{}".to_string()),
                                        Value::Cell(empty),
                                    ]) { Ok(v) => v, Err(e) => vm_bail!(e) };
                                    match v { Value::Cell(ca) => ca.data.iter().map(|p| (*(*p)).clone()).collect::<Vec<Value>>(), other => vec![other] }
                                }
                                _ => return Err("CallFunctionExpandMulti requires cell or object for expand_all".to_string()),
                            }
                        } else {
                            match (base, indices.len()) {
                                (Value::Cell(ca), 1) => match &indices[0] {
                                    Value::Num(n) => {
                                        let idx = *n as usize;
                                        if idx == 0 || idx > ca.data.len() {
                                            return Err(mex(
                                                "CellIndexOutOfBounds",
                                                "Cell index out of bounds",
                                            ));
                                        }
                                        vec![(*ca.data[idx - 1]).clone()]
                                    }
                                    Value::Int(i) => {
                                        let idx = i.to_i64() as usize;
                                        if idx == 0 || idx > ca.data.len() {
                                            return Err(mex(
                                                "CellIndexOutOfBounds",
                                                "Cell index out of bounds",
                                            ));
                                        }
                                        vec![(*ca.data[idx - 1]).clone()]
                                    }
                                    Value::Tensor(t) => {
                                        let mut out: Vec<Value> = Vec::with_capacity(t.data.len());
                                        for &val in &t.data {
                                            let iu = val as usize;
                                            if iu == 0 || iu > ca.data.len() {
                                                return Err(mex(
                                                    "CellIndexOutOfBounds",
                                                    "Cell index out of bounds",
                                                ));
                                            }
                                            out.push((*ca.data[iu - 1]).clone());
                                        }
                                        out
                                    }
                                    _ => {
                                        return Err(mex(
                                            "CellIndexType",
                                            "Unsupported cell index type",
                                        ))
                                    }
                                },
                                (Value::Cell(ca), 2) => {
                                    let r: f64 = (&indices[0]).try_into()?;
                                    let c: f64 = (&indices[1]).try_into()?;
                                    let (ir, ic) = (r as usize, c as usize);
                                    if ir == 0 || ir > ca.rows || ic == 0 || ic > ca.cols {
                                        return Err(mex(
                                            "CellSubscriptOutOfBounds",
                                            "Cell subscript out of bounds",
                                        ));
                                    }
                                    vec![(*ca.data[(ir - 1) * ca.cols + (ic - 1)]).clone()]
                                }
                                (Value::Object(obj), _) => {
                                    let cell = runmat_builtins::CellArray::new(
                                        indices.clone(),
                                        1,
                                        indices.len(),
                                    )
                                    .map_err(|e| format!("subsref build error: {e}"))?;
                                    let v = match runmat_runtime::call_builtin(
                                        "call_method",
                                        &[
                                            Value::Object(obj),
                                            Value::String("subsref".to_string()),
                                            Value::String("{}".to_string()),
                                            Value::Cell(cell),
                                        ],
                                    ) {
                                        Ok(v) => v,
                                        Err(e) => vm_bail!(e),
                                    };
                                    vec![v]
                                }
                                _ => return Err(
                                    "CallFunctionExpandMulti requires cell or object cell access"
                                        .to_string(),
                                ),
                            }
                        };
                        for v in expanded {
                            temp.push(v);
                        }
                    } else {
                        temp.push(
                            stack
                                .pop()
                                .ok_or(mex("StackUnderflow", "stack underflow"))?,
                        );
                    }
                }
                temp.reverse();
                let args = temp;
                let func: UserFunction = match bytecode.functions.get(&name) {
                    Some(f) => f.clone(),
                    None => vm_bail!(mex(
                        "UndefinedFunction",
                        &format!("Undefined function: {name}")
                    )),
                };
                let var_map = runmat_hir::remapping::create_complete_function_var_map(
                    &func.params,
                    &func.outputs,
                    &func.body,
                );
                let local_var_count = var_map.len();
                let remapped_body =
                    runmat_hir::remapping::remap_function_body(&func.body, &var_map);
                let func_vars_count = local_var_count.max(func.params.len());
                let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                for (i, _param_id) in func.params.iter().enumerate() {
                    if i < args.len() && i < func_vars.len() {
                        func_vars[i] = args[i].clone();
                    }
                }
                for (original_var_id, local_var_id) in &var_map {
                    let local_index = local_var_id.0;
                    let global_index = original_var_id.0;
                    if local_index < func_vars.len() && global_index < vars.len() {
                        let is_parameter = func
                            .params
                            .iter()
                            .any(|param_id| param_id == original_var_id);
                        if !is_parameter {
                            func_vars[local_index] = vars[global_index].clone();
                        }
                    }
                }
                let mut func_var_types = func.var_types.clone();
                if func_var_types.len() < local_var_count {
                    func_var_types.resize(local_var_count, Type::Unknown);
                }
                let func_program = runmat_hir::HirProgram {
                    body: remapped_body,
                    var_types: func_var_types,
                };
                let func_bytecode =
                    crate::compile_with_functions(&func_program, &bytecode.functions)?;
                let func_result_vars = match interpret_function(&func_bytecode, func_vars) {
                    Ok(v) => v,
                    Err(e) => vm_bail!(e),
                };
                if let Some(output_var_id) = func.outputs.first() {
                    let local_output_index = var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);
                    if local_output_index < func_result_vars.len() {
                        stack.push(func_result_vars[local_output_index].clone());
                    } else {
                        stack.push(Value::Num(0.0));
                    }
                } else {
                    stack.push(Value::Num(0.0));
                }
            }
            Instr::CallFunction(name, arg_count) => {
                let func: UserFunction = match bytecode.functions.get(&name) {
                    Some(f) => f.clone(),
                    None => vm_bail!(mex(
                        "UndefinedFunction",
                        &format!("Undefined function: {name}")
                    )),
                };
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                if !func.has_varargin {
                    if arg_count < func.params.len() {
                        vm_bail!(mex(
                            "NotEnoughInputs",
                            &format!(
                                "Function '{name}' expects {} inputs, got {arg_count}",
                                func.params.len()
                            )
                        ));
                    }
                    if arg_count > func.params.len() {
                        vm_bail!(mex(
                            "TooManyInputs",
                            &format!(
                                "Function '{name}' expects {} inputs, got {arg_count}",
                                func.params.len()
                            )
                        ));
                    }
                } else {
                    let min_args = func.params.len().saturating_sub(1);
                    if arg_count < min_args {
                        vm_bail!(mex(
                            "NotEnoughInputs",
                            &format!("Function '{name}' expects at least {min_args} inputs, got {arg_count}")
                        ));
                    }
                }
                let var_map = runmat_hir::remapping::create_complete_function_var_map(
                    &func.params,
                    &func.outputs,
                    &func.body,
                );
                let local_var_count = var_map.len();
                let remapped_body =
                    runmat_hir::remapping::remap_function_body(&func.body, &var_map);
                let func_vars_count = local_var_count.max(func.params.len());
                let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                if func.has_varargin {
                    // All fixed parameters except the last (varargin placeholder) are positional; pack the rest into a cell
                    let fixed = func.params.len().saturating_sub(1);
                    for i in 0..fixed {
                        if i < args.len() && i < func_vars.len() {
                            func_vars[i] = args[i].clone();
                        }
                    }
                    let mut rest: Vec<Value> = if args.len() > fixed {
                        args[fixed..].to_vec()
                    } else {
                        Vec::new()
                    };
                    // Create row cell for varargin
                    let cell = runmat_builtins::CellArray::new(
                        std::mem::take(&mut rest),
                        1,
                        if args.len() > fixed {
                            args.len() - fixed
                        } else {
                            0
                        },
                    )
                    .map_err(|e| format!("varargin: {e}"))?;
                    if fixed < func_vars.len() {
                        func_vars[fixed] = Value::Cell(cell);
                    }
                } else {
                    for (i, _param_id) in func.params.iter().enumerate() {
                        if i < args.len() && i < func_vars.len() {
                            func_vars[i] = args[i].clone();
                        }
                    }
                }
                // Copy referenced globals into local frame
                for (original_var_id, local_var_id) in &var_map {
                    let local_index = local_var_id.0;
                    let global_index = original_var_id.0;
                    if local_index < func_vars.len() && global_index < vars.len() {
                        let is_parameter = func
                            .params
                            .iter()
                            .any(|param_id| param_id == original_var_id);
                        if !is_parameter {
                            func_vars[local_index] = vars[global_index].clone();
                        }
                    }
                }
                // Initialize varargout cell if needed
                if func.has_varargout {
                    if let Some(varargout_oid) = func.outputs.last() {
                        if let Some(local_id) = var_map.get(varargout_oid) {
                            if local_id.0 < func_vars.len() {
                                let empty = runmat_builtins::CellArray::new(vec![], 1, 0)
                                    .map_err(|e| format!("varargout init: {e}"))?;
                                func_vars[local_id.0] = Value::Cell(empty);
                            }
                        }
                    }
                }
                let mut func_var_types = func.var_types.clone();
                if func_var_types.len() < local_var_count {
                    func_var_types.resize(local_var_count, Type::Unknown);
                }
                let func_program = runmat_hir::HirProgram {
                    body: remapped_body,
                    var_types: func_var_types,
                };
                let func_bytecode =
                    crate::compile_with_functions(&func_program, &bytecode.functions)?;
                let func_result_vars = match interpret_function_with_counts(
                    &func_bytecode,
                    func_vars,
                    &name,
                    1,
                    arg_count,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        if let Some((catch_pc, catch_var)) = try_stack.pop() {
                            if let Some(var_idx) = catch_var {
                                if var_idx >= vars.len() {
                                    vars.resize(var_idx + 1, Value::Num(0.0));
                                }
                                let mex = parse_exception(&e);
                                last_exception = Some(mex.clone());
                                vars[var_idx] = Value::MException(mex);
                            }
                            pc = catch_pc;
                            continue;
                        } else {
                            vm_bail!(e);
                        }
                    }
                };
                if func.has_varargout {
                    // Single-output call: return first varargout element if any, else 0
                    // For true multi-assign we already have CallFunctionMulti path
                    let first = func
                        .outputs
                        .first()
                        .and_then(|oid| var_map.get(oid))
                        .map(|lid| lid.0)
                        .unwrap_or(0);
                    if let Some(Value::Cell(ca)) = func_result_vars.get(first) {
                        if !ca.data.is_empty() {
                            stack.push((*ca.data[0]).clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    } else if let Some(v) = func_result_vars.get(first) {
                        stack.push(v.clone());
                    } else {
                        stack.push(Value::Num(0.0));
                    }
                } else if let Some(output_var_id) = func.outputs.first() {
                    let local_output_index = var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);
                    if local_output_index < func_result_vars.len() {
                        stack.push(func_result_vars[local_output_index].clone());
                    } else {
                        stack.push(Value::Num(0.0));
                    }
                } else {
                    vm_bail!(mex(
                        "TooManyOutputs",
                        &format!("Function '{name}' does not return outputs")
                    ));
                }
            }
            Instr::CallFunctionExpandAt(name, before_count, num_indices, after_count) => {
                // Assemble argument list with expansion at position
                let mut after: Vec<Value> = Vec::with_capacity(after_count);
                for _ in 0..after_count {
                    after.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                after.reverse();
                let mut indices = Vec::with_capacity(num_indices);
                for _ in 0..num_indices {
                    indices.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut before: Vec<Value> = Vec::with_capacity(before_count);
                for _ in 0..before_count {
                    before.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                before.reverse();
                let expanded = match (base, indices.len()) {
                    (Value::Cell(ca), 1) => match &indices[0] {
                        Value::Num(n) => {
                            let idx = *n as usize;
                            if idx == 0 || idx > ca.data.len() {
                                return Err(mex(
                                    "CellIndexOutOfBounds",
                                    "Cell index out of bounds",
                                ));
                            }
                            vec![(*ca.data[idx - 1]).clone()]
                        }
                        Value::Int(i) => {
                            let idx = i.to_i64() as usize;
                            if idx == 0 || idx > ca.data.len() {
                                return Err(mex(
                                    "CellIndexOutOfBounds",
                                    "Cell index out of bounds",
                                ));
                            }
                            vec![(*ca.data[idx - 1]).clone()]
                        }
                        Value::Tensor(t) => {
                            let mut out: Vec<Value> = Vec::with_capacity(t.data.len());
                            for &val in &t.data {
                                let iu = val as usize;
                                if iu == 0 || iu > ca.data.len() {
                                    return Err(mex(
                                        "CellIndexOutOfBounds",
                                        "Cell index out of bounds",
                                    ));
                                }
                                out.push((*ca.data[iu - 1]).clone());
                            }
                            out
                        }
                        _ => return Err(mex("CellIndexType", "Unsupported cell index type")),
                    },
                    (Value::Cell(ca), 2) => {
                        let r: f64 = (&indices[0]).try_into()?;
                        let c: f64 = (&indices[1]).try_into()?;
                        let (ir, ic) = (r as usize, c as usize);
                        if ir == 0 || ir > ca.rows || ic == 0 || ic > ca.cols {
                            return Err(mex(
                                "CellSubscriptOutOfBounds",
                                "Cell subscript out of bounds",
                            ));
                        }
                        vec![(*ca.data[(ir - 1) * ca.cols + (ic - 1)]).clone()]
                    }
                    (Value::Object(obj), _) => {
                        let idx_vals: Vec<Value> = indices
                            .iter()
                            .map(|v| Value::Num((v).try_into().unwrap_or(0.0)))
                            .collect();
                        let cell = runmat_runtime::call_builtin("__make_cell", &idx_vals)?;
                        let v = match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e),
                        };
                        vec![v]
                    }
                    _ => {
                        return Err(mex(
                            "ExpandError",
                            "CallBuiltinExpandAt requires cell or object cell access",
                        ))
                    }
                };
                let mut args = before;
                args.extend(expanded.into_iter());
                args.extend(after.into_iter());
                match call_builtin(&name, &args) {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CallFunctionMulti(name, arg_count, out_count) => {
                let func: UserFunction = match bytecode.functions.get(&name) {
                    Some(f) => f.clone(),
                    None => vm_bail!(format!("undefined function: {name}")),
                };
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                if !func.has_varargin {
                    if arg_count < func.params.len() {
                        vm_bail!(mex(
                            "NotEnoughInputs",
                            &format!(
                                "Function '{name}' expects {} inputs, got {arg_count}",
                                func.params.len()
                            )
                        ));
                    }
                    if arg_count > func.params.len() {
                        vm_bail!(mex(
                            "TooManyInputs",
                            &format!(
                                "Function '{name}' expects {} inputs, got {arg_count}",
                                func.params.len()
                            )
                        ));
                    }
                } else if arg_count + 1 < func.params.len() {
                    vm_bail!(mex(
                        "NotEnoughInputs",
                        &format!(
                            "Function '{name}' expects at least {} inputs, got {arg_count}",
                            func.params.len() - 1
                        )
                    ));
                }
                let var_map = runmat_hir::remapping::create_complete_function_var_map(
                    &func.params,
                    &func.outputs,
                    &func.body,
                );
                let local_var_count = var_map.len();
                let remapped_body =
                    runmat_hir::remapping::remap_function_body(&func.body, &var_map);
                let func_vars_count = local_var_count.max(func.params.len());
                let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                if func.has_varargin {
                    let fixed = func.params.len().saturating_sub(1);
                    for i in 0..fixed {
                        if i < args.len() && i < func_vars.len() {
                            func_vars[i] = args[i].clone();
                        }
                    }
                    let mut rest: Vec<Value> = if args.len() > fixed {
                        args[fixed..].to_vec()
                    } else {
                        Vec::new()
                    };
                    let cell = runmat_builtins::CellArray::new(
                        std::mem::take(&mut rest),
                        1,
                        if args.len() > fixed {
                            args.len() - fixed
                        } else {
                            0
                        },
                    )
                    .map_err(|e| format!("varargin: {e}"))?;
                    if fixed < func_vars.len() {
                        func_vars[fixed] = Value::Cell(cell);
                    }
                } else {
                    for (i, _param_id) in func.params.iter().enumerate() {
                        if i < args.len() && i < func_vars.len() {
                            func_vars[i] = args[i].clone();
                        }
                    }
                }
                for (original_var_id, local_var_id) in &var_map {
                    let local_index = local_var_id.0;
                    let global_index = original_var_id.0;
                    if local_index < func_vars.len() && global_index < vars.len() {
                        let is_parameter = func
                            .params
                            .iter()
                            .any(|param_id| param_id == original_var_id);
                        if !is_parameter {
                            func_vars[local_index] = vars[global_index].clone();
                        }
                    }
                }
                // Initialize varargout cell if needed
                if func.has_varargout {
                    if let Some(varargout_oid) = func.outputs.last() {
                        if let Some(local_id) = var_map.get(varargout_oid) {
                            if local_id.0 < func_vars.len() {
                                let empty = runmat_builtins::CellArray::new(vec![], 1, 0)
                                    .map_err(|e| format!("varargout init: {e}"))?;
                                func_vars[local_id.0] = Value::Cell(empty);
                            }
                        }
                    }
                }
                let mut func_var_types = func.var_types.clone();
                if func_var_types.len() < local_var_count {
                    func_var_types.resize(local_var_count, Type::Unknown);
                }
                let func_program = runmat_hir::HirProgram {
                    body: remapped_body,
                    var_types: func_var_types,
                };
                let func_bytecode =
                    crate::compile_with_functions(&func_program, &bytecode.functions)?;
                let func_result_vars = match interpret_function_with_counts(
                    &func_bytecode,
                    func_vars,
                    &name,
                    out_count,
                    arg_count,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        if let Some((catch_pc, catch_var)) = try_stack.pop() {
                            if let Some(var_idx) = catch_var {
                                if var_idx >= vars.len() {
                                    vars.resize(var_idx + 1, Value::Num(0.0));
                                }
                                let mex = parse_exception(&e);
                                last_exception = Some(mex.clone());
                                vars[var_idx] = Value::MException(mex);
                            }
                            pc = catch_pc;
                            continue;
                        } else {
                            vm_bail!(e);
                        }
                    }
                };
                if func.has_varargout {
                    // Push named outputs first (excluding varargout itself), then fill from varargout cell, then pad with 0.0
                    let total_named = func.outputs.len().saturating_sub(1);
                    let mut pushed = 0usize;
                    // Push named outputs in order
                    for i in 0..total_named.min(out_count) {
                        if let Some(oid) = func.outputs.get(i) {
                            if let Some(local_id) = var_map.get(oid) {
                                let idx = local_id.0;
                                let v = func_result_vars
                                    .get(idx)
                                    .cloned()
                                    .unwrap_or(Value::Num(0.0));
                                stack.push(v);
                                pushed += 1;
                            }
                        }
                    }
                    if pushed < out_count {
                        // Now consume from varargout cell (last output)
                        if let Some(varargout_oid) = func.outputs.last() {
                            if let Some(local_id) = var_map.get(varargout_oid) {
                                if let Some(Value::Cell(ca)) = func_result_vars.get(local_id.0) {
                                    let available = ca.data.len();
                                    let need = out_count - pushed;
                                    if need > available {
                                        vm_bail!(mex("VarargoutMismatch", &format!("Function '{name}' returned {available} varargout values, {need} requested")));
                                    }
                                    for vi in 0..need {
                                        stack.push((*ca.data[vi]).clone());
                                        pushed += 1;
                                    }
                                }
                            }
                        }
                    }
                    // No padding
                } else {
                    // Push out_count values; error if requesting more than defined
                    let defined = func.outputs.len();
                    if out_count > defined {
                        vm_bail!(mex(
                            "TooManyOutputs",
                            &format!("Function '{name}' defines {defined} outputs, {out_count} requested")
                        ));
                    }
                    for i in 0..out_count {
                        let v = func
                            .outputs
                            .get(i)
                            .and_then(|oid| var_map.get(oid))
                            .map(|lid| lid.0)
                            .and_then(|idx| func_result_vars.get(idx))
                            .cloned()
                            .unwrap_or(Value::Num(0.0));
                        stack.push(v);
                    }
                }
            }
            Instr::CallBuiltinMulti(name, arg_count, out_count) => {
                // Default behavior: try to call builtin; if success, use first output; pad rest with 0.0
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                // Special-case for 'find' to support [i,j,v] = find(A)
                if name == "find" && !args.is_empty() {
                    match &args[0] {
                        Value::Tensor(t) => {
                            let rows = *t.shape.first().unwrap_or(&1);
                            let _cols = *t.shape.get(1).unwrap_or(&1);
                            let mut lin_idx: Vec<usize> = Vec::new();
                            for (k, &v) in t.data.iter().enumerate() {
                                if v != 0.0 {
                                    lin_idx.push(k + 1);
                                }
                            }
                            if out_count <= 1 {
                                let data: Vec<f64> = lin_idx.iter().map(|&k| k as f64).collect();
                                let tens =
                                    runmat_builtins::Tensor::new(data, vec![lin_idx.len(), 1])
                                        .map_err(|e| format!("find: {e}"))?;
                                stack.push(Value::Tensor(tens));
                                for _ in 1..out_count {
                                    stack.push(Value::Num(0.0));
                                }
                            } else {
                                let mut rows_out: Vec<f64> = Vec::with_capacity(lin_idx.len());
                                let mut cols_out: Vec<f64> = Vec::with_capacity(lin_idx.len());
                                for &k in &lin_idx {
                                    let k0 = k - 1;
                                    let r = (k0 % rows) + 1;
                                    let c = (k0 / rows) + 1;
                                    rows_out.push(r as f64);
                                    cols_out.push(c as f64);
                                }
                                let r_t =
                                    runmat_builtins::Tensor::new(rows_out, vec![lin_idx.len(), 1])
                                        .map_err(|e| format!("find: {e}"))?;
                                let c_t =
                                    runmat_builtins::Tensor::new(cols_out, vec![lin_idx.len(), 1])
                                        .map_err(|e| format!("find: {e}"))?;
                                stack.push(Value::Tensor(r_t));
                                if out_count >= 2 {
                                    stack.push(Value::Tensor(c_t));
                                }
                                if out_count >= 3 {
                                    let mut vals: Vec<f64> = Vec::with_capacity(lin_idx.len());
                                    for &k in &lin_idx {
                                        vals.push(t.data[k - 1]);
                                    }
                                    let v_t =
                                        runmat_builtins::Tensor::new(vals, vec![lin_idx.len(), 1])
                                            .map_err(|e| format!("find: {e}"))?;
                                    stack.push(Value::Tensor(v_t));
                                }
                                // pad beyond 3 if requested
                                if out_count > 3 {
                                    for _ in 3..out_count {
                                        stack.push(Value::Num(0.0));
                                    }
                                }
                            }
                            continue;
                        }
                        _ => { /* fallthrough to generic */ }
                    }
                }
                match call_builtin(&name, &args) {
                    Ok(v) => match v {
                        Value::Tensor(t) => {
                            let mut pushed = 0usize;
                            for &val in t.data.iter() {
                                if pushed >= out_count {
                                    break;
                                }
                                stack.push(Value::Num(val));
                                pushed += 1;
                            }
                            for _ in pushed..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                        Value::Cell(ca) => {
                            let mut pushed = 0usize;
                            for v in &ca.data {
                                if pushed >= out_count {
                                    break;
                                }
                                stack.push((**v).clone());
                                pushed += 1;
                            }
                            for _ in pushed..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                        other => {
                            stack.push(other);
                            for _ in 1..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                    },
                    Err(e) => {
                        // Try wildcard imports resolution similar to CallBuiltin
                        let mut resolved = None;
                        for (path, wildcard) in &imports {
                            if !*wildcard {
                                continue;
                            }
                            let mut qual = String::new();
                            for (i, part) in path.iter().enumerate() {
                                if i > 0 {
                                    qual.push('.');
                                }
                                qual.push_str(part);
                            }
                            qual.push('.');
                            qual.push_str(&name);
                            if let Ok(v) = call_builtin(&qual, &args) {
                                resolved = Some(v);
                                break;
                            }
                        }
                        if let Some(v) = resolved {
                            match v {
                                Value::Tensor(t) => {
                                    let mut pushed = 0usize;
                                    for &val in t.data.iter() {
                                        if pushed >= out_count {
                                            break;
                                        }
                                        stack.push(Value::Num(val));
                                        pushed += 1;
                                    }
                                    for _ in pushed..out_count {
                                        stack.push(Value::Num(0.0));
                                    }
                                }
                                Value::Cell(ca) => {
                                    let mut pushed = 0usize;
                                    for v in &ca.data {
                                        if pushed >= out_count {
                                            break;
                                        }
                                        stack.push((**v).clone());
                                        pushed += 1;
                                    }
                                    for _ in pushed..out_count {
                                        stack.push(Value::Num(0.0));
                                    }
                                }
                                other => {
                                    stack.push(other);
                                    for _ in 1..out_count {
                                        stack.push(Value::Num(0.0));
                                    }
                                }
                            }
                        } else {
                            vm_bail!(e);
                        }
                    }
                }
            }
            Instr::EnterTry(catch_pc, catch_var) => {
                try_stack.push((catch_pc, catch_var));
            }
            Instr::PopTry => {
                try_stack.pop();
            }
            Instr::CreateMatrix(rows, cols) => {
                let total_elements = rows * cols;
                let mut row_major = Vec::with_capacity(total_elements);
                for _ in 0..total_elements {
                    let val: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    row_major.push(val);
                }
                row_major.reverse();
                // Reorder to column-major storage: cm[r + c*rows] = rm[r*cols + c]
                let mut data = vec![0.0; total_elements];
                for r in 0..rows {
                    for c in 0..cols {
                        data[r + c * rows] = row_major[r * cols + c];
                    }
                }
                let matrix = runmat_builtins::Tensor::new_2d(data, rows, cols)
                    .map_err(|e| format!("Matrix creation error: {e}"))?;
                stack.push(Value::Tensor(matrix));
            }
            Instr::CreateMatrixDynamic(num_rows) => {
                let mut row_lengths = Vec::new();
                for _ in 0..num_rows {
                    let row_len: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    row_lengths.push(row_len as usize);
                }
                row_lengths.reverse();
                let mut rows_data = Vec::new();
                for &row_len in row_lengths.iter().rev() {
                    let mut row_values = Vec::new();
                    for _ in 0..row_len {
                        row_values.push(
                            stack
                                .pop()
                                .ok_or(mex("StackUnderflow", "stack underflow"))?,
                        );
                    }
                    row_values.reverse();
                    rows_data.push(row_values);
                }
                rows_data.reverse();
                let result = runmat_runtime::create_matrix_from_values(&rows_data)?;
                stack.push(result);
            }
            Instr::CreateRange(has_step) => {
                if has_step {
                    let end: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    let step: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    let start: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    let range_result = runmat_runtime::create_range(start, Some(step), end)?;
                    stack.push(range_result);
                } else {
                    let end: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    let start: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    let range_result = runmat_runtime::create_range(start, None, end)?;
                    stack.push(range_result);
                }
            }
            Instr::Index(num_indices) => {
                let mut indices = Vec::new();
                let count = num_indices;
                for _ in 0..count {
                    let index_val: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    indices.push(index_val);
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                match base {
                    Value::Object(obj) => {
                        let cell = runmat_builtins::CellArray::new(
                            indices.iter().map(|n| Value::Num(*n)).collect(),
                            1,
                            indices.len(),
                        )
                        .map_err(|e| format!("subsref build error: {e}"))?;
                        match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("()".to_string()),
                                Value::Cell(cell),
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    other => {
                        let result = match runmat_runtime::perform_indexing(&other, &indices) {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e),
                        };
                        stack.push(result);
                    }
                }
            }
            Instr::IndexSlice(dims, numeric_count, colon_mask, end_mask) => {
                let __b = bench_start();
                // Pop numeric indices in reverse order (they were pushed in order), then base
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count {
                    numeric.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                numeric.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Object(obj) => {
                        let cell =
                            runmat_builtins::CellArray::new(numeric.to_vec(), 1, numeric.len())
                                .map_err(|e| format!("subsref build error: {e}"))?;
                        match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("()".to_string()),
                                Value::Cell(cell),
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::Tensor(t) => {
                        let rank = t.shape.len();
                        // Build per-dimension selectors
                        #[derive(Clone)]
                        enum Sel {
                            Colon,
                            Scalar(usize),
                            Indices(Vec<usize>),
                        }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        if dims == 1 {
                            let total = t.data.len();
                            let mut idxs: Vec<usize> = Vec::new();
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon {
                                idxs = (1..=total).collect();
                            } else if is_end {
                                idxs = vec![total];
                            } else if let Some(v) = numeric.first() {
                                match v {
                                    Value::Num(n) => {
                                        let i = *n as isize;
                                        if i < 1 {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        idxs = vec![i as usize];
                                    }
                                    Value::Tensor(idx_t) => {
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == total {
                                            for (i, &val) in idx_t.data.iter().enumerate() {
                                                if val != 0.0 {
                                                    idxs.push(i + 1);
                                                }
                                            }
                                        } else {
                                            for &val in &idx_t.data {
                                                let i = val as isize;
                                                if i < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                idxs.push(i as usize);
                                            }
                                        }
                                    }
                                    _ => vm_bail!(mex(
                                        "UnsupportedIndexType",
                                        "Unsupported index type"
                                    )),
                                }
                            } else {
                                vm_bail!(mex("MissingNumericIndex", "missing numeric index"));
                            }
                            if idxs.iter().any(|&i| i == 0 || i > total) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            if idxs.len() == 1 {
                                stack.push(Value::Num(t.data[idxs[0] - 1]));
                            } else {
                                let mut out = Vec::with_capacity(idxs.len());
                                for &i in &idxs {
                                    out.push(t.data[i - 1]);
                                }
                                let tens = runmat_builtins::Tensor::new(out, vec![idxs.len(), 1])
                                    .map_err(|e| format!("Slice error: {e}"))?;
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
                                    let v = numeric.get(num_iter).ok_or(mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                                    num_iter += 1;
                                    match v {
                                        Value::Num(n) => {
                                            let idx = *n as isize;
                                            if idx < 1 {
                                                return Err(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds",
                                                ));
                                            }
                                            selectors.push(Sel::Scalar(idx as usize));
                                        }
                                        Value::Tensor(idx_t) => {
                                            // Logical mask if length matches dimension
                                            let dim_len = *t.shape.get(d).unwrap_or(&1);
                                            let len = idx_t.shape.iter().product::<usize>();
                                            if len == dim_len {
                                                let mut indices = Vec::new();
                                                for (i, &val) in idx_t.data.iter().enumerate() {
                                                    if val != 0.0 {
                                                        indices.push(i + 1);
                                                    }
                                                }
                                                selectors.push(Sel::Indices(indices));
                                            } else {
                                                // Treat as explicit indices (1-based)
                                                let mut indices = Vec::with_capacity(len);
                                                for &val in &idx_t.data {
                                                    let idx = val as isize;
                                                    if idx < 1 {
                                                        return Err(mex(
                                                            "IndexOutOfBounds",
                                                            "Index out of bounds",
                                                        ));
                                                    }
                                                    indices.push(idx as usize);
                                                }
                                                selectors.push(Sel::Indices(indices));
                                            }
                                        }
                                        Value::LogicalArray(la) => {
                                            let dim_len = *t.shape.get(d).unwrap_or(&1);
                                            if la.data.len() == dim_len {
                                                let mut indices = Vec::new();
                                                for (i, &b) in la.data.iter().enumerate() {
                                                    if b != 0 {
                                                        indices.push(i + 1);
                                                    }
                                                }
                                                selectors.push(Sel::Indices(indices));
                                            } else {
                                                return Err(mex(
                                                    "IndexShape",
                                                    "Logical mask shape mismatch",
                                                ));
                                            }
                                        }
                                        _ => {
                                            return Err(mex(
                                                "UnsupportedIndexType",
                                                "Unsupported index type",
                                            ))
                                        }
                                    }
                                }
                            }
                            // 2-D fast paths
                            if dims == 2 {
                                let rows = if rank >= 1 { t.shape[0] } else { 1 };
                                let cols = if rank >= 2 { t.shape[1] } else { 1 };
                                match (&selectors[0], &selectors[1]) {
                                    // Full column
                                    (Sel::Colon, Sel::Scalar(j)) => {
                                        let j0 = *j - 1;
                                        if j0 >= cols {
                                            return Err(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds",
                                            ));
                                        }
                                        let start = j0 * rows;
                                        let out = t.data[start..start + rows].to_vec();
                                        if out.len() == 1 {
                                            stack.push(Value::Num(out[0]));
                                        } else {
                                            let tens =
                                                runmat_builtins::Tensor::new(out, vec![rows, 1])
                                                    .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        }
                                        bench_end("IndexSlice2D.fast_col", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    // Full row
                                    (Sel::Scalar(i), Sel::Colon) => {
                                        let i0 = *i - 1;
                                        if i0 >= rows {
                                            return Err(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds",
                                            ));
                                        }
                                        let mut out: Vec<f64> = Vec::with_capacity(cols);
                                        for c in 0..cols {
                                            out.push(t.data[i0 + c * rows]);
                                        }
                                        if out.len() == 1 {
                                            stack.push(Value::Num(out[0]));
                                        } else {
                                            let tens =
                                                runmat_builtins::Tensor::new(out, vec![1, cols])
                                                    .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        }
                                        bench_end("IndexSlice2D.fast_row", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    // Full columns subset: A(:, J)
                                    (Sel::Colon, Sel::Indices(js)) => {
                                        // Gather selected full columns into a [rows, |J|] tensor
                                        if js.is_empty() {
                                            let tens = runmat_builtins::Tensor::new(
                                                Vec::new(),
                                                vec![rows, 0],
                                            )
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        } else {
                                            let mut out: Vec<f64> =
                                                Vec::with_capacity(rows * js.len());
                                            for &j in js {
                                                let j0 = j - 1;
                                                if j0 >= cols {
                                                    return Err(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds",
                                                    ));
                                                }
                                                let start = j0 * rows;
                                                out.extend_from_slice(&t.data[start..start + rows]);
                                            }
                                            let tens = runmat_builtins::Tensor::new(
                                                out,
                                                vec![rows, js.len()],
                                            )
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        }
                                        bench_end("IndexSlice2D.fast_cols", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    // Selected rows full: A(I, :)
                                    (Sel::Indices(is), Sel::Colon) => {
                                        // Gather selected rows across all columns into [|I|, cols]
                                        if is.is_empty() {
                                            let tens = runmat_builtins::Tensor::new(
                                                Vec::new(),
                                                vec![0, cols],
                                            )
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        } else {
                                            let mut out: Vec<f64> =
                                                Vec::with_capacity(is.len() * cols);
                                            for c in 0..cols {
                                                for &i in is {
                                                    let i0 = i - 1;
                                                    if i0 >= rows {
                                                        return Err(mex(
                                                            "IndexOutOfBounds",
                                                            "Index out of bounds",
                                                        ));
                                                    }
                                                    out.push(t.data[i0 + c * rows]);
                                                }
                                            }
                                            let tens = runmat_builtins::Tensor::new(
                                                out,
                                                vec![is.len(), cols],
                                            )
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        }
                                        bench_end("IndexSlice2D.fast_rows_multi", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    _ => {}
                                }
                            }
                            {
                                // Compute output shape and gather
                                let mut out_dims: Vec<usize> = Vec::new();
                                let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                                for (d, sel) in selectors.iter().enumerate().take(dims) {
                                    let dim_len = *t.shape.get(d).unwrap_or(&1);
                                    let idxs = match sel {
                                        Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                                        Sel::Scalar(i) => vec![*i],
                                        Sel::Indices(v) => v.clone(),
                                    };
                                    if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                                    }
                                    if idxs.len() > 1 {
                                        out_dims.push(idxs.len());
                                    } else {
                                        out_dims.push(1);
                                    }
                                    per_dim_indices.push(idxs);
                                }
                                let mut out_dims: Vec<usize> =
                                    per_dim_indices.iter().map(|v| v.len()).collect();
                                // 2D mixed selectors shape correction to match MATLAB:
                                // (I, scalar) => column vector [len(I), 1]; (scalar, J) => row vector [1, len(J)]
                                if dims == 2 {
                                    match (
                                        &per_dim_indices[0].as_slice(),
                                        &per_dim_indices[1].as_slice(),
                                    ) {
                                        // I (len>1), scalar
                                        (i_list, j_list)
                                            if i_list.len() > 1 && j_list.len() == 1 =>
                                        {
                                            out_dims = vec![i_list.len(), 1];
                                        }
                                        // scalar, J (len>1)
                                        (i_list, j_list)
                                            if i_list.len() == 1 && j_list.len() > 1 =>
                                        {
                                            out_dims = vec![1, j_list.len()];
                                        }
                                        _ => {}
                                    }
                                }
                                // Strides for column-major order (first dimension fastest)
                                let mut strides: Vec<usize> = vec![0; dims];
                                let full_shape: Vec<usize> = if rank < dims {
                                    let mut s = t.shape.clone();
                                    s.resize(dims, 1);
                                    s
                                } else {
                                    t.shape.clone()
                                };
                                let mut acc = 1usize;
                                for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                                    *stride = acc;
                                    acc *= full_shape[d];
                                }
                                // Cartesian product gather
                                let total_out: usize = out_dims.iter().product();
                                let mut out_data: Vec<f64> = Vec::with_capacity(total_out);
                                if out_dims.contains(&0)
                                    || per_dim_indices.iter().any(|v| v.is_empty())
                                {
                                    // Empty selection on some dimension -> empty tensor
                                    let out_tensor =
                                        runmat_builtins::Tensor::new(out_data, out_dims)
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                    stack.push(Value::Tensor(out_tensor));
                                } else {
                                    fn cartesian<F: FnMut(&[usize])>(
                                        lists: &[Vec<usize>],
                                        mut f: F,
                                    ) {
                                        let dims = lists.len();
                                        let mut idx = vec![0usize; dims];
                                        loop {
                                            let current: Vec<usize> =
                                                (0..dims).map(|d| lists[d][idx[d]]).collect();
                                            f(&current);
                                            // Increment first dimension fastest (column-major order)
                                            let mut d = 0usize;
                                            while d < dims {
                                                idx[d] += 1;
                                                if idx[d] < lists[d].len() {
                                                    break;
                                                }
                                                idx[d] = 0;
                                                d += 1;
                                            }
                                            if d == dims {
                                                break;
                                            }
                                        }
                                    }
                                    cartesian(&per_dim_indices, |multi| {
                                        let mut lin = 0usize;
                                        for d in 0..dims {
                                            let i0 = multi[d] - 1;
                                            lin += i0 * strides[d];
                                        }
                                        out_data.push(t.data[lin]);
                                    });
                                    if out_data.len() == 1 {
                                        stack.push(Value::Num(out_data[0]));
                                    } else {
                                        let out_tensor =
                                            runmat_builtins::Tensor::new(out_data, out_dims)
                                                .map_err(|e| format!("Slice error: {e}"))?;
                                        stack.push(Value::Tensor(out_tensor));
                                    }
                                }
                            }
                        }
                    }
                    Value::StringArray(sa) => {
                        let rank = sa.shape.len();
                        #[derive(Clone)]
                        enum Sel {
                            Colon,
                            Scalar(usize),
                            Indices(Vec<usize>),
                        }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        if dims == 1 {
                            let total = sa.data.len();
                            let mut idxs: Vec<usize> = Vec::new();
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon {
                                idxs = (1..=total).collect();
                            } else if is_end {
                                idxs = vec![total];
                            } else if let Some(v) = numeric.first() {
                                match v {
                                    Value::Num(n) => {
                                        let i = *n as isize;
                                        if i < 1 {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        idxs = vec![i as usize];
                                    }
                                    Value::Tensor(idx_t) => {
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == total {
                                            for (i, &val) in idx_t.data.iter().enumerate() {
                                                if val != 0.0 {
                                                    idxs.push(i + 1);
                                                }
                                            }
                                        } else {
                                            for &val in &idx_t.data {
                                                let i = val as isize;
                                                if i < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                idxs.push(i as usize);
                                            }
                                        }
                                    }
                                    _ => vm_bail!(mex(
                                        "UnsupportedIndexType",
                                        "Unsupported index type"
                                    )),
                                }
                            } else {
                                vm_bail!(mex("MissingNumericIndex", "missing numeric index"));
                            }
                            if idxs.iter().any(|&i| i == 0 || i > total) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            if idxs.len() == 1 {
                                // MATLAB semantics: string array indexing returns a String (double-quoted)
                                stack.push(Value::String(sa.data[idxs[0] - 1].clone()));
                            } else {
                                let mut out: Vec<String> = Vec::with_capacity(idxs.len());
                                for &i in &idxs {
                                    out.push(sa.data[i - 1].clone());
                                }
                                let out_sa =
                                    runmat_builtins::StringArray::new(out, vec![idxs.len(), 1])
                                        .map_err(|e| format!("Slice error: {e}"))?;
                                stack.push(Value::StringArray(out_sa));
                            }
                        } else {
                            for d in 0..dims {
                                let is_colon = (colon_mask & (1u32 << d)) != 0;
                                let is_end = (end_mask & (1u32 << d)) != 0;
                                if is_colon {
                                    selectors.push(Sel::Colon);
                                } else if is_end {
                                    let dim_len = *sa.shape.get(d).unwrap_or(&1);
                                    selectors.push(Sel::Scalar(dim_len));
                                } else {
                                    let v = numeric.get(num_iter).ok_or(mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                                    num_iter += 1;
                                    match v {
                                        Value::Num(n) => {
                                            let idx = *n as isize;
                                            if idx < 1 {
                                                return Err(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds",
                                                ));
                                            }
                                            selectors.push(Sel::Scalar(idx as usize));
                                        }
                                        Value::Tensor(idx_t) => {
                                            let dim_len = *sa.shape.get(d).unwrap_or(&1);
                                            let len = idx_t.shape.iter().product::<usize>();
                                            let is_binary_mask = len == dim_len
                                                && idx_t.data.iter().all(|&x| x == 0.0 || x == 1.0);
                                            if is_binary_mask {
                                                let mut v = Vec::new();
                                                for (i, &val) in idx_t.data.iter().enumerate() {
                                                    if val != 0.0 {
                                                        v.push(i + 1);
                                                    }
                                                }
                                                selectors.push(Sel::Indices(v));
                                            } else {
                                                let mut v = Vec::with_capacity(len);
                                                for &val in &idx_t.data {
                                                    let idx = val as isize;
                                                    if idx < 1 {
                                                        vm_bail!(mex(
                                                            "IndexOutOfBounds",
                                                            "Index out of bounds"
                                                        ));
                                                    }
                                                    v.push(idx as usize);
                                                }
                                                selectors.push(Sel::Indices(v));
                                            }
                                        }
                                        _ => vm_bail!(mex(
                                            "UnsupportedIndexType",
                                            "Unsupported index type"
                                        )),
                                    }
                                }
                            }
                            let mut out_dims: Vec<usize> = Vec::new();
                            let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                            for (d, sel) in selectors.iter().enumerate().take(dims) {
                                let dim_len = *sa.shape.get(d).unwrap_or(&1);
                                let idxs = match sel {
                                    Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                                    Sel::Scalar(i) => vec![*i],
                                    Sel::Indices(v) => v.clone(),
                                };
                                if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                if idxs.len() > 1 {
                                    out_dims.push(idxs.len());
                                } else {
                                    out_dims.push(1);
                                }
                                per_dim_indices.push(idxs);
                            }
                            if dims == 2 {
                                match (
                                    &per_dim_indices[0].as_slice(),
                                    &per_dim_indices[1].as_slice(),
                                ) {
                                    (i_list, j_list) if i_list.len() > 1 && j_list.len() == 1 => {
                                        out_dims = vec![i_list.len(), 1];
                                    }
                                    (i_list, j_list) if i_list.len() == 1 && j_list.len() > 1 => {
                                        out_dims = vec![1, j_list.len()];
                                    }
                                    _ => {}
                                }
                            }
                            let mut strides: Vec<usize> = vec![0; dims];
                            let full_shape: Vec<usize> = if rank < dims {
                                let mut s = sa.shape.clone();
                                s.resize(dims, 1);
                                s
                            } else {
                                sa.shape.clone()
                            };
                            let mut acc = 1usize;
                            for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                                *stride = acc;
                                acc *= full_shape[d];
                            }
                            let total_out: usize = out_dims.iter().product();
                            if total_out == 0 {
                                stack.push(Value::StringArray(
                                    runmat_builtins::StringArray::new(Vec::new(), out_dims)
                                        .map_err(|e| format!("Slice error: {e}"))?,
                                ));
                            } else {
                                fn cartesian<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
                                    let dims = lists.len();
                                    let mut idx = vec![0usize; dims];
                                    loop {
                                        let current: Vec<usize> =
                                            (0..dims).map(|d| lists[d][idx[d]]).collect();
                                        f(&current);
                                        let mut d = 0usize;
                                        while d < dims {
                                            idx[d] += 1;
                                            if idx[d] < lists[d].len() {
                                                break;
                                            }
                                            idx[d] = 0;
                                            d += 1;
                                        }
                                        if d == dims {
                                            break;
                                        }
                                    }
                                }
                                let mut out_data: Vec<String> = Vec::with_capacity(total_out);
                                cartesian(&per_dim_indices, |multi| {
                                    let mut lin = 0usize;
                                    for d in 0..dims {
                                        let i0 = multi[d] - 1;
                                        lin += i0 * strides[d];
                                    }
                                    out_data.push(sa.data[lin].clone());
                                });
                                if out_data.len() == 1 {
                                    stack.push(Value::String(out_data[0].clone()));
                                } else {
                                    let out_sa =
                                        runmat_builtins::StringArray::new(out_data, out_dims)
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                    stack.push(Value::StringArray(out_sa));
                                }
                            }
                        }
                    }
                    other => {
                        // Support 1-D linear indexing and scalar(1) on non-tensors
                        if dims == 1 {
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon {
                                vm_bail!(mex(
                                    "SliceNonTensor",
                                    "Slicing only supported on tensors"
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
                            let v = match runmat_runtime::perform_indexing(&other, &[idx_val]) {
                                Ok(v) => v,
                                Err(_e) => vm_bail!(mex(
                                    "SliceNonTensor",
                                    "Slicing only supported on tensors"
                                )),
                            };
                            stack.push(v);
                        }
                        vm_bail!(mex("SliceNonTensor", "Slicing only supported on tensors"));
                    }
                }
                bench_end("IndexSlice", __b);
            }
            Instr::IndexRangeEnd {
                dims,
                numeric_count,
                colon_mask,
                end_mask,
                range_dims,
                range_has_step,
                end_offsets,
            } => {
                // Pop any numeric scalar indices (reverse), then for each range in reverse push step (if has), start; then base
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count {
                    numeric.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                numeric.reverse();
                // Gather per-range params in reverse order of pushes
                let mut range_params: Vec<(f64, f64)> = Vec::with_capacity(range_dims.len());
                for i in (0..range_dims.len()).rev() {
                    let has_step = range_has_step[i];
                    let step = if has_step {
                        let v = stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?;
                        match v {
                            Value::Num(n) => n,
                            Value::Int(i) => i.to_f64(),
                            Value::Tensor(t) if !t.data.is_empty() => t.data[0],
                            _ => 1.0,
                        }
                    } else {
                        1.0
                    };
                    let v = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let start: f64 = match v {
                        Value::Num(n) => n,
                        Value::Int(i) => i.to_f64(),
                        Value::Tensor(t) if !t.data.is_empty() => t.data[0],
                        _ => 1.0,
                    };
                    range_params.push((start, step));
                }
                range_params.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                match base {
                    Value::Tensor(t) => {
                        let rank = t.shape.len();
                        #[derive(Clone)]
                        enum Sel {
                            Colon,
                            Scalar(usize),
                            Indices(Vec<usize>),
                            Range { start: i64, step: i64, end_off: i64 },
                        }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon {
                                selectors.push(Sel::Colon);
                            } else if is_end {
                                selectors.push(Sel::Scalar(*t.shape.get(d).unwrap_or(&1)));
                            } else if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (st, sp) = range_params[rp_iter];
                                rp_iter += 1;
                                let off = end_offsets[pos];
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
                                let v = numeric
                                    .get(num_iter)
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                num_iter += 1;
                                match v {
                                    Value::Num(n) => {
                                        let idx = *n as isize;
                                        if idx < 1 {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        selectors.push(Sel::Scalar(idx as usize));
                                    }
                                    Value::Tensor(idx_t) => {
                                        let dim_len = *t.shape.get(d).unwrap_or(&1);
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == dim_len {
                                            let mut v = Vec::new();
                                            for (i, &val) in idx_t.data.iter().enumerate() {
                                                if val != 0.0 {
                                                    v.push(i + 1);
                                                }
                                            }
                                            selectors.push(Sel::Indices(v));
                                        } else {
                                            let mut v = Vec::with_capacity(len);
                                            for &val in &idx_t.data {
                                                let idx = val as isize;
                                                if idx < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                v.push(idx as usize);
                                            }
                                            selectors.push(Sel::Indices(v));
                                        }
                                    }
                                    _ => vm_bail!(mex(
                                        "UnsupportedIndexType",
                                        "Unsupported index type"
                                    )),
                                }
                            }
                        }
                        // Materialize per-dim indices, resolving ranges with end_off
                        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                        let full_shape: Vec<usize> = if rank < dims {
                            let mut s = t.shape.clone();
                            s.resize(dims, 1);
                            s
                        } else {
                            t.shape.clone()
                        };
                        for (d, sel) in selectors.iter().enumerate().take(dims) {
                            let dim_len = full_shape[d] as i64;
                            let idxs: Vec<usize> = match sel {
                                Sel::Colon => (1..=full_shape[d]).collect(),
                                Sel::Scalar(i) => vec![*i],
                                Sel::Indices(v) => v.clone(),
                                Sel::Range {
                                    start,
                                    step,
                                    end_off,
                                } => {
                                    let mut v = Vec::new();
                                    let mut cur = *start;
                                    let stp = *step;
                                    let end_i = dim_len - *end_off;
                                    if stp == 0 {
                                        vm_bail!(mex("IndexStepZero", "Index step cannot be zero"));
                                    }
                                    if stp > 0 {
                                        while cur <= end_i {
                                            if cur < 1 || cur > dim_len {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    } else {
                                        while cur >= end_i {
                                            if cur < 1 || cur > dim_len {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    }
                                    v
                                }
                            };
                            if idxs.iter().any(|&i| i == 0 || i > full_shape[d]) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            per_dim_indices.push(idxs);
                        }
                        // Strides and gather
                        let mut strides: Vec<usize> = vec![0; dims];
                        let mut acc = 1usize;
                        for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                            *stride = acc;
                            acc *= full_shape[d];
                        }
                        let total_out: usize = per_dim_indices.iter().map(|v| v.len()).product();
                        if total_out == 0 {
                            stack.push(Value::Tensor(
                                runmat_builtins::Tensor::new(Vec::new(), vec![0, 0])
                                    .map_err(|e| format!("Slice error: {e}"))?,
                            ));
                            continue;
                        }
                        let mut out_data: Vec<f64> = Vec::with_capacity(total_out);
                        fn cartesian<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
                            let dims = lists.len();
                            let mut idx = vec![0usize; dims];
                            loop {
                                let current: Vec<usize> =
                                    (0..dims).map(|d| lists[d][idx[d]]).collect();
                                f(&current);
                                let mut d = 0usize;
                                while d < dims {
                                    idx[d] += 1;
                                    if idx[d] < lists[d].len() {
                                        break;
                                    }
                                    idx[d] = 0;
                                    d += 1;
                                }
                                if d == dims {
                                    break;
                                }
                            }
                        }
                        cartesian(&per_dim_indices, |multi| {
                            let mut lin = 0usize;
                            for d in 0..dims {
                                let i0 = multi[d] - 1;
                                lin += i0 * strides[d];
                            }
                            out_data.push(t.data[lin]);
                        });
                        if out_data.len() == 1 {
                            stack.push(Value::Num(out_data[0]));
                        } else {
                            let shape: Vec<usize> =
                                per_dim_indices.iter().map(|v| v.len().max(1)).collect();
                            let tens = runmat_builtins::Tensor::new(out_data, shape)
                                .map_err(|e| format!("Slice error: {e}"))?;
                            stack.push(Value::Tensor(tens));
                        }
                    }
                    Value::StringArray(sa) => {
                        let rank = sa.shape.len();
                        #[derive(Clone)]
                        enum Sel {
                            Colon,
                            Scalar(usize),
                            Indices(Vec<usize>),
                            Range { start: i64, step: i64, end_off: i64 },
                        }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon {
                                selectors.push(Sel::Colon);
                            } else if is_end {
                                selectors.push(Sel::Scalar(*sa.shape.get(d).unwrap_or(&1)));
                            } else if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (st, sp) = range_params[rp_iter];
                                rp_iter += 1;
                                let off = end_offsets[pos];
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
                                let v = numeric
                                    .get(num_iter)
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                num_iter += 1;
                                match v {
                                    Value::Num(n) => {
                                        let idx = *n as isize;
                                        if idx < 1 {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        selectors.push(Sel::Scalar(idx as usize));
                                    }
                                    Value::Tensor(idx_t) => {
                                        let dim_len = *sa.shape.get(d).unwrap_or(&1);
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == dim_len {
                                            let mut v = Vec::new();
                                            for (i, &val) in idx_t.data.iter().enumerate() {
                                                if val != 0.0 {
                                                    v.push(i + 1);
                                                }
                                            }
                                            selectors.push(Sel::Indices(v));
                                        } else {
                                            let mut v = Vec::with_capacity(len);
                                            for &val in &idx_t.data {
                                                let idx = val as isize;
                                                if idx < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                v.push(idx as usize);
                                            }
                                            selectors.push(Sel::Indices(v));
                                        }
                                    }
                                    _ => vm_bail!(mex(
                                        "UnsupportedIndexType",
                                        "Unsupported index type"
                                    )),
                                }
                            }
                        }
                        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                        let full_shape: Vec<usize> = if rank < dims {
                            let mut s = sa.shape.clone();
                            s.resize(dims, 1);
                            s
                        } else {
                            sa.shape.clone()
                        };
                        for d in 0..dims {
                            let dim_len = full_shape[d] as i64;
                            let idxs: Vec<usize> = match &selectors[d] {
                                Sel::Colon => (1..=full_shape[d]).collect(),
                                Sel::Scalar(i) => vec![*i],
                                Sel::Indices(v) => v.clone(),
                                Sel::Range {
                                    start,
                                    step,
                                    end_off,
                                } => {
                                    let mut v = Vec::new();
                                    let mut cur = *start;
                                    let stp = *step;
                                    let end_i = dim_len - *end_off;
                                    if stp == 0 {
                                        vm_bail!(mex("IndexStepZero", "Index step cannot be zero"));
                                    }
                                    if stp > 0 {
                                        while cur <= end_i {
                                            if cur < 1 || cur > dim_len {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    } else {
                                        while cur >= end_i {
                                            if cur < 1 || cur > dim_len {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    }
                                    v
                                }
                            };
                            if idxs.iter().any(|&i| i == 0 || i > full_shape[d]) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            per_dim_indices.push(idxs);
                        }
                        let mut strides: Vec<usize> = vec![0; dims];
                        let mut acc = 1usize;
                        for d in 0..dims {
                            strides[d] = acc;
                            acc *= full_shape[d];
                        }
                        let total_out: usize = per_dim_indices.iter().map(|v| v.len()).product();
                        if total_out == 0 {
                            stack.push(Value::StringArray(
                                runmat_builtins::StringArray::new(Vec::new(), vec![0, 0])
                                    .map_err(|e| format!("Slice error: {e}"))?,
                            ));
                            continue;
                        }
                        let mut out_data: Vec<String> = Vec::with_capacity(total_out);
                        fn cartesian<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
                            let dims = lists.len();
                            let mut idx = vec![0usize; dims];
                            loop {
                                let current: Vec<usize> =
                                    (0..dims).map(|d| lists[d][idx[d]]).collect();
                                f(&current);
                                let mut d = 0usize;
                                while d < dims {
                                    idx[d] += 1;
                                    if idx[d] < lists[d].len() {
                                        break;
                                    }
                                    idx[d] = 0;
                                    d += 1;
                                }
                                if d == dims {
                                    break;
                                }
                            }
                        }
                        cartesian(&per_dim_indices, |multi| {
                            let mut lin = 0usize;
                            for d in 0..dims {
                                let i0 = multi[d] - 1;
                                lin += i0 * strides[d];
                            }
                            out_data.push(sa.data[lin].clone());
                        });
                        if out_data.len() == 1 {
                            stack.push(Value::String(out_data[0].clone()));
                        } else {
                            let shape: Vec<usize> =
                                per_dim_indices.iter().map(|v| v.len().max(1)).collect();
                            let sa_out = runmat_builtins::StringArray::new(out_data, shape)
                                .map_err(|e| format!("Slice error: {e}"))?;
                            stack.push(Value::StringArray(sa_out));
                        }
                    }
                    _ => vm_bail!(mex("SliceNonTensor", "Slicing only supported on tensors")),
                }
            }

            Instr::IndexSliceEx(dims, numeric_count, colon_mask, end_mask, end_offsets) => {
                // Like IndexSlice, but apply end arithmetic to specified numeric indices
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count {
                    numeric.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                numeric.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Tensor(t) => {
                        // Adjust numeric indices where specified: end - k
                        let mut adjusted = numeric.clone();
                        for (pos, off) in end_offsets {
                            if let Some(v) = adjusted.get_mut(pos) {
                                // Determine dimension length to apply 'end'
                                // Map numeric positions to dims by skipping colons and 'end' markers
                                let mut seen_numeric = 0usize;
                                let mut dim_for_pos = 0usize;
                                for (d, _) in (0..dims).enumerate().take(dims) {
                                    let is_colon = (colon_mask & (1u32 << d)) != 0;
                                    let is_end = (end_mask & (1u32 << d)) != 0;
                                    if is_colon || is_end {
                                        continue;
                                    }
                                    if seen_numeric == pos {
                                        dim_for_pos = d;
                                        break;
                                    }
                                    seen_numeric += 1;
                                }
                                let dim_len = *t.shape.get(dim_for_pos).unwrap_or(&1);
                                let idx_val = (dim_len as isize) - (off as isize);
                                *v = Value::Num(idx_val as f64);
                            }
                        }
                        // Now reuse IndexSlice logic: push base back and adjusted numerics
                        // Build selectors identical to IndexSlice path
                        let mut tmp_stack = Vec::new();
                        tmp_stack.push(Value::Tensor(t));
                        for v in adjusted {
                            tmp_stack.push(v);
                        }
                        // Swap stacks for reuse: assign and then fallthrough to IndexSlice body via small duplication
                        let mut numeric_vals: Vec<Value> = Vec::new();
                        let count = numeric_count;
                        let mut idx_iter = tmp_stack.into_iter();
                        let base = idx_iter
                            .next()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?;
                        for _ in 0..count {
                            match idx_iter.next() {
                                Some(v) => numeric_vals.push(v),
                                None => return Err(mex("StackUnderflow", "stack underflow")),
                            }
                        }
                        match base {
                            Value::Tensor(t2) => {
                                // Inline small subset of IndexSlice gather for t2
                                let rank = t2.shape.len();
                                #[derive(Clone)]
                                enum Sel {
                                    Colon,
                                    Scalar(usize),
                                    Indices(Vec<usize>),
                                }
                                let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                                let mut num_iter = 0usize;
                                if dims == 1 {
                                    let total = t2.data.len();
                                    let mut idxs: Vec<usize> = Vec::new();
                                    let is_colon = (colon_mask & 1u32) != 0;
                                    let is_end = (end_mask & 1u32) != 0;
                                    if is_colon {
                                        idxs = (1..=total).collect();
                                    } else if is_end {
                                        idxs = vec![total];
                                    } else if let Some(v) = numeric_vals.first() {
                                        match v {
                                            Value::Num(n) => {
                                                let i = *n as isize;
                                                if i < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                idxs = vec![i as usize];
                                            }
                                            Value::Tensor(idx_t) => {
                                                let len = idx_t.shape.iter().product::<usize>();
                                                if len == total {
                                                    for (i, &val) in idx_t.data.iter().enumerate() {
                                                        if val != 0.0 {
                                                            idxs.push(i + 1);
                                                        }
                                                    }
                                                } else {
                                                    for &val in &idx_t.data {
                                                        let i = val as isize;
                                                        if i < 1 {
                                                            vm_bail!(mex(
                                                                "IndexOutOfBounds",
                                                                "Index out of bounds"
                                                            ));
                                                        }
                                                        idxs.push(i as usize);
                                                    }
                                                }
                                            }
                                            _ => vm_bail!(mex(
                                                "UnsupportedIndexType",
                                                "Unsupported index type"
                                            )),
                                        }
                                    } else {
                                        vm_bail!(mex(
                                            "MissingNumericIndex",
                                            "missing numeric index"
                                        ));
                                    }
                                    if idxs.iter().any(|&i| i == 0 || i > total) {
                                        vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                    }
                                    if idxs.len() == 1 {
                                        stack.push(Value::Num(t2.data[idxs[0] - 1]));
                                    } else {
                                        let mut out = Vec::with_capacity(idxs.len());
                                        for &i in &idxs {
                                            out.push(t2.data[i - 1]);
                                        }
                                        let tens =
                                            runmat_builtins::Tensor::new(out, vec![idxs.len(), 1])
                                                .map_err(|e| format!("Slice error: {e}"))?;
                                        stack.push(Value::Tensor(tens));
                                    }
                                } else {
                                    for d in 0..dims {
                                        let is_colon = (colon_mask & (1u32 << d)) != 0;
                                        let is_end = (end_mask & (1u32 << d)) != 0;
                                        if is_colon {
                                            selectors.push(Sel::Colon);
                                        } else if is_end {
                                            let dim_len = *t2.shape.get(d).unwrap_or(&1);
                                            selectors.push(Sel::Scalar(dim_len));
                                        } else {
                                            let v = numeric_vals.get(num_iter).ok_or(mex(
                                                "MissingNumericIndex",
                                                "missing numeric index",
                                            ))?;
                                            num_iter += 1;
                                            match v {
                                                Value::Num(n) => {
                                                    let idx = *n as isize;
                                                    if idx < 1 {
                                                        return Err(mex(
                                                            "IndexOutOfBounds",
                                                            "Index out of bounds",
                                                        ));
                                                    }
                                                    selectors.push(Sel::Scalar(idx as usize));
                                                }
                                                Value::Tensor(idx_t) => {
                                                    let dim_len = *t2.shape.get(d).unwrap_or(&1);
                                                    let len = idx_t.shape.iter().product::<usize>();
                                                    if len == dim_len {
                                                        let mut indices = Vec::new();
                                                        for (i, &val) in
                                                            idx_t.data.iter().enumerate()
                                                        {
                                                            if val != 0.0 {
                                                                indices.push(i + 1);
                                                            }
                                                        }
                                                        selectors.push(Sel::Indices(indices));
                                                    } else {
                                                        let mut indices = Vec::with_capacity(len);
                                                        for &val in &idx_t.data {
                                                            let idx = val as isize;
                                                            if idx < 1 {
                                                                return Err(mex(
                                                                    "IndexOutOfBounds",
                                                                    "Index out of bounds",
                                                                ));
                                                            }
                                                            indices.push(idx as usize);
                                                        }
                                                        selectors.push(Sel::Indices(indices));
                                                    }
                                                }
                                                Value::LogicalArray(la) => {
                                                    let dim_len = *t2.shape.get(d).unwrap_or(&1);
                                                    if la.data.len() == dim_len {
                                                        let mut indices = Vec::new();
                                                        for (i, &b) in la.data.iter().enumerate() {
                                                            if b != 0 {
                                                                indices.push(i + 1);
                                                            }
                                                        }
                                                        selectors.push(Sel::Indices(indices));
                                                    } else {
                                                        return Err(mex(
                                                            "IndexShape",
                                                            "Logical mask shape mismatch",
                                                        ));
                                                    }
                                                }
                                                _ => {
                                                    return Err(mex(
                                                        "UnsupportedIndexType",
                                                        "Unsupported index type",
                                                    ))
                                                }
                                            }
                                        }
                                    }
                                    let mut out_dims: Vec<usize> = Vec::new();
                                    let mut per_dim_indices: Vec<Vec<usize>> =
                                        Vec::with_capacity(dims);
                                    for (d, sel) in selectors.iter().enumerate().take(dims) {
                                        let dim_len = *t2.shape.get(d).unwrap_or(&1);
                                        let idxs = match sel {
                                            Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                                            Sel::Scalar(i) => vec![*i],
                                            Sel::Indices(v) => v.clone(),
                                        };
                                        if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                            return Err(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds",
                                            ));
                                        }
                                        if idxs.len() > 1 {
                                            out_dims.push(idxs.len());
                                        } else {
                                            out_dims.push(1);
                                        }
                                        per_dim_indices.push(idxs);
                                    }
                                    if dims == 2 {
                                        match (
                                            &per_dim_indices[0].as_slice(),
                                            &per_dim_indices[1].as_slice(),
                                        ) {
                                            (i_list, j_list)
                                                if i_list.len() > 1 && j_list.len() == 1 =>
                                            {
                                                out_dims = vec![i_list.len(), 1];
                                            }
                                            (i_list, j_list)
                                                if i_list.len() == 1 && j_list.len() > 1 =>
                                            {
                                                out_dims = vec![1, j_list.len()];
                                            }
                                            _ => {}
                                        }
                                    }
                                    let mut strides: Vec<usize> = vec![0; dims];
                                    let full_shape: Vec<usize> = if rank < dims {
                                        let mut s = t2.shape.clone();
                                        s.resize(dims, 1);
                                        s
                                    } else {
                                        t2.shape.clone()
                                    };
                                    let mut acc = 1usize;
                                    for d in 0..dims {
                                        strides[d] = acc;
                                        acc *= full_shape[d];
                                    }
                                    let total_out: usize = out_dims.iter().product();
                                    let mut out_data: Vec<f64> = Vec::with_capacity(total_out);
                                    if out_dims.contains(&0) {
                                        let out_tensor =
                                            runmat_builtins::Tensor::new(out_data, out_dims)
                                                .map_err(|e| format!("Slice error: {e}"))?;
                                        stack.push(Value::Tensor(out_tensor));
                                    } else {
                                        fn cartesian<F: FnMut(&[usize])>(
                                            lists: &[Vec<usize>],
                                            mut f: F,
                                        ) {
                                            let dims = lists.len();
                                            let mut idx = vec![0usize; dims];
                                            loop {
                                                let current: Vec<usize> =
                                                    (0..dims).map(|d| lists[d][idx[d]]).collect();
                                                f(&current);
                                                let mut d = 0usize;
                                                while d < dims {
                                                    idx[d] += 1;
                                                    if idx[d] < lists[d].len() {
                                                        break;
                                                    }
                                                    idx[d] = 0;
                                                    d += 1;
                                                }
                                                if d == dims {
                                                    break;
                                                }
                                            }
                                        }
                                        cartesian(&per_dim_indices, |multi| {
                                            let mut lin = 0usize;
                                            for d in 0..dims {
                                                let i0 = multi[d] - 1;
                                                lin += i0 * strides[d];
                                            }
                                            out_data.push(t2.data[lin]);
                                        });
                                        if out_data.len() == 1 {
                                            stack.push(Value::Num(out_data[0]));
                                        } else {
                                            let out_tensor =
                                                runmat_builtins::Tensor::new(out_data, out_dims)
                                                    .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(out_tensor));
                                        }
                                    }
                                }
                            }
                            other => {
                                stack.push(other);
                            }
                        }
                    }
                    other => {
                        vm_bail!(mex(
                            "SliceNonTensor",
                            &format!("Slicing only supported on tensors: got {other:?}")
                        ));
                    }
                }
            }
            Instr::Index1DRangeEnd { has_step, offset } => {
                // Legacy 1-D path for end arithmetic
                let step_val: f64 = if has_step {
                    let v: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    v
                } else {
                    1.0
                };
                let start_val: f64 = (&stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Tensor(t) => {
                        let total = t.data.len();
                        let end_idx = (total as i64) - offset; // inclusive
                        let mut out: Vec<f64> = Vec::new();
                        let mut cur = start_val as i64;
                        let step_i = if step_val >= 0.0 {
                            step_val as i64
                        } else {
                            -(step_val.abs() as i64)
                        };
                        if step_i == 0 {
                            return Err(mex("IndexStepZero", "Index step cannot be zero"));
                        }
                        if step_i > 0 {
                            while cur as i64 <= end_idx {
                                let idx0 = cur as usize;
                                if idx0 == 0 || idx0 > total {
                                    break;
                                }
                                out.push(t.data[idx0 - 1]);
                                cur += step_i;
                            }
                        } else {
                            while (cur as i64) >= end_idx {
                                let idx0 = cur as usize;
                                if idx0 == 0 || idx0 > total {
                                    break;
                                }
                                out.push(t.data[idx0 - 1]);
                                cur += step_i;
                            }
                        }
                        if out.len() == 1 {
                            stack.push(Value::Num(out[0]));
                        } else {
                            let tens =
                                runmat_builtins::Tensor::new(out.clone(), vec![out.len(), 1])
                                    .map_err(|e| format!("Range slice error: {e}"))?;
                            stack.push(Value::Tensor(tens));
                        }
                    }
                    _ => vm_bail!(mex("SliceNonTensor", "Slicing only supported on tensors")),
                }
            }
            Instr::StoreSlice(dims, numeric_count, colon_mask, end_mask) => {
                let __b = bench_start();
                // RHS value to scatter, then numeric indices, then base
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count {
                    numeric.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                numeric.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Object(obj) => {
                        let cell =
                            runmat_builtins::CellArray::new(numeric.clone(), 1, numeric.len())
                                .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj.clone()),
                                Value::String("subsasgn".to_string()),
                                Value::String("()".to_string()),
                                Value::Cell(cell.clone()),
                                rhs.clone(),
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(_e) => {
                                // Fallback to direct builtin OverIdx.subsasgn if class method isn't registered
                                // Determine class name and call fully qualified builtin if present
                                let qualified = format!("{}.subsasgn", obj.class_name);
                                match runmat_runtime::call_builtin(
                                    &qualified,
                                    &[
                                        Value::Object(obj),
                                        Value::String("()".to_string()),
                                        Value::Cell(cell),
                                        rhs,
                                    ],
                                ) {
                                    Ok(v2) => stack.push(v2),
                                    Err(e2) => vm_bail!(e2),
                                }
                            }
                        }
                    }
                    Value::Tensor(mut t) => {
                        // Linear 1-D indexing assignment: A(I) = rhs
                        if dims == 1 {
                            let total = t.data.len();
                            // Build linear index list
                            let mut lin_indices: Vec<usize> = Vec::new();
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon {
                                lin_indices = (1..=total).collect();
                            } else if is_end {
                                lin_indices = vec![total];
                            } else {
                                let v = numeric
                                    .first()
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                match v {
                                    Value::Num(n) => {
                                        let i = *n as isize;
                                        if i < 1 || (i as usize) > total {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        lin_indices.push(i as usize);
                                    }
                                    Value::Tensor(idx_t) => {
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == total {
                                            for (i, &val) in idx_t.data.iter().enumerate() {
                                                if val != 0.0 {
                                                    lin_indices.push(i + 1);
                                                }
                                            }
                                        } else {
                                            for &val in &idx_t.data {
                                                let i = val as isize;
                                                if i < 1 || (i as usize) > total {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                lin_indices.push(i as usize);
                                            }
                                        }
                                    }
                                    _ => vm_bail!(mex(
                                        "UnsupportedIndexType",
                                        "Unsupported index type"
                                    )),
                                }
                            }
                            // Scatter RHS
                            match rhs {
                                Value::Num(v) => {
                                    for &li in &lin_indices {
                                        t.data[li - 1] = v;
                                    }
                                }
                                Value::Tensor(rt) => {
                                    if rt.data.len() == 1 {
                                        let v = rt.data[0];
                                        for &li in &lin_indices {
                                            t.data[li - 1] = v;
                                        }
                                    } else if rt.data.len() == lin_indices.len() {
                                        for (k, &li) in lin_indices.iter().enumerate() {
                                            t.data[li - 1] = rt.data[k];
                                        }
                                    } else {
                                        vm_bail!(
                                            "shape mismatch for linear slice assign".to_string()
                                        );
                                    }
                                }
                                _ => vm_bail!("rhs must be numeric or tensor".into()),
                            }
                            stack.push(Value::Tensor(t));
                        } else {
                            let rank = t.shape.len();
                            #[derive(Clone)]
                            enum Sel {
                                Colon,
                                Scalar(usize),
                                Indices(Vec<usize>),
                            }
                            let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                            let mut num_iter = 0usize;
                            for d in 0..dims {
                                let is_colon = (colon_mask & (1u32 << d)) != 0;
                                let is_end = (end_mask & (1u32 << d)) != 0;
                                if is_colon {
                                    selectors.push(Sel::Colon);
                                } else if is_end {
                                    selectors.push(Sel::Scalar(*t.shape.get(d).unwrap_or(&1)));
                                } else {
                                    let v = numeric.get(num_iter).ok_or(mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                                    num_iter += 1;
                                    match v {
                                        Value::Num(n) => {
                                            let idx = *n as isize;
                                            if idx < 1 {
                                                vm_bail!(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds"
                                                ));
                                            }
                                            selectors.push(Sel::Scalar(idx as usize));
                                        }
                                        Value::Tensor(idx_t) => {
                                            let dim_len = *t.shape.get(d).unwrap_or(&1);
                                            let len = idx_t.shape.iter().product::<usize>();
                                            if len == dim_len {
                                                let mut v = Vec::new();
                                                for (i, &val) in idx_t.data.iter().enumerate() {
                                                    if val != 0.0 {
                                                        v.push(i + 1);
                                                    }
                                                }
                                                selectors.push(Sel::Indices(v));
                                            } else {
                                                let mut v = Vec::with_capacity(len);
                                                for &val in &idx_t.data {
                                                    let idx = val as isize;
                                                    if idx < 1 {
                                                        vm_bail!(mex(
                                                            "IndexOutOfBounds",
                                                            "Index out of bounds"
                                                        ));
                                                    }
                                                    v.push(idx as usize);
                                                }
                                                selectors.push(Sel::Indices(v));
                                            }
                                        }
                                        _ => vm_bail!(mex(
                                            "UnsupportedIndexType",
                                            "Unsupported index type"
                                        )),
                                    }
                                }
                            }
                            // 2-D write fast paths (full column/row) with strict broadcast checks
                            if dims == 2 {
                                let rows = if rank >= 1 { t.shape[0] } else { 1 };
                                let cols = if rank >= 2 { t.shape[1] } else { 1 };
                                match (&selectors[0], &selectors[1]) {
                                    // A(:, j) = rhs
                                    (Sel::Colon, Sel::Scalar(j)) => {
                                        let j0 = *j - 1;
                                        if j0 >= cols {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        let start = j0 * rows;
                                        match rhs {
                                            Value::Num(v) => {
                                                for r in 0..rows {
                                                    t.data[start + r] = v;
                                                }
                                            }
                                            Value::Tensor(rt) => {
                                                let len = rt.data.len();
                                                if len == rows {
                                                    for r in 0..rows {
                                                        t.data[start + r] = rt.data[r];
                                                    }
                                                } else if len == 1 {
                                                    for r in 0..rows {
                                                        t.data[start + r] = rt.data[0];
                                                    }
                                                } else {
                                                    vm_bail!(
                                                        "shape mismatch for slice assign".into()
                                                    );
                                                }
                                            }
                                            _ => vm_bail!("rhs must be numeric or tensor".into()),
                                        }
                                        stack.push(Value::Tensor(t));
                                        bench_end("StoreSlice2D.fast_col", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    // A(i, :) = rhs
                                    (Sel::Scalar(i), Sel::Colon) => {
                                        let i0 = *i - 1;
                                        if i0 >= rows {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        match rhs {
                                            Value::Num(v) => {
                                                for c in 0..cols {
                                                    t.data[i0 + c * rows] = v;
                                                }
                                            }
                                            Value::Tensor(rt) => {
                                                let len = rt.data.len();
                                                if len == cols {
                                                    for c in 0..cols {
                                                        t.data[i0 + c * rows] = rt.data[c];
                                                    }
                                                } else if len == 1 {
                                                    for c in 0..cols {
                                                        t.data[i0 + c * rows] = rt.data[0];
                                                    }
                                                } else {
                                                    vm_bail!(
                                                        "shape mismatch for slice assign".into()
                                                    );
                                                }
                                            }
                                            _ => vm_bail!("rhs must be numeric or tensor".into()),
                                        }
                                        stack.push(Value::Tensor(t));
                                        bench_end("StoreSlice2D.fast_row", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    _ => {}
                                }
                            }
                            // Generic N-D writer path
                            // Build per-dim index lists and strides
                            let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                            let full_shape: Vec<usize> = if rank < dims {
                                let mut s = t.shape.clone();
                                s.resize(dims, 1);
                                s
                            } else {
                                t.shape.clone()
                            };
                            for d in 0..dims {
                                let dim_len = full_shape[d];
                                let idxs = match &selectors[d] {
                                    Sel::Colon => (1..=dim_len).collect(),
                                    Sel::Scalar(i) => vec![*i],
                                    Sel::Indices(v) => v.clone(),
                                };
                                if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                    vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                per_dim_indices.push(idxs);
                            }
                            // Column-major strides (first dimension fastest)
                            let mut strides: Vec<usize> = vec![0; dims];
                            let mut acc = 1usize;
                            for d in 0..dims {
                                strides[d] = acc;
                                acc *= full_shape[d];
                            }
                            let total_out: usize =
                                per_dim_indices.iter().map(|v| v.len()).product();
                            // Prepare RHS values
                            enum RhsView {
                                Scalar(f64),
                                Tensor {
                                    data: Vec<f64>,
                                    shape: Vec<usize>,
                                    strides: Vec<usize>,
                                },
                            }
                            let rhs_view = match rhs {
                                Value::Num(n) => RhsView::Scalar(n),
                                Value::Tensor(rt) => {
                                    // Allow exact match or N-D broadcasting where rhs_dim is 1 or equals out_dim
                                    let mut shape = rt.shape.clone();
                                    if shape.len() < dims {
                                        shape.resize(dims, 1);
                                    }
                                    if shape.len() > dims {
                                        if shape.iter().skip(dims).any(|&s| s != 1) {
                                            vm_bail!("shape mismatch for slice assign".into());
                                        }
                                        shape.truncate(dims);
                                    }
                                    let mut ok = true;
                                    for d in 0..dims {
                                        let out_len = per_dim_indices[d].len();
                                        let rhs_len = shape[d];
                                        if !(rhs_len == 1 || rhs_len == out_len) {
                                            ok = false;
                                            break;
                                        }
                                    }
                                    if !ok {
                                        vm_bail!("shape mismatch for slice assign".into());
                                    }
                                    let mut rstrides = vec![0usize; dims];
                                    let mut racc = 1usize;
                                    for d in 0..dims {
                                        rstrides[d] = racc;
                                        racc *= shape[d];
                                    }
                                    RhsView::Tensor {
                                        data: rt.data,
                                        shape,
                                        strides: rstrides,
                                    }
                                }
                                _ => vm_bail!("rhs must be numeric or tensor".into()),
                            };
                            // Iterate and scatter
                            let mut _k = 0usize;
                            let mut idx = vec![0usize; dims];
                            if total_out == 0 {
                                stack.push(Value::Tensor(t));
                            } else {
                                loop {
                                    let mut lin = 0usize;
                                    for d in 0..dims {
                                        let i0 = per_dim_indices[d][idx[d]] - 1;
                                        lin += i0 * strides[d];
                                    }
                                    match &rhs_view {
                                        RhsView::Scalar(val) => t.data[lin] = *val,
                                        RhsView::Tensor {
                                            data,
                                            shape,
                                            strides,
                                        } => {
                                            let mut rlin = 0usize;
                                            for d in 0..dims {
                                                let rhs_len = shape[d];
                                                let pos = if rhs_len == 1 { 0 } else { idx[d] };
                                                rlin += pos * strides[d];
                                            }
                                            t.data[lin] = data[rlin];
                                        }
                                    }
                                    _k += 1;
                                    // Increment first dim fastest
                                    let mut d = 0usize;
                                    while d < dims {
                                        idx[d] += 1;
                                        if idx[d] < per_dim_indices[d].len() {
                                            break;
                                        }
                                        idx[d] = 0;
                                        d += 1;
                                    }
                                    if d == dims {
                                        break;
                                    }
                                }
                                stack.push(Value::Tensor(t));
                            }
                        }
                    }
                    Value::StringArray(mut sa) => {
                        // Linear 1-D indexing assignment on string arrays
                        if dims == 1 {
                            let total = sa.data.len();
                            let mut lin_indices: Vec<usize> = Vec::new();
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon {
                                lin_indices = (1..=total).collect();
                            } else if is_end {
                                lin_indices = vec![total];
                            } else {
                                let v = numeric
                                    .first()
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                match v {
                                    Value::Num(n) => {
                                        let i = *n as isize;
                                        if i < 1 || (i as usize) > total {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        lin_indices.push(i as usize);
                                    }
                                    Value::Tensor(idx_t) => {
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == total {
                                            for (i, &val) in idx_t.data.iter().enumerate() {
                                                if val != 0.0 {
                                                    lin_indices.push(i + 1);
                                                }
                                            }
                                        } else {
                                            for &val in &idx_t.data {
                                                let i = val as isize;
                                                if i < 1 || (i as usize) > total {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                lin_indices.push(i as usize);
                                            }
                                        }
                                    }
                                    _ => vm_bail!(mex(
                                        "UnsupportedIndexType",
                                        "Unsupported index type"
                                    )),
                                }
                            }
                            // Scatter RHS
                            match rhs {
                                Value::String(s) => {
                                    for &li in &lin_indices {
                                        sa.data[li - 1] = s.clone();
                                    }
                                }
                                Value::StringArray(rsa) => {
                                    if rsa.data.len() == 1 {
                                        let s = rsa.data[0].clone();
                                        for &li in &lin_indices {
                                            sa.data[li - 1] = s.clone();
                                        }
                                    } else if rsa.data.len() == lin_indices.len() {
                                        for (k, &li) in lin_indices.iter().enumerate() {
                                            sa.data[li - 1] = rsa.data[k].clone();
                                        }
                                    } else {
                                        vm_bail!(
                                            "shape mismatch for linear slice assign".to_string()
                                        );
                                    }
                                }
                                Value::Num(n) => {
                                    let s = n.to_string();
                                    for &li in &lin_indices {
                                        sa.data[li - 1] = s.clone();
                                    }
                                }
                                Value::Int(i) => {
                                    let s = i.to_i64().to_string();
                                    for &li in &lin_indices {
                                        sa.data[li - 1] = s.clone();
                                    }
                                }
                                _ => vm_bail!("rhs must be string or string array".into()),
                            }
                            stack.push(Value::StringArray(sa));
                        } else {
                            let rank = sa.shape.len();
                            #[derive(Clone)]
                            enum Sel {
                                Colon,
                                Scalar(usize),
                                Indices(Vec<usize>),
                            }
                            let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                            let mut num_iter = 0usize;
                            for d in 0..dims {
                                let is_colon = (colon_mask & (1u32 << d)) != 0;
                                let is_end = (end_mask & (1u32 << d)) != 0;
                                if is_colon {
                                    selectors.push(Sel::Colon);
                                } else if is_end {
                                    selectors.push(Sel::Scalar(*sa.shape.get(d).unwrap_or(&1)));
                                } else {
                                    let v = numeric.get(num_iter).ok_or(mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                                    num_iter += 1;
                                    match v {
                                        Value::Num(n) => {
                                            let idx = *n as isize;
                                            if idx < 1 {
                                                vm_bail!(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds"
                                                ));
                                            }
                                            selectors.push(Sel::Scalar(idx as usize));
                                        }
                                        Value::Tensor(idx_t) => {
                                            let dim_len = *sa.shape.get(d).unwrap_or(&1);
                                            let len = idx_t.shape.iter().product::<usize>();
                                            if len == dim_len {
                                                let mut v = Vec::new();
                                                for (i, &val) in idx_t.data.iter().enumerate() {
                                                    if val != 0.0 {
                                                        v.push(i + 1);
                                                    }
                                                }
                                                selectors.push(Sel::Indices(v));
                                            } else {
                                                let mut v = Vec::with_capacity(len);
                                                for &val in &idx_t.data {
                                                    let idx = val as isize;
                                                    if idx < 1 {
                                                        vm_bail!(mex(
                                                            "IndexOutOfBounds",
                                                            "Index out of bounds"
                                                        ));
                                                    }
                                                    v.push(idx as usize);
                                                }
                                                selectors.push(Sel::Indices(v));
                                            }
                                        }
                                        _ => vm_bail!(mex(
                                            "UnsupportedIndexType",
                                            "Unsupported index type"
                                        )),
                                    }
                                }
                            }
                            // Build per-dim index lists and strides
                            let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                            let full_shape: Vec<usize> = if rank < dims {
                                let mut s = sa.shape.clone();
                                s.resize(dims, 1);
                                s
                            } else {
                                sa.shape.clone()
                            };
                            for d in 0..dims {
                                let dim_len = full_shape[d];
                                let idxs = match &selectors[d] {
                                    Sel::Colon => (1..=dim_len).collect(),
                                    Sel::Scalar(i) => vec![*i],
                                    Sel::Indices(v) => v.clone(),
                                };
                                if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                    vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                per_dim_indices.push(idxs);
                            }
                            // Column-major strides
                            let mut strides: Vec<usize> = vec![0; dims];
                            let mut acc = 1usize;
                            for d in 0..dims {
                                strides[d] = acc;
                                acc *= full_shape[d];
                            }
                            // Prepare RHS view
                            enum RhsViewS {
                                Scalar(String),
                                Array {
                                    data: Vec<String>,
                                    shape: Vec<usize>,
                                    strides: Vec<usize>,
                                },
                            }
                            let rhs_view = match rhs {
                                Value::String(s) => RhsViewS::Scalar(s),
                                Value::StringArray(rsa) => {
                                    let mut shape = rsa.shape.clone();
                                    if shape.len() < dims {
                                        shape.resize(dims, 1);
                                    }
                                    if shape.len() > dims {
                                        if shape.iter().skip(dims).any(|&s| s != 1) {
                                            vm_bail!("shape mismatch for slice assign".into());
                                        }
                                        shape.truncate(dims);
                                    }
                                    for d in 0..dims {
                                        let out_len = per_dim_indices[d].len();
                                        let rhs_len = shape[d];
                                        if !(rhs_len == 1 || rhs_len == out_len) {
                                            vm_bail!("shape mismatch for slice assign".into());
                                        }
                                    }
                                    let mut rstrides = vec![0usize; dims];
                                    let mut racc = 1usize;
                                    for d in 0..dims {
                                        rstrides[d] = racc;
                                        racc *= shape[d];
                                    }
                                    RhsViewS::Array {
                                        data: rsa.data,
                                        shape,
                                        strides: rstrides,
                                    }
                                }
                                Value::Num(n) => RhsViewS::Scalar(n.to_string()),
                                Value::Int(i) => RhsViewS::Scalar(i.to_i64().to_string()),
                                Value::Tensor(rt) => {
                                    // Convert numeric tensor to strings with broadcast
                                    let mut shape = rt.shape.clone();
                                    if shape.len() < dims {
                                        shape.resize(dims, 1);
                                    }
                                    if shape.len() > dims {
                                        if shape.iter().skip(dims).any(|&s| s != 1) {
                                            vm_bail!("shape mismatch for slice assign".into());
                                        }
                                        shape.truncate(dims);
                                    }
                                    for d in 0..dims {
                                        let out_len = per_dim_indices[d].len();
                                        let rhs_len = shape[d];
                                        if !(rhs_len == 1 || rhs_len == out_len) {
                                            vm_bail!("shape mismatch for slice assign".into());
                                        }
                                    }
                                    let mut rstrides = vec![0usize; dims];
                                    let mut racc = 1usize;
                                    for d in 0..dims {
                                        rstrides[d] = racc;
                                        racc *= shape[d];
                                    }
                                    let data: Vec<String> =
                                        rt.data.iter().map(|x| x.to_string()).collect();
                                    RhsViewS::Array {
                                        data,
                                        shape,
                                        strides: rstrides,
                                    }
                                }
                                _ => vm_bail!("rhs must be string or string array".into()),
                            };
                            // Iterate and scatter
                            let mut idx = vec![0usize; dims];
                            if per_dim_indices.iter().any(|v| v.is_empty()) {
                                stack.push(Value::StringArray(sa));
                            } else {
                                loop {
                                    let mut lin = 0usize;
                                    for d in 0..dims {
                                        let i0 = per_dim_indices[d][idx[d]] - 1;
                                        lin += i0 * strides[d];
                                    }
                                    match &rhs_view {
                                        RhsViewS::Scalar(s) => sa.data[lin] = s.clone(),
                                        RhsViewS::Array {
                                            data,
                                            shape,
                                            strides,
                                        } => {
                                            let mut rlin = 0usize;
                                            for d in 0..dims {
                                                let rhs_len = shape[d];
                                                let pos = if rhs_len == 1 { 0 } else { idx[d] };
                                                rlin += pos * strides[d];
                                            }
                                            sa.data[lin] = data[rlin].clone();
                                        }
                                    }
                                    // Increment first dim fastest
                                    let mut d = 0usize;
                                    while d < dims {
                                        idx[d] += 1;
                                        if idx[d] < per_dim_indices[d].len() {
                                            break;
                                        }
                                        idx[d] = 0;
                                        d += 1;
                                    }
                                    if d == dims {
                                        break;
                                    }
                                }
                                stack.push(Value::StringArray(sa));
                            }
                        }
                    }
                    _ => vm_bail!(
                        "Slicing assignment only supported on tensors or string arrays".into()
                    ),
                }
                bench_end("StoreSlice", __b);
            }
            Instr::StoreSliceEx(dims, numeric_count, colon_mask, end_mask, end_offsets) => {
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count {
                    numeric.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                numeric.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Tensor(t) => {
                        // Adjust numeric indices for end offsets, mapping numeric position to actual dimension
                        let mut adjusted = numeric.clone();
                        for (pos, off) in end_offsets {
                            if let Some(v) = adjusted.get_mut(pos) {
                                // Map numeric index position to dimension index by skipping colon and plain end dims
                                let mut seen_numeric = 0usize;
                                let mut dim_for_pos = 0usize;
                                for d in 0..dims {
                                    let is_colon = (colon_mask & (1u32 << d)) != 0;
                                    let is_end = (end_mask & (1u32 << d)) != 0;
                                    if is_colon || is_end {
                                        continue;
                                    }
                                    if seen_numeric == pos {
                                        dim_for_pos = d;
                                        break;
                                    }
                                    seen_numeric += 1;
                                }
                                let dim_len = *t.shape.get(dim_for_pos).unwrap_or(&1);
                                let idx_val = (dim_len as isize) - (off as isize);
                                *v = Value::Num(idx_val as f64);
                            }
                        }
                        // Reuse StoreSlice by pushing base back along with adjusted numerics and rhs
                        stack.push(Value::Tensor(t));
                        for v in adjusted {
                            stack.push(v);
                        }
                        stack.push(rhs);
                        // Fallthrough emulation: replicate logic of StoreSlice with broadcasting
                        let rhs = stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?;
                        let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                        for _ in 0..numeric_count {
                            numeric.push(
                                stack
                                    .pop()
                                    .ok_or(mex("StackUnderflow", "stack underflow"))?,
                            );
                        }
                        numeric.reverse();
                        let base = stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?;
                        match base {
                            Value::Tensor(mut t) => {
                                #[derive(Clone)]
                                enum Sel {
                                    Colon,
                                    Scalar(usize),
                                    Indices(Vec<usize>),
                                }
                                let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                                let mut num_iter = 0usize;
                                for d in 0..dims {
                                    let is_colon = (colon_mask & (1u32 << d)) != 0;
                                    let is_end = (end_mask & (1u32 << d)) != 0;
                                    if is_colon {
                                        selectors.push(Sel::Colon);
                                    } else if is_end {
                                        selectors.push(Sel::Scalar(*t.shape.get(d).unwrap_or(&1)));
                                    } else {
                                        let v = numeric.get(num_iter).ok_or(mex(
                                            "MissingNumericIndex",
                                            "missing numeric index",
                                        ))?;
                                        num_iter += 1;
                                        match v {
                                            Value::Num(n) => {
                                                let idx = *n as isize;
                                                if idx < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                selectors.push(Sel::Scalar(idx as usize));
                                            }
                                            Value::Tensor(idx_t) => {
                                                let dim_len = *t.shape.get(d).unwrap_or(&1);
                                                let len = idx_t.shape.iter().product::<usize>();
                                                if len == dim_len {
                                                    let mut vi = Vec::new();
                                                    for (i, &val) in idx_t.data.iter().enumerate() {
                                                        if val != 0.0 {
                                                            vi.push(i + 1);
                                                        }
                                                    }
                                                    selectors.push(Sel::Indices(vi));
                                                } else {
                                                    let mut vi = Vec::with_capacity(len);
                                                    for &val in &idx_t.data {
                                                        let idx = val as isize;
                                                        if idx < 1 {
                                                            vm_bail!(mex(
                                                                "IndexOutOfBounds",
                                                                "Index out of bounds"
                                                            ));
                                                        }
                                                        vi.push(idx as usize);
                                                    }
                                                    selectors.push(Sel::Indices(vi));
                                                }
                                            }
                                            _ => vm_bail!(mex(
                                                "UnsupportedIndexType",
                                                "Unsupported index type"
                                            )),
                                        }
                                    }
                                }
                                // Compute per-dim indices and strides
                                let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                                for (d, sel) in selectors.iter().enumerate().take(dims) {
                                    let dim_len = *t.shape.get(d).unwrap_or(&1);
                                    let idxs = match sel {
                                        Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                                        Sel::Scalar(i) => vec![*i],
                                        Sel::Indices(v) => v.clone(),
                                    };
                                    per_dim_indices.push(idxs);
                                }
                                let mut strides: Vec<usize> = vec![0; dims];
                                let mut acc = 1usize;
                                for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                                    *stride = acc;
                                    acc *= *t.shape.get(d).unwrap_or(&1);
                                }
                                // Build RHS view with broadcasting like StoreSlice
                                enum RhsView {
                                    Scalar(f64),
                                    Tensor {
                                        data: Vec<f64>,
                                        shape: Vec<usize>,
                                        strides: Vec<usize>,
                                    },
                                }
                                let rhs_view = match rhs {
                                    Value::Num(n) => RhsView::Scalar(n),
                                    Value::Tensor(rt) => {
                                        let mut rshape = rt.shape.clone();
                                        if rshape.len() < dims {
                                            rshape.resize(dims, 1);
                                        }
                                        if rshape.len() > dims {
                                            if rshape.iter().skip(dims).any(|&s| s != 1) {
                                                vm_bail!("shape mismatch for slice assign".into());
                                            }
                                            rshape.truncate(dims);
                                        }
                                        for d in 0..dims {
                                            let out_len = per_dim_indices[d].len();
                                            let rhs_len = rshape[d];
                                            if !(rhs_len == 1 || rhs_len == out_len) {
                                                vm_bail!("shape mismatch for slice assign".into());
                                            }
                                        }
                                        let mut rstrides = vec![0usize; dims];
                                        let mut racc = 1usize;
                                        for d in 0..dims {
                                            rstrides[d] = racc;
                                            racc *= rshape[d];
                                        }
                                        RhsView::Tensor {
                                            data: rt.data,
                                            shape: rshape,
                                            strides: rstrides,
                                        }
                                    }
                                    _ => vm_bail!("rhs must be numeric or tensor".into()),
                                };
                                // Map absolute indices to selection positions per dimension
                                use std::collections::HashMap;
                                let mut pos_maps: Vec<HashMap<usize, usize>> =
                                    Vec::with_capacity(dims);
                                for (_d, dim_idxs) in per_dim_indices.iter().enumerate().take(dims)
                                {
                                    let mut m = HashMap::new();
                                    for (p, &idx) in dim_idxs.iter().enumerate() {
                                        m.insert(idx, p);
                                    }
                                    pos_maps.push(m);
                                }
                                fn cartesian2<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
                                    let dims = lists.len();
                                    let mut idx = vec![0usize; dims];
                                    loop {
                                        let cur: Vec<usize> =
                                            (0..dims).map(|d| lists[d][idx[d]]).collect();
                                        f(&cur);
                                        let mut d = 0usize;
                                        while d < dims {
                                            idx[d] += 1;
                                            if idx[d] < lists[d].len() {
                                                break;
                                            }
                                            idx[d] = 0;
                                            d += 1;
                                        }
                                        if d == dims {
                                            break;
                                        }
                                    }
                                }
                                cartesian2(&per_dim_indices, |multi| {
                                    let mut lin = 0usize;
                                    for d in 0..dims {
                                        let i0 = multi[d] - 1;
                                        lin += i0 * strides[d];
                                    }
                                    match &rhs_view {
                                        RhsView::Scalar(v) => {
                                            t.data[lin] = *v;
                                        }
                                        RhsView::Tensor {
                                            data,
                                            shape,
                                            strides: rstrides,
                                        } => {
                                            let mut rlin = 0usize;
                                            for d in 0..dims {
                                                let rhs_len = shape[d];
                                                let pos_in_dim = if rhs_len == 1 {
                                                    0
                                                } else {
                                                    *pos_maps[d].get(&multi[d]).unwrap_or(&0)
                                                };
                                                rlin += pos_in_dim * rstrides[d];
                                            }
                                            t.data[lin] = data[rlin];
                                        }
                                    }
                                });
                                stack.push(Value::Tensor(t));
                            }
                            other => vm_bail!(format!("StoreSliceEx unsupported base: {other:?}")),
                        }
                    }
                    other => vm_bail!(format!(
                        "StoreSliceEx only supports tensors currently, got {other:?}"
                    )),
                }
            }
            Instr::StoreRangeEnd {
                dims,
                numeric_count,
                colon_mask,
                end_mask,
                range_dims,
                range_has_step,
                end_offsets,
            } => {
                // RHS, range params (per range dim), then base with numeric scalar indices interleaved
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                // Pop per-range params in reverse order
                let mut range_params: Vec<(f64, f64)> = Vec::with_capacity(range_dims.len());
                for i in (0..range_dims.len()).rev() {
                    let has = range_has_step[i];
                    let step = if has {
                        let v: f64 = (&stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?)
                            .try_into()?;
                        v
                    } else {
                        1.0
                    };
                    let st: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    range_params.push((st, step));
                }
                range_params.reverse();
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count {
                    numeric.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                numeric.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                match base {
                    Value::Tensor(mut t) => {
                        #[derive(Clone)]
                        enum Sel {
                            Colon,
                            Scalar(usize),
                            Indices(Vec<usize>),
                            Range { start: i64, step: i64, end_off: i64 },
                        }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (st, sp) = range_params[rp_iter];
                                rp_iter += 1;
                                let step_i = if sp >= 0.0 {
                                    sp as i64
                                } else {
                                    -(sp.abs() as i64)
                                };
                                selectors.push(Sel::Range {
                                    start: st as i64,
                                    step: step_i,
                                    end_off: end_offsets[pos],
                                });
                                continue;
                            }
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon {
                                selectors.push(Sel::Colon);
                                continue;
                            }
                            if is_end {
                                selectors.push(Sel::Scalar(*t.shape.get(d).unwrap_or(&1)));
                                continue;
                            }
                            let v = numeric
                                .get(num_iter)
                                .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                            num_iter += 1;
                            match v {
                                Value::Num(n) => {
                                    let idx = *n as isize;
                                    if idx < 1 {
                                        vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                    }
                                    selectors.push(Sel::Scalar(idx as usize));
                                }
                                Value::Tensor(idx_t) => {
                                    let dim_len = *t.shape.get(d).unwrap_or(&1);
                                    let len = idx_t.shape.iter().product::<usize>();
                                    if len == dim_len {
                                        let mut vi = Vec::new();
                                        for (i, &val) in idx_t.data.iter().enumerate() {
                                            if val != 0.0 {
                                                vi.push(i + 1);
                                            }
                                        }
                                        selectors.push(Sel::Indices(vi));
                                    } else {
                                        let mut vi = Vec::with_capacity(len);
                                        for &val in &idx_t.data {
                                            let idx = val as isize;
                                            if idx < 1 {
                                                vm_bail!(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds"
                                                ));
                                            }
                                            vi.push(idx as usize);
                                        }
                                        selectors.push(Sel::Indices(vi));
                                    }
                                }
                                _ => {
                                    vm_bail!(mex("UnsupportedIndexType", "Unsupported index type"))
                                }
                            }
                        }
                        // Build index lists and scatter rhs with broadcasting
                        // debug removed
                        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                        for (d, sel) in selectors.iter().enumerate().take(dims) {
                            let dim_len = *t.shape.get(d).unwrap_or(&1);
                            let idxs = match sel {
                                Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                                Sel::Scalar(i) => vec![*i],
                                Sel::Indices(v) => v.clone(),
                                Sel::Range {
                                    start,
                                    step,
                                    end_off,
                                } => {
                                    let mut v = Vec::new();
                                    let mut cur = *start;
                                    let end_i = (dim_len as i64) - *end_off;
                                    let stp = *step;
                                    if stp == 0 {
                                        vm_bail!(mex("IndexStepZero", "Index step cannot be zero"));
                                    }
                                    if stp > 0 {
                                        while cur <= end_i {
                                            if cur < 1 || cur > dim_len as i64 {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    } else {
                                        while cur >= end_i {
                                            if cur < 1 || cur > dim_len as i64 {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    }
                                    v
                                }
                            };
                            if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            per_dim_indices.push(idxs);
                        }
                        let mut strides: Vec<usize> = vec![0; dims];
                        let mut acc = 1usize;
                        for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                            *stride = acc;
                            acc *= *t.shape.get(d).unwrap_or(&1);
                        }
                        let selection_empty = per_dim_indices.iter().any(|v| v.is_empty());
                        if selection_empty {
                            stack.push(Value::Tensor(t));
                        } else {
                            // Build broadcasting view for RHS with per-dimension shape
                            enum RhsView {
                                Scalar(f64),
                                Tensor {
                                    data: Vec<f64>,
                                    shape: Vec<usize>,
                                    strides: Vec<usize>,
                                },
                            }
                            let rhs_view = match rhs {
                                Value::Num(n) => RhsView::Scalar(n),
                                Value::Tensor(rt) => {
                                    if rt.data.is_empty() {
                                        vm_bail!("shape mismatch for slice assign".into());
                                    }
                                    // Normalize RHS shape to dims by padding with ones or validating extra dims are ones
                                    let mut rshape = rt.shape.clone();
                                    if rshape.len() < dims {
                                        rshape.resize(dims, 1);
                                    }
                                    if rshape.len() > dims {
                                        if rshape.iter().skip(dims).any(|&s| s != 1) {
                                            vm_bail!("shape mismatch for slice assign".into());
                                        }
                                        rshape.truncate(dims);
                                    }
                                    // Validate broadcasting compatibility
                                    for d in 0..dims {
                                        let out_len = per_dim_indices[d].len();
                                        let rhs_len = rshape[d];
                                        if !(rhs_len == 1 || rhs_len == out_len) {
                                            vm_bail!("shape mismatch for slice assign".into());
                                        }
                                    }
                                    // Build column-major strides for RHS
                                    let mut rstrides = vec![0usize; dims];
                                    let mut racc = 1usize;
                                    for d in 0..dims {
                                        rstrides[d] = racc;
                                        racc *= rshape[d];
                                    }
                                    if racc != rt.data.len() {
                                        vm_bail!("shape mismatch for slice assign".into());
                                    }
                                    RhsView::Tensor {
                                        data: rt.data,
                                        shape: rshape,
                                        strides: rstrides,
                                    }
                                }
                                _ => vm_bail!("rhs must be numeric or tensor".into()),
                            };
                            // Precompute mapping from absolute index to position-in-selection per dimension to ensure column-major consistent mapping
                            use std::collections::HashMap;
                            let mut pos_maps: Vec<HashMap<usize, usize>> = Vec::with_capacity(dims);
                            for dim_idxs in per_dim_indices.iter().take(dims) {
                                let mut m: HashMap<usize, usize> = HashMap::new();
                                for (p, &idx) in dim_idxs.iter().enumerate() {
                                    m.insert(idx, p);
                                }
                                pos_maps.push(m);
                            }
                            fn cartesian2<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
                                let dims = lists.len();
                                let mut idx = vec![0usize; dims];
                                loop {
                                    let cur: Vec<usize> =
                                        (0..dims).map(|d| lists[d][idx[d]]).collect();
                                    f(&cur);
                                    let mut d = 0usize;
                                    while d < dims {
                                        idx[d] += 1;
                                        if idx[d] < lists[d].len() {
                                            break;
                                        }
                                        idx[d] = 0;
                                        d += 1;
                                    }
                                    if d == dims {
                                        break;
                                    }
                                }
                            }
                            // debug removed
                            let mut err_opt: Option<String> = None;
                            let mut _debug_count = 0usize;
                            cartesian2(&per_dim_indices, |multi| {
                                if err_opt.is_some() {
                                    return;
                                }
                                let mut lin = 0usize;
                                for d in 0..dims {
                                    let i0 = multi[d] - 1;
                                    lin += i0 * strides[d];
                                }
                                match &rhs_view {
                                    RhsView::Scalar(val) => t.data[lin] = *val,
                                    RhsView::Tensor {
                                        data,
                                        shape,
                                        strides: rstrides,
                                    } => {
                                        // Map selection coordinate to RHS coordinate with broadcasting
                                        let mut rlin = 0usize;
                                        for d in 0..dims {
                                            let rhs_len = shape[d];
                                            let pos_in_dim = if rhs_len == 1 {
                                                0
                                            } else {
                                                *pos_maps[d].get(&multi[d]).unwrap_or(&0)
                                            };
                                            rlin += pos_in_dim * rstrides[d];
                                        }
                                        if rlin >= data.len() {
                                            err_opt =
                                                Some("shape mismatch for slice assign".to_string());
                                            return;
                                        }
                                        t.data[lin] = data[rlin];
                                    }
                                }
                            });
                            let _ = (t.data.first(), t.data.len());
                            if let Some(e) = err_opt {
                                vm_bail!(e);
                            }
                            stack.push(Value::Tensor(t));
                        }
                    }
                    Value::Object(obj) => {
                        // Build cell of per-dim index descriptors to pass to subsasgn
                        let mut idx_values: Vec<Value> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon {
                                idx_values.push(Value::String(":".to_string()));
                                continue;
                            }
                            if is_end {
                                idx_values.push(Value::String("end".to_string()));
                                continue;
                            }
                            if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (st, sp) = range_params[rp_iter];
                                rp_iter += 1;
                                let off = end_offsets[pos];
                                let cell = runmat_builtins::CellArray::new(
                                    vec![
                                        Value::Num(st),
                                        Value::Num(sp),
                                        Value::String("end".to_string()),
                                        Value::Num(off as f64),
                                    ],
                                    1,
                                    4,
                                )
                                .map_err(|e| format!("obj range: {e}"))?;
                                idx_values.push(Value::Cell(cell));
                            } else {
                                let v = numeric
                                    .get(num_iter)
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                num_iter += 1;
                                match v {
                                    Value::Num(n) => idx_values.push(Value::Num(*n)),
                                    Value::Int(i) => idx_values.push(Value::Num(i.to_f64())),
                                    Value::Tensor(t) => idx_values.push(Value::Tensor(t.clone())),
                                    other => {
                                        return Err(format!(
                                            "Unsupported index type for object: {other:?}"
                                        ))
                                    }
                                }
                            }
                        }
                        let cell = runmat_builtins::CellArray::new(idx_values, 1, dims)
                            .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsasgn".to_string()),
                                Value::String("()".to_string()),
                                Value::Cell(cell),
                                rhs,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    _ => vm_bail!("StoreRangeEnd only supports tensors currently".to_string()),
                }
            }
            Instr::StoreSlice1DRangeEnd { has_step, offset } => {
                // RHS, then start[, step], then base
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let step_val: f64 = if has_step {
                    let v: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    v
                } else {
                    1.0
                };
                let start_val: f64 = (&stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                match base {
                    Value::Tensor(mut t) => {
                        let total = t.data.len();
                        let end_idx = (total as i64) - offset;
                        let mut cur = start_val as i64;
                        let step_i = if step_val >= 0.0 {
                            step_val as i64
                        } else {
                            -(step_val.abs() as i64)
                        };
                        if step_i == 0 {
                            return Err(mex("IndexStepZero", "Index step cannot be zero"));
                        }
                        // Broadcast rhs if scalar
                        let rhs_vals: Vec<f64> = match rhs {
                            Value::Num(n) => vec![n],
                            Value::Tensor(rt) => rt.data.clone(),
                            _ => vec![0.0],
                        };
                        let mut rpos = 0usize;
                        if step_i > 0 {
                            while cur as i64 <= end_idx {
                                let idx0 = cur as usize;
                                if idx0 == 0 || idx0 > total {
                                    break;
                                }
                                let v = rhs_vals
                                    .get(rpos)
                                    .cloned()
                                    .unwrap_or(*rhs_vals.last().unwrap_or(&0.0));
                                t.data[idx0 - 1] = v;
                                rpos += 1;
                                cur += step_i;
                            }
                        } else {
                            while (cur as i64) >= end_idx {
                                let idx0 = cur as usize;
                                if idx0 == 0 || idx0 > total {
                                    break;
                                }
                                let v = rhs_vals
                                    .get(rpos)
                                    .cloned()
                                    .unwrap_or(*rhs_vals.last().unwrap_or(&0.0));
                                t.data[idx0 - 1] = v;
                                rpos += 1;
                                cur += step_i;
                            }
                        }
                        stack.push(Value::Tensor(t));
                    }
                    _ => vm_bail!("Store range with end only supported on tensors".to_string()),
                }
            }
            Instr::CreateCell2D(rows, cols) => {
                let mut elems = Vec::with_capacity(rows * cols);
                for _ in 0..rows * cols {
                    elems.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                elems.reverse();
                let ca = runmat_builtins::CellArray::new(elems, rows, cols)
                    .map_err(|e| format!("Cell creation error: {e}"))?;
                stack.push(Value::Cell(ca));
            }
            Instr::IndexCell(num_indices) => {
                // Pop indices first (in reverse), then base
                let mut indices = Vec::with_capacity(num_indices);
                for _ in 0..num_indices {
                    let v: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    indices.push(v as usize);
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Object(obj) => {
                        // Route to subsref(obj, '{}', {indices})
                        let cell = runmat_runtime::call_builtin(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::Cell(ca) => match indices.len() {
                        1 => {
                            let i = indices[0];
                            if i == 0 || i > ca.data.len() {
                                return Err(mex(
                                    "CellIndexOutOfBounds",
                                    "Cell index out of bounds",
                                ));
                            }
                            stack.push((*ca.data[i - 1]).clone());
                        }
                        2 => {
                            let r = indices[0];
                            let c = indices[1];
                            if r == 0 || r > ca.rows || c == 0 || c > ca.cols {
                                return Err(mex(
                                    "CellSubscriptOutOfBounds",
                                    "Cell subscript out of bounds",
                                ));
                            }
                            stack.push((*ca.data[(r - 1) * ca.cols + (c - 1)]).clone());
                        }
                        _ => return Err("Unsupported number of cell indices".to_string()),
                    },
                    _ => return Err("Cell indexing on non-cell".to_string()),
                }
            }
            Instr::IndexCellExpand(num_indices, out_count) => {
                // Same as IndexCell but flatten cell contents into multiple outputs
                let mut indices = Vec::with_capacity(num_indices);
                if num_indices > 0 {
                    for _ in 0..num_indices {
                        let v: f64 = (&stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?)
                            .try_into()?;
                        indices.push(v as usize);
                    }
                    indices.reverse();
                }
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Cell(ca) => {
                        // Expand in column-major order up to out_count elements
                        let mut values: Vec<Value> = Vec::new();
                        if indices.is_empty() {
                            // Expand all elements in column-major order
                            values.extend(ca.data.iter().map(|p| (*(*p)).clone()));
                        } else {
                            match indices.len() {
                                1 => {
                                    let i = indices[0];
                                    if i == 0 || i > ca.data.len() {
                                        return Err(mex(
                                            "CellIndexOutOfBounds",
                                            "Cell index out of bounds",
                                        ));
                                    }
                                    values.push((*ca.data[i - 1]).clone());
                                }
                                2 => {
                                    let r = indices[0];
                                    let c = indices[1];
                                    if r == 0 || r > ca.rows || c == 0 || c > ca.cols {
                                        return Err(mex(
                                            "CellSubscriptOutOfBounds",
                                            "Cell subscript out of bounds",
                                        ));
                                    }
                                    values.push((*ca.data[(r - 1) * ca.cols + (c - 1)]).clone());
                                }
                                _ => return Err("Unsupported number of cell indices".to_string()),
                            }
                        }
                        // Pad or truncate to out_count
                        if values.len() >= out_count {
                            for v in values.iter().take(out_count) {
                                stack.push(v.clone());
                            }
                        } else {
                            for v in &values {
                                stack.push(v.clone());
                            }
                            for _ in values.len()..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                    }
                    Value::Object(obj) => {
                        // Defer to subsref; expect a cell back; then expand one element
                        let cell = runmat_runtime::call_builtin(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        let v = match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e),
                        };
                        // Push returned value and pad to out_count
                        stack.push(v);
                        for _ in 1..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    _ => return Err("Cell expansion on non-cell".to_string()),
                }
            }
            Instr::Pop => {
                stack.pop();
            }
            Instr::Return => {
                break;
            }
            Instr::ReturnValue => {
                let return_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                stack.push(return_value);
                break;
            }
            Instr::StoreIndex(num_indices) => {
                // RHS to assign, then indices, then base
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut indices = Vec::new();
                for _ in 0..num_indices {
                    let v: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    indices.push(v as usize);
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                // TODO(GC): write barrier hook if base is in older generation and rhs/indices reference younger objects
                match base {
                    Value::Object(obj) => {
                        // subsasgn(obj, '()', {indices...}, rhs)
                        let cell = runmat_runtime::call_builtin(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsasgn".to_string()),
                                Value::String("()".to_string()),
                                cell,
                                rhs,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::Tensor(mut t) => {
                        // 1D linear or 2D scalar assignment only for now
                        if indices.len() == 1 {
                            let total = t.rows() * t.cols();
                            let idx = indices[0];
                            if idx == 0 || idx > total {
                                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            let val: f64 = (&rhs).try_into()?;
                            t.data[idx - 1] = val;
                            stack.push(Value::Tensor(t));
                        } else if indices.len() == 2 {
                            let i = indices[0];
                            let j = indices[1];
                            let rows = t.rows();
                            let cols = t.cols();
                            if i == 0 || i > rows || j == 0 || j > cols {
                                return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
                            }
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
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut indices = Vec::new();
                for _ in 0..num_indices {
                    let v: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    indices.push(v as usize);
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                // TODO(GC): write barrier hook for cell element updates
                match base {
                    Value::Object(obj) => {
                        // subsasgn(obj, '{}', {indices}, rhs)
                        let cell = runmat_builtins::CellArray::new(
                            indices.iter().map(|n| Value::Num(*n as f64)).collect(),
                            1,
                            indices.len(),
                        )
                        .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsasgn".to_string()),
                                Value::String("{}".to_string()),
                                Value::Cell(cell),
                                rhs,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::Cell(mut ca) => match indices.len() {
                        1 => {
                            let i = indices[0];
                            if i == 0 || i > ca.data.len() {
                                return Err(mex(
                                    "CellIndexOutOfBounds",
                                    "Cell index out of bounds",
                                ));
                            }
                            if let Some(oldv) = ca.data.get(i - 1) {
                                runmat_gc::gc_record_write(oldv, &rhs);
                            }
                            *ca.data[i - 1] = rhs;
                            stack.push(Value::Cell(ca));
                        }
                        2 => {
                            let i = indices[0];
                            let j = indices[1];
                            if i == 0 || i > ca.rows || j == 0 || j > ca.cols {
                                return Err(mex(
                                    "CellSubscriptOutOfBounds",
                                    "Cell subscript out of bounds",
                                ));
                            }
                            let lin = (i - 1) * ca.cols + (j - 1);
                            if let Some(oldv) = ca.data.get(lin) {
                                runmat_gc::gc_record_write(oldv, &rhs);
                            }
                            *ca.data[lin] = rhs;
                            stack.push(Value::Cell(ca));
                        }
                        _ => return Err("Unsupported number of cell indices".to_string()),
                    },
                    _ => return Err("Cell assignment on non-cell".to_string()),
                }
            }
            Instr::LoadMember(field) => {
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Object(obj) => {
                        if let Some((p, _owner)) =
                            runmat_builtins::lookup_property(&obj.class_name, &field)
                        {
                            if p.is_static {
                                vm_bail!(format!(
                                    "Property '{}' is static; use classref('{}').{}",
                                    field, obj.class_name, field
                                ));
                            }
                            if p.get_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", field))
                            }
                            if p.is_dependent {
                                // Call get.<field>(obj)
                                let getter = format!("get.{field}");
                                match runmat_runtime::call_builtin(
                                    &getter,
                                    &[Value::Object(obj.clone())],
                                ) {
                                    Ok(v) => {
                                        stack.push(v);
                                        continue;
                                    }
                                    Err(_e) => {}
                                }
                            }
                        }
                        if let Some(v) = obj.properties.get(&field) {
                            stack.push(v.clone());
                        } else if let Some((p2, _)) =
                            runmat_builtins::lookup_property(&obj.class_name, &field)
                        {
                            if p2.is_dependent {
                                let backing = format!("{field}_backing");
                                if let Some(vb) = obj.properties.get(&backing) {
                                    stack.push(vb.clone());
                                    continue;
                                }
                            }
                        } else if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                            if cls.methods.contains_key("subsref") {
                                match runmat_runtime::call_builtin(
                                    "call_method",
                                    &[
                                        Value::Object(obj),
                                        Value::String("subsref".to_string()),
                                        Value::String(".".to_string()),
                                        Value::String(field),
                                    ],
                                ) {
                                    Ok(v) => stack.push(v),
                                    Err(e) => vm_bail!(e),
                                }
                            } else {
                                vm_bail!(format!(
                                    "Undefined property '{}' for class {}",
                                    field, obj.class_name
                                ));
                            }
                        } else {
                            vm_bail!(format!("Unknown class {}", obj.class_name));
                        }
                    }
                    Value::Struct(st) => {
                        if let Some(v) = st.fields.get(&field) {
                            stack.push(v.clone());
                        } else {
                            vm_bail!(format!("Undefined field '{}'", field));
                        }
                    }
                    Value::Cell(ca) => {
                        // Extract field from each struct element; build a cell with same shape
                        let mut out: Vec<Value> = Vec::with_capacity(ca.data.len());
                        for v in &ca.data {
                            match &**v {
                                Value::Struct(st) => {
                                    if let Some(fv) = st.fields.get(&field) {
                                        out.push(fv.clone());
                                    } else {
                                        out.push(Value::Num(0.0));
                                    }
                                }
                                other => {
                                    out.push(other.clone());
                                }
                            }
                        }
                        let new_cell = runmat_builtins::CellArray::new(out, ca.rows, ca.cols)
                            .map_err(|e| format!("cell field gather: {e}"))?;
                        stack.push(Value::Cell(new_cell));
                    }
                    _ => vm_bail!("LoadMember on non-object".into()),
                }
            }
            Instr::LoadMemberDynamic => {
                let name_val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let name: String = (&name_val).try_into()?;
                match base {
                    Value::Object(obj) => {
                        if let Some((p, _owner)) =
                            runmat_builtins::lookup_property(&obj.class_name, &name)
                        {
                            if p.is_static {
                                vm_bail!(format!(
                                    "Property '{}' is static; use classref('{}').{}",
                                    name, obj.class_name, name
                                ));
                            }
                            if p.get_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", name))
                            }
                        }
                        if let Some(v) = obj.properties.get(&name) {
                            stack.push(v.clone());
                        } else if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                            if cls.methods.contains_key("subsref") {
                                match runmat_runtime::call_builtin(
                                    "call_method",
                                    &[
                                        Value::Object(obj),
                                        Value::String("subsref".to_string()),
                                        Value::String(".".to_string()),
                                        Value::String(name),
                                    ],
                                ) {
                                    Ok(v) => stack.push(v),
                                    Err(e) => vm_bail!(e),
                                }
                            } else {
                                vm_bail!(format!(
                                    "Undefined property '{}' for class {}",
                                    name, obj.class_name
                                ));
                            }
                        } else {
                            vm_bail!(format!("Unknown class {}", obj.class_name));
                        }
                    }
                    Value::Struct(st) => {
                        if let Some(v) = st.fields.get(&name) {
                            stack.push(v.clone());
                        } else {
                            vm_bail!(format!("Undefined field '{}'", name));
                        }
                    }
                    _ => vm_bail!("LoadMemberDynamic on non-struct/object".into()),
                }
            }
            Instr::StoreMember(field) => {
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                // TODO(GC): write barrier hook for object/struct field write
                match base {
                    Value::Object(mut obj) => {
                        if let Some((p, _owner)) =
                            runmat_builtins::lookup_property(&obj.class_name, &field)
                        {
                            if p.is_static {
                                vm_bail!(format!(
                                    "Property '{}' is static; use classref('{}').{}",
                                    field, obj.class_name, field
                                ));
                            }
                            if p.set_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", field))
                            }
                            if p.is_dependent {
                                // Call set.<field>(obj, rhs)
                                let setter = format!("set.{field}");
                                match runmat_runtime::call_builtin(
                                    &setter,
                                    &[Value::Object(obj.clone()), rhs.clone()],
                                ) {
                                    Ok(v) => {
                                        stack.push(v);
                                        continue;
                                    }
                                    Err(_e) => {}
                                }
                            }
                            if let Some(oldv) = obj.properties.get(&field) {
                                runmat_gc::gc_record_write(oldv, &rhs);
                            }
                            obj.properties.insert(field, rhs);
                            stack.push(Value::Object(obj));
                        } else if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                            if cls.methods.contains_key("subsasgn") {
                                match runmat_runtime::call_builtin(
                                    "call_method",
                                    &[
                                        Value::Object(obj),
                                        Value::String("subsasgn".to_string()),
                                        Value::String(".".to_string()),
                                        Value::String(field),
                                        rhs,
                                    ],
                                ) {
                                    Ok(v) => stack.push(v),
                                    Err(e) => vm_bail!(e),
                                }
                            } else {
                                vm_bail!(format!(
                                    "Undefined property '{}' for class {}",
                                    field, obj.class_name
                                ));
                            }
                        } else {
                            vm_bail!(format!("Unknown class {}", obj.class_name));
                        }
                    }
                    Value::ClassRef(cls) => {
                        if let Some((p, owner)) = runmat_builtins::lookup_property(&cls, &field) {
                            if !p.is_static {
                                vm_bail!(format!("Property '{}' is not static", field));
                            }
                            if p.set_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", field))
                            }
                            runmat_builtins::set_static_property_value_in_owner(
                                &owner, &field, rhs,
                            )?;
                            stack.push(Value::ClassRef(cls));
                        } else {
                            vm_bail!(format!("Unknown property '{}' on class {}", field, cls));
                        }
                    }
                    Value::Struct(mut st) => {
                        if let Some(oldv) = st.fields.get(&field) {
                            runmat_gc::gc_record_write(oldv, &rhs);
                        }
                        st.fields.insert(field, rhs);
                        stack.push(Value::Struct(st));
                    }
                    Value::Cell(mut ca) => {
                        // Assign field across each element; support scalar rhs or cell rhs of same shape
                        let is_cell_rhs = matches!(rhs, Value::Cell(_));
                        let rhs_cell = if let Value::Cell(rc) = &rhs {
                            Some(rc)
                        } else {
                            None
                        };
                        if is_cell_rhs {
                            if let Some(rc) = rhs_cell {
                                if rc.rows != ca.rows || rc.cols != ca.cols {
                                    vm_bail!(
                                        "Field assignment: cell rhs shape mismatch".to_string()
                                    );
                                }
                            }
                        }
                        for i in 0..ca.data.len() {
                            let rv = if let Some(rc) = rhs_cell {
                                (*rc.data[i]).clone()
                            } else {
                                rhs.clone()
                            };
                            match &mut *ca.data[i] {
                                Value::Struct(st) => {
                                    if let Some(oldv) = st.fields.get(&field) {
                                        runmat_gc::gc_record_write(oldv, &rv);
                                    }
                                    st.fields.insert(field.clone(), rv);
                                }
                                other => {
                                    // If not struct, convert to struct with this single field
                                    let mut st = runmat_builtins::StructValue::new();
                                    st.fields.insert(field.clone(), rv);
                                    *other = Value::Struct(st);
                                }
                            }
                        }
                        stack.push(Value::Cell(ca));
                    }
                    _ => vm_bail!("StoreMember on non-object".into()),
                }
            }
            Instr::StoreMemberDynamic => {
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let name_val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let name: String = (&name_val).try_into()?;
                // TODO(GC): write barrier hook for dynamic field write
                match base {
                    Value::Object(mut obj) => {
                        if let Some((p, _owner)) =
                            runmat_builtins::lookup_property(&obj.class_name, &name)
                        {
                            if p.is_static {
                                vm_bail!(format!(
                                    "Property '{}' is static; use classref('{}').{}",
                                    name, obj.class_name, name
                                ));
                            }
                            if p.set_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", name))
                            }
                        }
                        if let Some(oldv) = obj.properties.get(&name) {
                            runmat_gc::gc_record_write(oldv, &rhs);
                        }
                        obj.properties.insert(name, rhs);
                        stack.push(Value::Object(obj));
                    }
                    Value::Struct(mut st) => {
                        if let Some(oldv) = st.fields.get(&name) {
                            runmat_gc::gc_record_write(oldv, &rhs);
                        }
                        st.fields.insert(name, rhs);
                        stack.push(Value::Struct(st));
                    }
                    Value::Cell(mut ca) => {
                        let is_cell_rhs = matches!(rhs, Value::Cell(_));
                        let rhs_cell = if let Value::Cell(rc) = &rhs {
                            Some(rc)
                        } else {
                            None
                        };
                        if is_cell_rhs {
                            if let Some(rc) = rhs_cell {
                                if rc.rows != ca.rows || rc.cols != ca.cols {
                                    vm_bail!(
                                        "Field assignment: cell rhs shape mismatch".to_string()
                                    );
                                }
                            }
                        }
                        for i in 0..ca.data.len() {
                            let rv = if let Some(rc) = rhs_cell {
                                (*rc.data[i]).clone()
                            } else {
                                rhs.clone()
                            };
                            match &mut *ca.data[i] {
                                Value::Struct(st) => {
                                    if let Some(oldv) = st.fields.get(&name) {
                                        runmat_gc::gc_record_write(oldv, &rv);
                                    }
                                    st.fields.insert(name.clone(), rv);
                                }
                                other => {
                                    let mut st = runmat_builtins::StructValue::new();
                                    st.fields.insert(name.clone(), rv);
                                    *other = Value::Struct(st);
                                }
                            }
                        }
                        stack.push(Value::Cell(ca));
                    }
                    _ => vm_bail!("StoreMemberDynamic on non-struct/object".into()),
                }
            }
            Instr::CallMethod(name, arg_count) => {
                // base, then args are on stack in order: [..., base, a1, a2, ...]
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Object(obj) => {
                        // Compose qualified and try runtime builtin dispatch, passing receiver first
                        if let Some((m, _owner)) =
                            runmat_builtins::lookup_method(&obj.class_name, &name)
                        {
                            if m.is_static {
                                vm_bail!(format!(
                                    "Method '{}' is static; use classref({}).{}",
                                    name, obj.class_name, name
                                ));
                            }
                            if m.access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Method '{}' is private", name))
                            }
                            let mut full_args = Vec::with_capacity(1 + args.len());
                            full_args.push(Value::Object(obj));
                            full_args.extend(args.into_iter());
                            let v = runmat_runtime::call_builtin(&m.function_name, &full_args)?;
                            stack.push(v);
                            continue;
                        }
                        let qualified = format!("{}.{}", obj.class_name, name);
                        let mut full_args = Vec::with_capacity(1 + args.len());
                        full_args.push(Value::Object(obj));
                        full_args.extend(args.into_iter());
                        if let Ok(v) = runmat_runtime::call_builtin(&qualified, &full_args) {
                            stack.push(v);
                        } else {
                            match runmat_runtime::call_builtin(&name, &full_args) {
                                Ok(v) => {
                                    stack.push(v);
                                }
                                Err(e) => {
                                    vm_bail!(e);
                                }
                            }
                        }
                    }
                    _ => vm_bail!("CallMethod on non-object".into()),
                }
            }
            Instr::LoadMethod(name) => {
                // Base object on stack; return a closure that calls the method with receiver as first captured arg
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Object(obj) => {
                        let func_qual = format!("{}.{}", obj.class_name, name);
                        stack.push(Value::Closure(runmat_builtins::Closure {
                            function_name: func_qual,
                            captures: vec![Value::Object(obj)],
                        }));
                    }
                    Value::ClassRef(cls) => {
                        // Bound static method handle (no receiver capture), resolve via inheritance
                        if let Some((m, _owner)) = runmat_builtins::lookup_method(&cls, &name) {
                            if !m.is_static {
                                vm_bail!(format!("Method '{}' is not static", name));
                            }
                            stack.push(Value::Closure(runmat_builtins::Closure {
                                function_name: m.function_name,
                                captures: vec![],
                            }));
                        } else {
                            vm_bail!(format!("Unknown static method '{}' on class {}", name, cls));
                        }
                    }
                    _ => vm_bail!("LoadMethod requires object or classref".into()),
                }
            }
            Instr::CreateClosure(func_name, capture_count) => {
                let mut captures = Vec::with_capacity(capture_count);
                for _ in 0..capture_count {
                    captures.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                captures.reverse();
                stack.push(Value::Closure(runmat_builtins::Closure {
                    function_name: func_name,
                    captures,
                }));
            }
            Instr::LoadStaticProperty(class_name, prop) => {
                // Enforce access and static-ness via registry (with inheritance)
                if let Some((p, owner)) = runmat_builtins::lookup_property(&class_name, &prop) {
                    if !p.is_static {
                        vm_bail!(format!("Property '{}' is not static", prop));
                    }
                    if p.get_access == runmat_builtins::Access::Private {
                        vm_bail!(format!("Property '{}' is private", prop))
                    }
                    if let Some(v) = runmat_builtins::get_static_property_value(&owner, &prop) {
                        stack.push(v);
                    } else if let Some(v) = &p.default_value {
                        stack.push(v.clone());
                    } else {
                        stack.push(Value::Num(0.0));
                    }
                } else {
                    vm_bail!(format!(
                        "Unknown property '{}' on class {}",
                        prop, class_name
                    ));
                }
            }
            Instr::CallStaticMethod(class_name, method, arg_count) => {
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                if let Some((m, _owner)) = runmat_builtins::lookup_method(&class_name, &method) {
                    if !m.is_static {
                        vm_bail!(format!("Method '{}' is not static", method));
                    }
                    if m.access == runmat_builtins::Access::Private {
                        vm_bail!(format!("Method '{}' is private", method))
                    }
                    let v = match runmat_runtime::call_builtin(&m.function_name, &args) {
                        Ok(v) => v,
                        Err(e) => vm_bail!(e),
                    };
                    stack.push(v);
                } else {
                    vm_bail!(format!(
                        "Unknown static method '{}' on class {}",
                        method, class_name
                    ));
                }
            }
            Instr::RegisterClass {
                name,
                super_class,
                properties,
                methods,
            } => {
                // Build a minimal ClassDef and register it in runtime builtins registry
                let mut prop_map = std::collections::HashMap::new();
                for (p, is_static, get_access, set_access) in properties {
                    let gacc = if get_access.eq_ignore_ascii_case("private") {
                        runmat_builtins::Access::Private
                    } else {
                        runmat_builtins::Access::Public
                    };
                    let sacc = if set_access.eq_ignore_ascii_case("private") {
                        runmat_builtins::Access::Private
                    } else {
                        runmat_builtins::Access::Public
                    };
                    let (is_dep, clean_name) = if let Some(stripped) = p.strip_prefix("@dep:") {
                        (true, stripped.to_string())
                    } else {
                        (false, p.clone())
                    };
                    prop_map.insert(
                        clean_name.clone(),
                        runmat_builtins::PropertyDef {
                            name: clean_name,
                            is_static,
                            is_dependent: is_dep,
                            get_access: gacc,
                            set_access: sacc,
                            default_value: None,
                        },
                    );
                }
                let mut method_map = std::collections::HashMap::new();
                for (mname, fname, is_static, access) in methods {
                    let access = if access.eq_ignore_ascii_case("private") {
                        runmat_builtins::Access::Private
                    } else {
                        runmat_builtins::Access::Public
                    };
                    method_map.insert(
                        mname.clone(),
                        runmat_builtins::MethodDef {
                            name: mname,
                            is_static,
                            access,
                            function_name: fname,
                        },
                    );
                }
                let def = runmat_builtins::ClassDef {
                    name: name.clone(),
                    parent: super_class.clone(),
                    properties: prop_map,
                    methods: method_map,
                };
                runmat_builtins::register_class(def);
            }
            Instr::CallFeval(arg_count) => {
                // Stack layout: [..., f, a1, a2, ...]
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                let func_val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match func_val {
                    Value::Closure(c) => {
                        // First try runtime builtin dispatch with captures prepended
                        let mut full_args = c.captures.clone();
                        full_args.extend(args.into_iter());
                        match runmat_runtime::call_builtin(&c.function_name, &full_args) {
                            Ok(v) => stack.push(v),
                            Err(_e) => {
                                // If not a builtin, try user-defined function
                                if let Some(func) = context.functions.get(&c.function_name).cloned()
                                {
                                    let argc = full_args.len();
                                    if !func.has_varargin {
                                        if argc != func.params.len() {
                                            vm_bail!(format!("Function '{}' expects {} arguments, got {} - Not enough input arguments", c.function_name, func.params.len(), argc));
                                        }
                                    } else if argc + 1 < func.params.len() {
                                        vm_bail!(format!(
                                            "Function '{}' expects at least {} arguments, got {}",
                                            c.function_name,
                                            func.params.len() - 1,
                                            argc
                                        ));
                                    }
                                    let var_map =
                                        runmat_hir::remapping::create_complete_function_var_map(
                                            &func.params,
                                            &func.outputs,
                                            &func.body,
                                        );
                                    let local_var_count = var_map.len();
                                    let remapped_body = runmat_hir::remapping::remap_function_body(
                                        &func.body, &var_map,
                                    );
                                    let func_vars_count = local_var_count.max(func.params.len());
                                    let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                                    for (i, _param_id) in func.params.iter().enumerate() {
                                        if i < full_args.len() && i < func_vars.len() {
                                            func_vars[i] = full_args[i].clone();
                                        }
                                    }
                                    // Copy referenced globals into local frame
                                    for (original_var_id, local_var_id) in &var_map {
                                        let local_index = local_var_id.0;
                                        let global_index = original_var_id.0;
                                        if local_index < func_vars.len()
                                            && global_index < vars.len()
                                        {
                                            let is_parameter = func
                                                .params
                                                .iter()
                                                .any(|param_id| param_id == original_var_id);
                                            if !is_parameter {
                                                func_vars[local_index] = vars[global_index].clone();
                                            }
                                        }
                                    }
                                    let mut func_var_types = func.var_types.clone();
                                    if func_var_types.len() < local_var_count {
                                        func_var_types.resize(local_var_count, Type::Unknown);
                                    }
                                    let func_program = runmat_hir::HirProgram {
                                        body: remapped_body,
                                        var_types: func_var_types,
                                    };
                                    let func_bytecode = crate::compile_with_functions(
                                        &func_program,
                                        &context.functions,
                                    )?;
                                    // Merge any newly synthesized nested functions into current context
                                    for (k, v) in func_bytecode.functions.iter() {
                                        context.functions.insert(k.clone(), v.clone());
                                    }
                                    let func_result_vars =
                                        match interpret_function(&func_bytecode, func_vars) {
                                            Ok(v) => v,
                                            Err(e) => vm_bail!(e),
                                        };
                                    if let Some(output_var_id) = func.outputs.first() {
                                        let local_output_index =
                                            var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);
                                        if local_output_index < func_result_vars.len() {
                                            stack
                                                .push(func_result_vars[local_output_index].clone());
                                        } else {
                                            stack.push(Value::Num(0.0));
                                        }
                                    } else {
                                        stack.push(Value::Num(0.0));
                                    }
                                } else {
                                    vm_bail!(format!("unknown builtin `{}'", c.function_name));
                                }
                            }
                        }
                    }
                    Value::String(s) => {
                        if let Some(name) = s.strip_prefix('@') {
                            match runmat_runtime::call_builtin(name, &args) {
                                Ok(v) => {
                                    stack.push(v);
                                }
                                Err(_e) => {
                                    // Fallback: user-defined function named `name`
                                    if let Some(func) = context.functions.get(name).cloned() {
                                        let argc = args.len();
                                        if !func.has_varargin {
                                            if argc != func.params.len() {
                                                vm_bail!(format!("Function '{}' expects {} arguments, got {} - Not enough input arguments", name, func.params.len(), argc));
                                            }
                                        } else if argc + 1 < func.params.len() {
                                            vm_bail!(format!("Function '{name}' expects at least {} arguments, got {argc}", func.params.len()-1));
                                        }
                                        let var_map =
                                            runmat_hir::remapping::create_complete_function_var_map(
                                                &func.params,
                                                &func.outputs,
                                                &func.body,
                                            );
                                        let local_var_count = var_map.len();
                                        let remapped_body =
                                            runmat_hir::remapping::remap_function_body(
                                                &func.body, &var_map,
                                            );
                                        let func_vars_count =
                                            local_var_count.max(func.params.len());
                                        let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                                        if func.has_varargout {
                                            let fixed = func.params.len().saturating_sub(1);
                                            for i in 0..fixed {
                                                if i < args.len() && i < func_vars.len() {
                                                    func_vars[i] = args[i].clone();
                                                }
                                            }
                                            let mut rest: Vec<Value> = if args.len() > fixed {
                                                args[fixed..].to_vec()
                                            } else {
                                                Vec::new()
                                            };
                                            let cell = runmat_builtins::CellArray::new(
                                                std::mem::take(&mut rest),
                                                1,
                                                if args.len() > fixed {
                                                    args.len() - fixed
                                                } else {
                                                    0
                                                },
                                            )
                                            .map_err(|e| format!("varargin: {e}"))?;
                                            if fixed < func_vars.len() {
                                                func_vars[fixed] = Value::Cell(cell);
                                            }
                                        } else {
                                            for (i, _param_id) in func.params.iter().enumerate() {
                                                if i < args.len() && i < func_vars.len() {
                                                    func_vars[i] = args[i].clone();
                                                }
                                            }
                                        }
                                        // Copy referenced globals into local frame
                                        for (original_var_id, local_var_id) in &var_map {
                                            let local_index = local_var_id.0;
                                            let global_index = original_var_id.0;
                                            if local_index < func_vars.len()
                                                && global_index < vars.len()
                                            {
                                                let is_parameter = func
                                                    .params
                                                    .iter()
                                                    .any(|param_id| param_id == original_var_id);
                                                if !is_parameter {
                                                    func_vars[local_index] =
                                                        vars[global_index].clone();
                                                }
                                            }
                                        }
                                        // Initialize varargout cell if needed
                                        if func.has_varargout {
                                            if let Some(varargout_oid) = func.outputs.last() {
                                                if let Some(local_id) = var_map.get(varargout_oid) {
                                                    if local_id.0 < func_vars.len() {
                                                        let empty =
                                                            runmat_builtins::CellArray::new(
                                                                vec![],
                                                                1,
                                                                0,
                                                            )
                                                            .map_err(|e| {
                                                                format!("varargout init: {e}")
                                                            })?;
                                                        func_vars[local_id.0] = Value::Cell(empty);
                                                    }
                                                }
                                            }
                                        }
                                        let mut func_var_types = func.var_types.clone();
                                        if func_var_types.len() < local_var_count {
                                            func_var_types.resize(local_var_count, Type::Unknown);
                                        }
                                        let func_program = runmat_hir::HirProgram {
                                            body: remapped_body,
                                            var_types: func_var_types,
                                        };
                                        let func_bytecode = crate::compile_with_functions(
                                            &func_program,
                                            &context.functions,
                                        )?;
                                        let func_result_vars =
                                            match interpret_function(&func_bytecode, func_vars) {
                                                Ok(v) => v,
                                                Err(e) => vm_bail!(e),
                                            };
                                        if func.has_varargout {
                                            // Return first varargout element if present
                                            let first = func
                                                .outputs
                                                .first()
                                                .and_then(|oid| var_map.get(oid))
                                                .map(|lid| lid.0)
                                                .unwrap_or(0);
                                            if let Some(Value::Cell(ca)) =
                                                func_result_vars.get(first)
                                            {
                                                if !ca.data.is_empty() {
                                                    stack.push((*ca.data[0]).clone());
                                                } else {
                                                    stack.push(Value::Num(0.0));
                                                }
                                            } else if let Some(v) = func_result_vars.get(first) {
                                                stack.push(v.clone());
                                            } else {
                                                stack.push(Value::Num(0.0));
                                            }
                                        } else if let Some(output_var_id) = func.outputs.first() {
                                            let local_output_index = var_map
                                                .get(output_var_id)
                                                .map(|id| id.0)
                                                .unwrap_or(0);
                                            if local_output_index < func_result_vars.len() {
                                                stack.push(
                                                    func_result_vars[local_output_index].clone(),
                                                );
                                            } else {
                                                stack.push(Value::Num(0.0));
                                            }
                                        } else {
                                            stack.push(Value::Num(0.0));
                                        }
                                    } else {
                                        vm_bail!(format!("unknown builtin `{name}`"));
                                    }
                                }
                            }
                        } else {
                            vm_bail!(format!(
                                "feval: expected function handle string starting with '@', got {s}"
                            ));
                        }
                    }
                    Value::FunctionHandle(name) => {
                        match runmat_runtime::call_builtin(&name, &args) {
                            Ok(v) => {
                                stack.push(v);
                            }
                            Err(e) => {
                                vm_bail!(e);
                            }
                        }
                    }
                    Value::CharArray(ca) => {
                        let s: String = ca.data.iter().collect();
                        if let Some(name) = s.strip_prefix('@') {
                            match runmat_runtime::call_builtin(name, &args) {
                                Ok(v) => {
                                    stack.push(v);
                                }
                                Err(_e) => {
                                    if let Some(func) = context.functions.get(name).cloned() {
                                        let argc = args.len();
                                        if !func.has_varargin {
                                            if argc != func.params.len() {
                                                vm_bail!(format!("Function '{}' expects {} arguments, got {} - Not enough input arguments", name, func.params.len(), argc));
                                            }
                                        } else if argc + 1 < func.params.len() {
                                            vm_bail!(format!("Function '{name}' expects at least {} arguments, got {argc}", func.params.len()-1));
                                        }
                                        let var_map =
                                            runmat_hir::remapping::create_complete_function_var_map(
                                                &func.params,
                                                &func.outputs,
                                                &func.body,
                                            );
                                        let local_var_count = var_map.len();
                                        let remapped_body =
                                            runmat_hir::remapping::remap_function_body(
                                                &func.body, &var_map,
                                            );
                                        let mut func_var_types = func.var_types.clone();
                                        if func_var_types.len() < local_var_count {
                                            func_var_types.resize(local_var_count, Type::Unknown);
                                        }
                                        let func_program = runmat_hir::HirProgram {
                                            body: remapped_body,
                                            var_types: func_var_types,
                                        };
                                        let func_bytecode = crate::compile_with_functions(
                                            &func_program,
                                            &context.functions,
                                        )?;
                                        for (k, v) in func_bytecode.functions.iter() {
                                            context.functions.insert(k.clone(), v.clone());
                                        }
                                        let mut func_vars =
                                            vec![
                                                Value::Num(0.0);
                                                var_map.len().max(func.params.len())
                                            ];
                                        for (i, _param_id) in func.params.iter().enumerate() {
                                            if i < args.len() && i < func_vars.len() {
                                                func_vars[i] = args[i].clone();
                                            }
                                        }
                                        let func_result_vars =
                                            match interpret_function(&func_bytecode, func_vars) {
                                                Ok(v) => v,
                                                Err(e) => vm_bail!(e),
                                            };
                                        if let Some(output_var_id) = func.outputs.first() {
                                            let local_output_index = var_map
                                                .get(output_var_id)
                                                .map(|id| id.0)
                                                .unwrap_or(0);
                                            if local_output_index < func_result_vars.len() {
                                                stack.push(
                                                    func_result_vars[local_output_index].clone(),
                                                );
                                            } else {
                                                stack.push(Value::Num(0.0));
                                            }
                                        } else {
                                            stack.push(Value::Num(0.0));
                                        }
                                    } else {
                                        vm_bail!(format!("unknown builtin `{}`", name));
                                    }
                                }
                            }
                        } else {
                            // Treat as plain function name
                            match runmat_runtime::call_builtin(&s, &args) {
                                Ok(v) => stack.push(v),
                                Err(e) => vm_bail!(e),
                            }
                        }
                    }
                    other => vm_bail!(format!("feval: unsupported function value {other:?}")),
                }
            }
            Instr::CallFevalExpandMulti(specs) => {
                // Stack layout: [..., f, args in spec order]; we build final args vector by walking specs reversed
                // First, collect per-spec values into temp, then call feval with those args
                let mut temp: Vec<Value> = Vec::new();
                for spec in specs.iter().rev() {
                    if spec.is_expand {
                        // Pop indices if any, then base
                        let mut indices: Vec<Value> = Vec::with_capacity(spec.num_indices);
                        for _ in 0..spec.num_indices {
                            indices.push(
                                stack
                                    .pop()
                                    .ok_or(mex("StackUnderflow", "stack underflow"))?,
                            );
                        }
                        indices.reverse();
                        let base = stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?;
                        let expanded: Vec<Value> = if spec.expand_all {
                            match base {
                                Value::Cell(ca) => ca
                                    .data
                                    .iter()
                                    .map(|p| (*(*p)).clone())
                                    .collect::<Vec<Value>>(),
                                Value::Object(obj) => {
                                    let empty = runmat_builtins::CellArray::new(vec![], 1, 0)
                                        .map_err(|e| format!("subsref build error: {e}"))?;
                                    let v = match runmat_runtime::call_builtin(
                                        "call_method",
                                        &[
                                            Value::Object(obj),
                                            Value::String("subsref".to_string()),
                                            Value::String("{}".to_string()),
                                            Value::Cell(empty),
                                        ],
                                    ) {
                                        Ok(v) => v,
                                        Err(e) => vm_bail!(e),
                                    };
                                    match v {
                                        Value::Cell(ca) => ca
                                            .data
                                            .iter()
                                            .map(|p| (*(*p)).clone())
                                            .collect::<Vec<Value>>(),
                                        other => vec![other],
                                    }
                                }
                                _ => return Err(
                                    "CallFevalExpandMulti requires cell or object for expand_all"
                                        .to_string(),
                                ),
                            }
                        } else {
                            match (base, indices.len()) {
                                (Value::Cell(ca), 1) => match &indices[0] {
                                    Value::Num(n) => {
                                        let idx = *n as usize;
                                        if idx == 0 || idx > ca.data.len() {
                                            return Err(mex(
                                                "CellIndexOutOfBounds",
                                                "Cell index out of bounds",
                                            ));
                                        }
                                        vec![(*ca.data[idx - 1]).clone()]
                                    }
                                    Value::Int(i) => {
                                        let idx = i.to_i64() as usize;
                                        if idx == 0 || idx > ca.data.len() {
                                            return Err(mex(
                                                "CellIndexOutOfBounds",
                                                "Cell index out of bounds",
                                            ));
                                        }
                                        vec![(*ca.data[idx - 1]).clone()]
                                    }
                                    Value::Tensor(t) => {
                                        let mut out: Vec<Value> = Vec::with_capacity(t.data.len());
                                        for &val in &t.data {
                                            let iu = val as usize;
                                            if iu == 0 || iu > ca.data.len() {
                                                return Err(mex(
                                                    "CellIndexOutOfBounds",
                                                    "Cell index out of bounds",
                                                ));
                                            }
                                            out.push((*ca.data[iu - 1]).clone());
                                        }
                                        out
                                    }
                                    _ => {
                                        return Err(mex(
                                            "CellIndexType",
                                            "Unsupported cell index type",
                                        ))
                                    }
                                },
                                (Value::Cell(ca), 2) => {
                                    let r: f64 = (&indices[0]).try_into()?;
                                    let c: f64 = (&indices[1]).try_into()?;
                                    let (ir, ic) = (r as usize, c as usize);
                                    if ir == 0 || ir > ca.rows || ic == 0 || ic > ca.cols {
                                        return Err(mex(
                                            "CellSubscriptOutOfBounds",
                                            "Cell subscript out of bounds",
                                        ));
                                    }
                                    vec![(*ca.data[(ir - 1) * ca.cols + (ic - 1)]).clone()]
                                }
                                (Value::Object(obj), _) => {
                                    let cell = runmat_builtins::CellArray::new(
                                        indices.clone(),
                                        1,
                                        indices.len(),
                                    )
                                    .map_err(|e| format!("subsref build error: {e}"))?;
                                    let v = match runmat_runtime::call_builtin(
                                        "call_method",
                                        &[
                                            Value::Object(obj),
                                            Value::String("subsref".to_string()),
                                            Value::String("{}".to_string()),
                                            Value::Cell(cell),
                                        ],
                                    ) {
                                        Ok(v) => v,
                                        Err(e) => vm_bail!(e),
                                    };
                                    vec![v]
                                }
                                _ => {
                                    return Err(
                                        "CallFevalExpandMulti requires cell or object cell access"
                                            .to_string(),
                                    )
                                }
                            }
                        };
                        for v in expanded {
                            temp.push(v);
                        }
                    } else {
                        temp.push(
                            stack
                                .pop()
                                .ok_or(mex("StackUnderflow", "stack underflow"))?,
                        );
                    }
                }
                temp.reverse();
                // Now pop function value and call feval runtime with assembled args
                let func_val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match func_val {
                    Value::Closure(c) => {
                        let mut full_args = c.captures.clone();
                        full_args.extend(temp.into_iter());
                        match runmat_runtime::call_builtin(&c.function_name, &full_args) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::String(s) => {
                        if let Some(name) = s.strip_prefix('@') {
                            match runmat_runtime::call_builtin(name, &temp) {
                                Ok(v) => stack.push(v),
                                Err(_e) => {
                                    // Fallback to user-defined function named `name`
                                    if let Some(func) = context.functions.get(name).cloned() {
                                        let argc = temp.len();
                                        if !func.has_varargin {
                                            if argc != func.params.len() {
                                                vm_bail!(format!("Function '{}' expects {} arguments, got {} - Not enough input arguments", name, func.params.len(), argc));
                                            }
                                        } else if argc + 1 < func.params.len() {
                                            vm_bail!(format!("Function '{name}' expects at least {} arguments, got {argc}", func.params.len()-1));
                                        }
                                        let var_map =
                                            runmat_hir::remapping::create_complete_function_var_map(
                                                &func.params,
                                                &func.outputs,
                                                &func.body,
                                            );
                                        let local_var_count = var_map.len();
                                        let remapped_body =
                                            runmat_hir::remapping::remap_function_body(
                                                &func.body, &var_map,
                                            );
                                        let func_vars_count =
                                            local_var_count.max(func.params.len());
                                        let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                                        if func.has_varargin {
                                            let fixed = func.params.len().saturating_sub(1);
                                            for i in 0..fixed {
                                                if i < temp.len() && i < func_vars.len() {
                                                    func_vars[i] = temp[i].clone();
                                                }
                                            }
                                            let mut rest: Vec<Value> = if temp.len() > fixed {
                                                temp[fixed..].to_vec()
                                            } else {
                                                Vec::new()
                                            };
                                            let cell = runmat_builtins::CellArray::new(
                                                std::mem::take(&mut rest),
                                                1,
                                                if temp.len() > fixed {
                                                    temp.len() - fixed
                                                } else {
                                                    0
                                                },
                                            )
                                            .map_err(|e| format!("varargin: {e}"))?;
                                            if fixed < func_vars.len() {
                                                func_vars[fixed] = Value::Cell(cell);
                                            }
                                        } else {
                                            for (i, _param_id) in func.params.iter().enumerate() {
                                                if i < temp.len() && i < func_vars.len() {
                                                    func_vars[i] = temp[i].clone();
                                                }
                                            }
                                        }
                                        // Copy referenced globals into local frame
                                        for (original_var_id, local_var_id) in &var_map {
                                            let local_index = local_var_id.0;
                                            let global_index = original_var_id.0;
                                            if local_index < func_vars.len()
                                                && global_index < vars.len()
                                            {
                                                let is_parameter = func
                                                    .params
                                                    .iter()
                                                    .any(|param_id| param_id == original_var_id);
                                                if !is_parameter {
                                                    func_vars[local_index] =
                                                        vars[global_index].clone();
                                                }
                                            }
                                        }
                                        // Initialize varargout cell if needed
                                        if func.has_varargout {
                                            if let Some(varargout_oid) = func.outputs.last() {
                                                if let Some(local_id) = var_map.get(varargout_oid) {
                                                    if local_id.0 < func_vars.len() {
                                                        let empty =
                                                            runmat_builtins::CellArray::new(
                                                                vec![],
                                                                1,
                                                                0,
                                                            )
                                                            .map_err(|e| {
                                                                format!("varargout init: {e}")
                                                            })?;
                                                        func_vars[local_id.0] = Value::Cell(empty);
                                                    }
                                                }
                                            }
                                        }
                                        let mut func_var_types = func.var_types.clone();
                                        if func_var_types.len() < local_var_count {
                                            func_var_types.resize(local_var_count, Type::Unknown);
                                        }
                                        let func_program = runmat_hir::HirProgram {
                                            body: remapped_body,
                                            var_types: func_var_types,
                                        };
                                        let func_bytecode = crate::compile_with_functions(
                                            &func_program,
                                            &context.functions,
                                        )?;
                                        let func_result_vars =
                                            match interpret_function(&func_bytecode, func_vars) {
                                                Ok(v) => v,
                                                Err(e) => vm_bail!(e),
                                            };
                                        if func.has_varargout {
                                            let first = func
                                                .outputs
                                                .first()
                                                .and_then(|oid| var_map.get(oid))
                                                .map(|lid| lid.0)
                                                .unwrap_or(0);
                                            if let Some(Value::Cell(ca)) =
                                                func_result_vars.get(first)
                                            {
                                                if !ca.data.is_empty() {
                                                    stack.push((*ca.data[0]).clone());
                                                } else {
                                                    stack.push(Value::Num(0.0));
                                                }
                                            } else if let Some(v) = func_result_vars.get(first) {
                                                stack.push(v.clone());
                                            } else {
                                                stack.push(Value::Num(0.0));
                                            }
                                        } else if let Some(output_var_id) = func.outputs.first() {
                                            let local_output_index = var_map
                                                .get(output_var_id)
                                                .map(|id| id.0)
                                                .unwrap_or(0);
                                            if local_output_index < func_result_vars.len() {
                                                stack.push(
                                                    func_result_vars[local_output_index].clone(),
                                                );
                                            } else {
                                                stack.push(Value::Num(0.0));
                                            }
                                        } else {
                                            stack.push(Value::Num(0.0));
                                        }
                                    } else {
                                        vm_bail!(format!("unknown builtin `{name}`"));
                                    }
                                }
                            }
                        } else {
                            vm_bail!(format!(
                                "feval: expected function handle string starting with '@', got {s}"
                            ));
                        }
                    }
                    Value::FunctionHandle(name) => {
                        match runmat_runtime::call_builtin(&name, &temp) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    other => vm_bail!(format!("feval: unsupported function value {other:?}")),
                }
            }
            Instr::AndAnd(target) => {
                // Stack top holds lhs != 0 result (1 or 0). If false (0), jump to target and push 0
                let cond: f64 = (&stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                if cond == 0.0 {
                    pc = target;
                    continue;
                } else { /* leave evaluation of rhs result already pushed by compiler path */
                }
            }
            Instr::OrOr(target) => {
                // Stack top holds lhs != 0 result (1 or 0). If true (non-zero), jump to target and push 1
                let cond: f64 = (&stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                if cond != 0.0 {
                    pc = target;
                    continue;
                } else { /* evaluate rhs result path */
                }
            }
            Instr::Swap => {
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                stack.push(a);
                stack.push(b);
            }
        }
        pc += 1;
    }
    for (i, var) in vars.iter().enumerate() {
        if i < initial_vars.len() {
            initial_vars[i] = var.clone();
        }
    }
    Ok(vars)
}

#[cfg(feature = "native-accel")]
fn try_execute_fusion_group(
    plan: &runmat_accelerate::FusionGroupPlan,
    graph: &runmat_accelerate::AccelGraph,
    stack: &mut Vec<Value>,
    vars: &mut [Value],
    context: &ExecutionContext,
) -> Result<Value, String> {
    let mut inputs: Vec<Option<Value>> = vec![None; plan.inputs.len()];

    for (idx, value) in &plan.constants {
        if let Some(slot) = inputs.get_mut(*idx) {
            if slot.is_none() {
                *slot = Some(value.clone());
            }
        }
    }

    let describe_value = |value: &Value| -> &'static str {
        match value {
            Value::Int(_) => "Int",
            Value::Num(_) => "Num",
            Value::Complex(_, _) => "Complex",
            Value::Bool(_) => "Bool",
            Value::LogicalArray(_) => "LogicalArray",
            Value::String(_) => "String",
            Value::StringArray(_) => "StringArray",
            Value::CharArray(_) => "CharArray",
            Value::Tensor(_) => "Tensor",
            Value::ComplexTensor(_) => "ComplexTensor",
            Value::Cell(_) => "Cell",
            Value::Struct(_) => "Struct",
            Value::GpuTensor(_) => "GpuTensor",
            Value::Object(_) => "Object",
            Value::HandleObject(_) => "HandleObject",
            Value::Listener(_) => "Listener",
            Value::FunctionHandle(_) => "FunctionHandle",
            Value::Closure(_) => "Closure",
            Value::ClassRef(_) => "ClassRef",
            Value::MException(_) => "MException",
        }
    };

    for (idx, value_id) in plan.inputs.iter().enumerate() {
        let info = graph
            .value(*value_id)
            .ok_or_else(|| format!("fusion: missing value metadata for id {value_id}"))?;
        match &info.origin {
            ValueOrigin::Variable { kind, index } => {
                let value =
                    match kind {
                        VarKind::Global => vars
                            .get(*index)
                            .cloned()
                            .ok_or_else(|| format!("fusion: global var {index} out of range"))?,
                        VarKind::Local => {
                            if let Some(frame) = context.call_stack.last() {
                                let absolute = frame.locals_start + index;
                                context.locals.get(absolute).cloned().ok_or_else(|| {
                                    format!("fusion: local var {index} unavailable")
                                })?
                            } else {
                                vars.get(*index).cloned().ok_or_else(|| {
                                    format!("fusion: local var {index} unavailable")
                                })?
                            }
                        }
                    };
                debug_assert!(
                    inputs[idx].is_none(),
                    "fusion: duplicate input slot {} for plan {}",
                    idx,
                    plan.index
                );
                inputs[idx] = Some(value);
            }
            ValueOrigin::Constant | ValueOrigin::NodeOutput { .. } | ValueOrigin::Unknown => {}
        }
    }

    if log::log_enabled!(log::Level::Debug) {
        let stack_needed_preview = plan.stack_pattern.len();
        let stack_snapshot: Vec<&Value> = stack
            .iter()
            .rev()
            .take(stack_needed_preview)
            .map(|v| v)
            .collect();
        let stack_kinds: Vec<&'static str> = stack_snapshot
            .iter()
            .rev()
            .map(|v| describe_value(v))
            .collect();
        let input_meta: Vec<String> = plan
            .inputs
            .iter()
            .enumerate()
            .map(|(i, value_id)| {
                if let Some(info) = graph.value(*value_id) {
                    format!("#{i}:id={} origin={:?}", value_id, info.origin)
                } else {
                    format!("#{i}:id={} origin=<missing>", value_id)
                }
            })
            .collect();
        log::debug!(
            "fusion group {} gather: stack_depth={} stack_needed={} stack_kinds={:?} pattern={:?} inputs={:?}",
            plan.index,
            stack.len(),
            stack_needed_preview,
            stack_kinds,
            &plan.stack_pattern,
            input_meta
        );
    }

    let stack_needed = plan.stack_pattern.len();
    if stack.len() < stack_needed {
        return Err("fusion: stack underflow gathering inputs".to_string());
    }

    let slice_start = stack.len() - stack_needed;
    let slice = stack[slice_start..].to_vec();
    stack.truncate(slice_start);

    debug_assert_eq!(slice.len(), stack_needed);

    let mut consumed: Vec<Value> = Vec::new();
    for (offset, input_idx) in plan.stack_pattern.iter().enumerate() {
        let val = slice
            .get(offset)
            .cloned()
            .ok_or_else(|| "fusion: unable to read stack segment".to_string())?;
        consumed.push(val.clone());
        if inputs[*input_idx].is_none() {
            inputs[*input_idx] = Some(val);
        }
    }

    let inputs: Vec<Value> = inputs
        .into_iter()
        .map(|opt| opt.ok_or_else(|| "fusion: missing input value".to_string()))
        .collect::<Result<_, _>>()?;

    let request = FusionExecutionRequest { plan, inputs };
    if plan.group.kind.is_elementwise() {
        match execute_elementwise(request) {
            Ok(result) => Ok(result),
            Err(err) => {
                for value in consumed.into_iter().rev() {
                    stack.push(value);
                }
                Err(err.to_string())
            }
        }
    } else if plan.group.kind.is_reduction() {
        // Derive (reduce_len, num_slices) primarily from the first input's actual shape.
        // Default MATLAB reduction is along dimension 1 (rows) producing 1 x ncols.
        let (reduce_len, num_slices) = {
            // The plan inputs vector aligns with consumed values by stack_pattern
            // and variables/constants. We look at the first input value we gathered.
            let first_input = request.inputs.get(0);
            if let Some(val) = first_input {
                match val {
                    Value::GpuTensor(h) => {
                        if h.shape.len() >= 2 {
                            (h.shape[0].max(1), h.shape[1].max(1))
                        } else if h.shape.len() == 1 {
                            (h.shape[0].max(1), 1)
                        } else {
                            (1, 1)
                        }
                    }
                    Value::Tensor(t) => {
                        if t.shape.len() >= 2 {
                            (t.shape[0].max(1), t.shape[1].max(1))
                        } else if t.shape.len() == 1 {
                            (t.shape[0].max(1), 1)
                        } else {
                            (1, 1)
                        }
                    }
                    _ => (1, 1),
                }
            } else {
                (1, 1)
            }
        };
        let workgroup_size = 256u32;
        match execute_reduction(request, reduce_len, num_slices, workgroup_size) {
            Ok(result) => Ok(result),
            Err(err) => {
                for value in consumed.into_iter().rev() {
                    stack.push(value);
                }
                Err(err.to_string())
            }
        }
    } else {
        Ok(result) => Ok(result),
        Err(err) => {
            for value in consumed.into_iter().rev() {
                stack.push(value);
            }
            Err(err.to_string())
        }
    }
}

#[cfg(feature = "native-accel")]
fn clear_residency(value: &Value) {
    if let Value::GpuTensor(handle) = value {
        fusion_residency::clear(handle);
    }
}

fn parse_exception(err: &str) -> runmat_builtins::MException {
    // Prefer the last occurrence of ": " to split IDENT: message, preserving nested identifiers
    if let Some(idx) = err.rfind(": ") {
        let (id, msg) = err.split_at(idx);
        let message = msg.trim_start_matches(':').trim().to_string();
        let ident = if id.trim().is_empty() {
            format!("{ERROR_NAMESPACE}:error")
        } else {
            id.trim().to_string()
        };
        return runmat_builtins::MException::new(ident, message);
    }
    // Fallback: if any ':' present, use the last as separator
    if let Some(idx) = err.rfind(':') {
        let (id, msg) = err.split_at(idx);
        let message = msg.trim_start_matches(':').trim().to_string();
        let ident = if id.trim().is_empty() {
            format!("{ERROR_NAMESPACE}:error")
        } else {
            id.trim().to_string()
        };
        runmat_builtins::MException::new(ident, message)
    } else {
        runmat_builtins::MException::new(format!("{ERROR_NAMESPACE}:error"), err.to_string())
    }
}

/// Interpret bytecode with default variable initialization
pub fn interpret(bytecode: &Bytecode) -> Result<Vec<Value>, String> {
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    interpret_with_vars(bytecode, &mut vars, Some("<main>"))
}

pub fn interpret_function(bytecode: &Bytecode, vars: Vec<Value>) -> Result<Vec<Value>, String> {
    // Delegate to the counted variant with anonymous name and zero counts
    interpret_function_with_counts(bytecode, vars, "<anonymous>", 0, 0)
}

fn interpret_function_with_counts(
    bytecode: &Bytecode,
    mut vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
) -> Result<Vec<Value>, String> {
    // Push (nargin, nargout), run, then pop
    let res = CALL_COUNTS.with(|cc| {
        cc.borrow_mut().push((in_count, out_count));
        let r = interpret_with_vars(bytecode, &mut vars, Some(name));
        cc.borrow_mut().pop();
        r
    });
    // Persist any variables declared persistent in this bytecode under the given function name
    let func_name = name.to_string();
    for instr in &bytecode.instructions {
        match instr {
            crate::instr::Instr::DeclarePersistent(indices) => {
                for &i in indices {
                    if i < vars.len() {
                        let key = (func_name.clone(), i);
                        PERSISTENTS.with(|p| {
                            p.borrow_mut().insert(key, vars[i].clone());
                        });
                    }
                }
            }
            crate::instr::Instr::DeclarePersistentNamed(indices, names) => {
                for (pos, &i) in indices.iter().enumerate() {
                    if i < vars.len() {
                        let key = (func_name.clone(), i);
                        let name_key = (
                            func_name.clone(),
                            names
                                .get(pos)
                                .cloned()
                                .unwrap_or_else(|| format!("var_{i}")),
                        );
                        let val = vars[i].clone();
                        PERSISTENTS.with(|p| {
                            p.borrow_mut().insert(key, val.clone());
                        });
                        PERSISTENTS_BY_NAME.with(|p| {
                            p.borrow_mut().insert(name_key, val);
                        });
                    }
                }
            }
            _ => {}
        }
    }
    res
}
