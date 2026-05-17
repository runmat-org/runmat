use crate::interpreter::errors::mex;
use crate::interpreter::stack::{pop_args, pop_value};
use runmat_builtins::{
    builtin_functions, get_static_property_value, lookup_method, lookup_property, Access,
    CellArray, Closure, Value,
};
use runmat_runtime::RuntimeError;

async fn call_runtime_builtin(
    name: &str,
    args: &[Value],
    requested_outputs: Option<usize>,
) -> Result<Value, RuntimeError> {
    match requested_outputs {
        Some(count) => runmat_runtime::call_builtin_async_with_outputs(name, args, count).await,
        None => runmat_runtime::call_builtin_async(name, args).await,
    }
}

pub fn create_closure(
    stack: &mut Vec<Value>,
    func_name: String,
    capture_count: usize,
) -> Result<(), RuntimeError> {
    let mut captures = Vec::with_capacity(capture_count);
    for _ in 0..capture_count {
        captures.push(pop_value(stack)?);
    }
    captures.reverse();
    stack.push(Value::Closure(Closure {
        function_name: func_name,
        semantic_function: None,
        captures,
    }));
    Ok(())
}

pub fn create_semantic_closure(
    stack: &mut Vec<Value>,
    function: runmat_hir::FunctionId,
    display_name: String,
    capture_count: usize,
) -> Result<(), RuntimeError> {
    let mut captures = Vec::with_capacity(capture_count);
    for _ in 0..capture_count {
        captures.push(pop_value(stack)?);
    }
    captures.reverse();
    stack.push(Value::Closure(Closure {
        function_name: display_name,
        semantic_function: Some(function.0),
        captures,
    }));
    Ok(())
}

pub fn load_method_closure(base: Value, name: String) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => Ok(Value::Closure(Closure {
            function_name: format!("{}.{}", obj.class_name, name),
            semantic_function: None,
            captures: vec![Value::Object(obj)],
        })),
        Value::ClassRef(cls) => {
            if let Some((m, _owner)) = lookup_method(&cls, &name) {
                if !m.is_static {
                    return Err(format!("Method '{}' is not static", name).into());
                }
                return Ok(Value::Closure(Closure {
                    function_name: m.function_name,
                    semantic_function: None,
                    captures: vec![],
                }));
            }
            let qualified = format!("{cls}.{name}");
            if builtin_functions().iter().any(|b| b.name == qualified) {
                Ok(Value::Closure(Closure {
                    function_name: qualified,
                    semantic_function: None,
                    captures: vec![],
                }))
            } else {
                Err(format!("Unknown static method '{}' on class {}", name, cls).into())
            }
        }
        _ => Err(mex("LoadMethod", "LoadMethod requires object or classref")),
    }
}

pub async fn call_method_with_outputs(
    base: Value,
    name: &str,
    args: Vec<Value>,
    requested_outputs: Option<usize>,
) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => {
            if let Some((m, _owner)) = lookup_method(&obj.class_name, name) {
                if m.is_static {
                    return Err(format!(
                        "Method '{}' is static; use classref({}).{}",
                        name, obj.class_name, name
                    )
                    .into());
                }
                if m.access == Access::Private {
                    return Err(format!("Method '{}' is private", name).into());
                }
                let mut full_args = Vec::with_capacity(1 + args.len());
                full_args.push(Value::Object(obj));
                full_args.extend(args);
                return call_runtime_builtin(&m.function_name, &full_args, requested_outputs).await;
            }
            let qualified = format!("{}.{}", obj.class_name, name);
            let mut full_args = Vec::with_capacity(1 + args.len());
            full_args.push(Value::Object(obj));
            full_args.extend(args.clone());
            if let Ok(v) = call_runtime_builtin(&qualified, &full_args, requested_outputs).await {
                Ok(v)
            } else {
                call_runtime_builtin(name, &full_args, requested_outputs).await
            }
        }
        _ => Err(mex("CallMethod", "CallMethod on non-object")),
    }
}

pub async fn call_static_method_with_outputs(
    class_name: &str,
    method: &str,
    args: Vec<Value>,
    requested_outputs: Option<usize>,
) -> Result<Value, RuntimeError> {
    if let Some((m, _owner)) = lookup_method(class_name, method) {
        if !m.is_static {
            return Err(format!("Method '{}' is not static", method).into());
        }
        if m.access == Access::Private {
            return Err(format!("Method '{}' is private", method).into());
        }
        return call_runtime_builtin(&m.function_name, &args, requested_outputs).await;
    }
    let qualified = format!("{}.{}", class_name, method);
    call_runtime_builtin(&qualified, &args, requested_outputs).await
}

pub fn load_static_property(class_name: &str, prop: &str) -> Result<Value, RuntimeError> {
    if let Some((p, owner)) = lookup_property(class_name, prop) {
        if !p.is_static {
            return Err(format!("Property '{}' is not static", prop).into());
        }
        if p.get_access == Access::Private {
            return Err(format!("Property '{}' is private", prop).into());
        }
        if let Some(v) = get_static_property_value(&owner, prop) {
            Ok(v)
        } else if let Some(v) = &p.default_value {
            Ok(v.clone())
        } else {
            Ok(Value::Num(0.0))
        }
    } else {
        Err(format!("Unknown property '{}' on class {}", prop, class_name).into())
    }
}

pub async fn call_method_or_member_index_with_outputs(
    base: Value,
    name: String,
    args: Vec<Value>,
    requested_outputs: Option<usize>,
) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => {
            if let Some((m, _owner)) = lookup_method(&obj.class_name, &name) {
                if m.is_static {
                    return Err(format!(
                        "Method '{}' is static; use classref({}).{}",
                        name, obj.class_name, name
                    )
                    .into());
                }
                if m.access == Access::Private {
                    return Err(format!("Method '{}' is private", name).into());
                }
                let mut full_args = Vec::with_capacity(1 + args.len());
                full_args.push(Value::Object(obj));
                full_args.extend(args);
                return call_runtime_builtin(&m.function_name, &full_args, requested_outputs).await;
            }

            let mut method_args = Vec::with_capacity(1 + args.len());
            method_args.push(Value::Object(obj.clone()));
            method_args.extend(args.iter().cloned());
            let qualified = format!("{}.{}", obj.class_name, name);
            if let Ok(v) = call_runtime_builtin(&qualified, &method_args, requested_outputs).await {
                return Ok(v);
            }
            if let Ok(v) = call_runtime_builtin(&name, &method_args, requested_outputs).await {
                return Ok(v);
            }

            let mut getfield_args = Vec::with_capacity(3);
            getfield_args.push(Value::Object(obj));
            getfield_args.push(Value::String(name));
            if !args.is_empty() {
                let idx_count = args.len();
                let idx_cell = CellArray::new(args, 1, idx_count)
                    .map_err(|e| format!("getfield idx build: {e}"))?;
                getfield_args.push(Value::Cell(idx_cell));
            }
            call_runtime_builtin("getfield", &getfield_args, requested_outputs).await
        }
        Value::HandleObject(handle) => {
            if let Ok(v) = crate::call::shared::call_object_named_method_with_outputs(
                Value::HandleObject(handle.clone()),
                name.clone(),
                args.clone(),
                requested_outputs,
            )
            .await
            {
                return Ok(v);
            }

            let mut getfield_args = Vec::with_capacity(3);
            getfield_args.push(Value::HandleObject(handle));
            getfield_args.push(Value::String(name));
            if !args.is_empty() {
                let idx_count = args.len();
                let idx_cell = CellArray::new(args, 1, idx_count)
                    .map_err(|e| format!("getfield idx build: {e}"))?;
                getfield_args.push(Value::Cell(idx_cell));
            }
            call_runtime_builtin("getfield", &getfield_args, requested_outputs).await
        }
        Value::ClassRef(cls) => {
            if let Some((m, _owner)) = lookup_method(&cls, &name) {
                if !m.is_static {
                    return Err(format!("Method '{}' is not static", name).into());
                }
                return call_runtime_builtin(&m.function_name, &args, requested_outputs).await;
            }

            let qualified = format!("{cls}.{name}");
            if let Ok(v) = call_runtime_builtin(&qualified, &args, requested_outputs).await {
                return Ok(v);
            }

            if args.is_empty() {
                return load_static_property(&cls, &name);
            }

            Err(format!("Unknown static member '{}' on class {}", name, cls).into())
        }
        other => {
            let mut getfield_args = Vec::with_capacity(3);
            getfield_args.push(other);
            getfield_args.push(Value::String(name));
            if !args.is_empty() {
                let idx_count = args.len();
                let idx_cell = CellArray::new(args, 1, idx_count)
                    .map_err(|e| format!("getfield idx build: {e}"))?;
                getfield_args.push(Value::Cell(idx_cell));
            }
            call_runtime_builtin("getfield", &getfield_args, requested_outputs).await
        }
    }
}

pub fn collect_method_args(
    stack: &mut Vec<Value>,
    arg_count: usize,
) -> Result<(Value, Vec<Value>), RuntimeError> {
    let args = pop_args(stack, arg_count)?;
    let base = pop_value(stack)?;
    Ok((base, args))
}
