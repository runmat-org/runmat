use crate::interpreter::errors::mex;
use crate::interpreter::stack::{pop_args, pop_value};
use runmat_builtins::{
    builtin_functions, get_static_property_value, lookup_method, lookup_property, Access,
    CellArray, Closure, Value,
};
use runmat_runtime::RuntimeError;

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
        captures,
    }));
    Ok(())
}

pub fn load_method_closure(base: Value, name: String) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => Ok(Value::Closure(Closure {
            function_name: format!("{}.{}", obj.class_name, name),
            captures: vec![Value::Object(obj)],
        })),
        Value::ClassRef(cls) => {
            if let Some((m, _owner)) = lookup_method(&cls, &name) {
                if !m.is_static {
                    return Err(format!("Method '{}' is not static", name).into());
                }
                return Ok(Value::Closure(Closure {
                    function_name: m.function_name,
                    captures: vec![],
                }));
            }
            let qualified = format!("{cls}.{name}");
            if builtin_functions().iter().any(|b| b.name == qualified) {
                Ok(Value::Closure(Closure {
                    function_name: qualified,
                    captures: vec![],
                }))
            } else {
                Err(format!("Unknown static method '{}' on class {}", name, cls).into())
            }
        }
        _ => Err(mex("LoadMethod", "LoadMethod requires object or classref")),
    }
}

pub async fn call_method(base: Value, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
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
                return runmat_runtime::call_builtin_async(&m.function_name, &full_args).await;
            }
            let qualified = format!("{}.{}", obj.class_name, name);
            let mut full_args = Vec::with_capacity(1 + args.len());
            full_args.push(Value::Object(obj));
            full_args.extend(args.clone());
            if let Ok(v) = runmat_runtime::call_builtin_async(&qualified, &full_args).await {
                Ok(v)
            } else {
                runmat_runtime::call_builtin_async(name, &full_args).await
            }
        }
        _ => Err(mex("CallMethod", "CallMethod on non-object")),
    }
}

pub async fn call_static_method(
    class_name: &str,
    method: &str,
    args: Vec<Value>,
) -> Result<Value, RuntimeError> {
    if let Some((m, _owner)) = lookup_method(class_name, method) {
        if !m.is_static {
            return Err(format!("Method '{}' is not static", method).into());
        }
        if m.access == Access::Private {
            return Err(format!("Method '{}' is private", method).into());
        }
        return runmat_runtime::call_builtin_async(&m.function_name, &args).await;
    }
    let qualified = format!("{}.{}", class_name, method);
    runmat_runtime::call_builtin_async(&qualified, &args).await
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

pub async fn call_method_or_member_index(
    base: Value,
    name: String,
    args: Vec<Value>,
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
                return runmat_runtime::call_builtin_async(&m.function_name, &full_args).await;
            }

            let mut method_args = Vec::with_capacity(1 + args.len());
            method_args.push(Value::Object(obj.clone()));
            method_args.extend(args.iter().cloned());
            let qualified = format!("{}.{}", obj.class_name, name);
            if let Ok(v) = runmat_runtime::call_builtin_async(&qualified, &method_args).await {
                return Ok(v);
            }
            if let Ok(v) = runmat_runtime::call_builtin_async(&name, &method_args).await {
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
            runmat_runtime::call_builtin_async("getfield", &getfield_args).await
        }
        Value::HandleObject(handle) => {
            let mut method_args = Vec::with_capacity(2 + args.len());
            method_args.push(Value::HandleObject(handle.clone()));
            method_args.push(Value::String(name.clone()));
            method_args.extend(args.iter().cloned());
            if let Ok(v) = runmat_runtime::call_builtin_async("call_method", &method_args).await {
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
            runmat_runtime::call_builtin_async("getfield", &getfield_args).await
        }
        Value::ClassRef(cls) => {
            if let Some((m, _owner)) = lookup_method(&cls, &name) {
                if !m.is_static {
                    return Err(format!("Method '{}' is not static", name).into());
                }
                return runmat_runtime::call_builtin_async(&m.function_name, &args).await;
            }

            let qualified = format!("{cls}.{name}");
            if let Ok(v) = runmat_runtime::call_builtin_async(&qualified, &args).await {
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
            runmat_runtime::call_builtin_async("getfield", &getfield_args).await
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
