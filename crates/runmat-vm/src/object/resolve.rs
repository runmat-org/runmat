use crate::interpreter::errors::mex;
use runmat_builtins::{self, Access, Closure, StructValue, Value};
use runmat_runtime::RuntimeError;

pub async fn load_member(
    base: Value,
    field: String,
    allow_init: bool,
) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => {
            if let Some((p, _owner)) = runmat_builtins::lookup_property(&obj.class_name, &field) {
                if p.is_static {
                    return Err(format!(
                        "Property '{}' is static; use classref('{}').{}",
                        field, obj.class_name, field
                    )
                    .into());
                }
                if p.get_access == Access::Private {
                    return Err(format!("Property '{}' is private", field).into());
                }
                if p.is_dependent {
                    let getter = format!("get.{field}");
                    if let Ok(v) =
                        runmat_runtime::call_builtin_async(&getter, &[Value::Object(obj.clone())])
                            .await
                    {
                        return Ok(v);
                    }
                }
            }
            if let Some(v) = obj.properties.get(&field) {
                Ok(v.clone())
            } else if let Some((p2, _)) = runmat_builtins::lookup_property(&obj.class_name, &field)
            {
                if p2.is_dependent {
                    let backing = format!("{field}_backing");
                    if let Some(vb) = obj.properties.get(&backing) {
                        return Ok(vb.clone());
                    }
                }
                Err(format!(
                    "Undefined property '{}' for class {}",
                    field, obj.class_name
                )
                .into())
            } else if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                if cls.methods.contains_key("subsref") {
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsref".to_string()),
                        Value::String(".".to_string()),
                        Value::String(field),
                    ];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                } else {
                    Err(format!(
                        "Undefined property '{}' for class {}",
                        field, obj.class_name
                    )
                    .into())
                }
            } else {
                Err(format!("Unknown class {}", obj.class_name).into())
            }
        }
        Value::HandleObject(handle) => {
            let args = vec![
                Value::HandleObject(handle),
                Value::String("subsref".to_string()),
                Value::String(".".to_string()),
                Value::String(field),
            ];
            runmat_runtime::call_builtin_async("call_method", &args).await
        }
        Value::ClassRef(cls) => load_static_member(&cls, &field),
        base @ (Value::Num(_) | Value::Int(_)) => {
            if !is_possible_graphics_handle_value(&base) {
                return Err(mex("LoadMember", "LoadMember on non-object"));
            }
            load_graphics_member(base, &field).await.map_err(|err| {
                if is_invalid_graphics_handle_error(&err) {
                    mex("LoadMember", "LoadMember on non-object")
                } else {
                    err
                }
            })
        }
        Value::Struct(st) => {
            if let Some(v) = st.fields.get(&field) {
                Ok(v.clone())
            } else if allow_init {
                Ok(Value::Struct(StructValue::new()))
            } else {
                Err(format!("Undefined field '{}'", field).into())
            }
        }
        Value::Cell(ca) => {
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
                    other => out.push(other.clone()),
                }
            }
            let new_cell = runmat_builtins::CellArray::new(out, ca.rows, ca.cols)
                .map_err(|e| format!("cell field gather: {e}"))?;
            Ok(Value::Cell(new_cell))
        }
        Value::MException(mexn) => {
            let value = match field.as_str() {
                "identifier" => Value::String(mexn.identifier.clone()),
                "message" => Value::String(mexn.message.clone()),
                "stack" => {
                    let values: Vec<Value> = mexn
                        .stack
                        .iter()
                        .map(|s| Value::String(s.clone()))
                        .collect();
                    let rows = values.len();
                    let cell = runmat_builtins::CellArray::new(values, rows, 1)
                        .map_err(|e| format!("MException.stack: {e}"))?;
                    Value::Cell(cell)
                }
                other => return Err(format!("Reference to non-existent field '{}'.", other).into()),
            };
            Ok(value)
        }
        _ => Err(mex("LoadMember", "LoadMember on non-object")),
    }
}

pub async fn load_member_dynamic(
    base: Value,
    name: String,
    allow_init: bool,
) -> Result<Value, RuntimeError> {
    load_member(base, name, allow_init).await
}

pub fn load_static_member(cls: &str, field: &str) -> Result<Value, RuntimeError> {
    if let Some((p, owner)) = runmat_builtins::lookup_property(cls, field) {
        if !p.is_static {
            return Err(format!("Property '{}' is not static", field).into());
        }
        if p.get_access == Access::Private {
            return Err(format!("Property '{}' is private", field).into());
        }
        if let Some(v) = runmat_builtins::get_static_property_value(&owner, field) {
            Ok(v)
        } else if let Some(v) = &p.default_value {
            Ok(v.clone())
        } else {
            Ok(Value::Num(0.0))
        }
    } else if let Some((m, _owner)) = runmat_builtins::lookup_method(cls, field) {
        if !m.is_static {
            return Err(format!("Method '{}' is not static", field).into());
        }
        Ok(Value::Closure(Closure {
            function_name: m.function_name,
            captures: vec![],
        }))
    } else {
        let qualified = format!("{cls}.{field}");
        if runmat_builtins::builtin_functions()
            .iter()
            .any(|b| b.name == qualified)
        {
            Ok(Value::Closure(Closure {
                function_name: qualified,
                captures: vec![],
            }))
        } else {
            Err(format!("Unknown property '{}' on class {}", field, cls).into())
        }
    }
}

pub async fn store_member<OnWrite>(
    base: Value,
    field: String,
    rhs: Value,
    allow_init: bool,
    mut on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    match base {
        Value::Object(mut obj) => {
            if let Some((p, _owner)) = runmat_builtins::lookup_property(&obj.class_name, &field) {
                if p.is_static {
                    return Err(format!(
                        "Property '{}' is static; use classref('{}').{}",
                        field, obj.class_name, field
                    )
                    .into());
                }
                if p.set_access == Access::Private {
                    return Err(format!("Property '{}' is private", field).into());
                }
                if p.is_dependent {
                    let setter = format!("set.{field}");
                    if let Ok(v) = runmat_runtime::call_builtin_async(
                        &setter,
                        &[Value::Object(obj.clone()), rhs.clone()],
                    )
                    .await
                    {
                        return Ok(v);
                    }
                }
                if let Some(oldv) = obj.properties.get(&field) {
                    on_write(oldv, &rhs);
                }
                obj.properties.insert(field, rhs);
                Ok(Value::Object(obj))
            } else if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                if cls.methods.contains_key("subsasgn") {
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsasgn".to_string()),
                        Value::String(".".to_string()),
                        Value::String(field),
                        rhs,
                    ];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                } else {
                    Err(format!("Undefined property '{}' for class {}", field, cls.name).into())
                }
            } else {
                Err(format!("Unknown class {}", obj.class_name).into())
            }
        }
        Value::ClassRef(cls) => {
            if let Some((p, owner)) = runmat_builtins::lookup_property(&cls, &field) {
                if !p.is_static {
                    return Err(format!("Property '{}' is not static", field).into());
                }
                if p.set_access == Access::Private {
                    return Err(format!("Property '{}' is private", field).into());
                }
                runmat_builtins::set_static_property_value_in_owner(&owner, &field, rhs)?;
                Ok(Value::ClassRef(cls))
            } else {
                Err(format!("Unknown property '{}' on class {}", field, cls).into())
            }
        }
        Value::HandleObject(handle) => {
            let args = vec![
                Value::HandleObject(handle),
                Value::String("subsasgn".to_string()),
                Value::String(".".to_string()),
                Value::String(field),
                rhs,
            ];
            runmat_runtime::call_builtin_async("call_method", &args).await
        }
        Value::Num(0.0) if allow_init => {
            let mut st = StructValue::new();
            st.fields.insert(field, rhs);
            Ok(Value::Struct(st))
        }
        base @ (Value::Num(_) | Value::Int(_)) => {
            if !is_possible_graphics_handle_value(&base) {
                return Err(mex("StoreMember", "StoreMember on non-object"));
            }
            store_graphics_member(base, &field, rhs)
                .await
                .map_err(|err| {
                    if is_invalid_graphics_handle_error(&err) {
                        mex("StoreMember", "StoreMember on non-object")
                    } else {
                        err
                    }
                })
        }
        Value::Struct(mut st) => {
            if let Some(oldv) = st.fields.get(&field) {
                on_write(oldv, &rhs);
            }
            st.fields.insert(field, rhs);
            Ok(Value::Struct(st))
        }
        Value::Cell(mut ca) => {
            let rhs_cell = if let Value::Cell(rc) = &rhs {
                Some(rc)
            } else {
                None
            };
            if let Some(rc) = rhs_cell {
                if rc.rows != ca.rows || rc.cols != ca.cols {
                    return Err("Field assignment: cell rhs shape mismatch"
                        .to_string()
                        .into());
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
                            on_write(oldv, &rv);
                        }
                        st.fields.insert(field.clone(), rv);
                    }
                    other => {
                        let mut st = StructValue::new();
                        st.fields.insert(field.clone(), rv);
                        *other = Value::Struct(st);
                    }
                }
            }
            Ok(Value::Cell(ca))
        }
        _ => Err(mex("StoreMember", "StoreMember on non-object")),
    }
}

pub async fn store_member_dynamic<OnWrite>(
    base: Value,
    name: String,
    rhs: Value,
    allow_init: bool,
    on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    store_member(base, name, rhs, allow_init, on_write).await
}

async fn load_graphics_member(base: Value, field: &str) -> Result<Value, RuntimeError> {
    runmat_runtime::call_builtin_async("get", &[base, Value::String(field.to_string())]).await
}

async fn store_graphics_member(
    base: Value,
    field: &str,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    runmat_runtime::call_builtin_async(
        "set",
        &[base.clone(), Value::String(field.to_string()), rhs],
    )
    .await?;
    Ok(base)
}

fn is_invalid_graphics_handle_error(err: &RuntimeError) -> bool {
    let text = err.to_string().to_ascii_lowercase();
    text.contains("unsupported or invalid plotting handle")
        || text.contains("invalid plotting handle")
        || text.contains("invalid figure handle")
}

fn is_possible_graphics_handle_value(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v > 0.0,
        Value::Int(i) => i.to_f64() > 0.0,
        _ => false,
    }
}
