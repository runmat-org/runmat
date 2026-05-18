use crate::call::shared::{
    call_object_member_subsasgn, call_object_member_subsref,
    call_object_property_getter_with_outputs, call_object_property_setter_with_outputs,
    class_defines_member_subsasgn, class_defines_member_subsref, external_qualified_display_name,
};
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
                    if let Ok(v) = call_object_property_getter_with_outputs(
                        Value::Object(obj.clone()),
                        &field,
                        1,
                    )
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
                if class_defines_member_subsref(&cls) {
                    call_object_member_subsref(Value::Object(obj), field).await
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
            call_object_member_subsref(Value::HandleObject(handle), field).await
        }
        Value::ClassRef(cls) => load_static_member(&cls, &field),
        Value::Struct(st) => {
            if let Some(v) = st.fields.get(&field) {
                Ok(v.clone())
            } else if allow_init {
                Ok(Value::Struct(StructValue::new()))
            } else {
                Err(format!("Undefined field '{}'", field).into())
            }
        }
        Value::Cell(ca) => crate::ops::cells::gather_cell_member(&ca, &field),
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
            semantic_function: None,
            captures: vec![],
        }))
    } else {
        let qualified = external_qualified_display_name(cls, field);
        if runmat_builtins::builtin_functions()
            .iter()
            .any(|b| b.name == qualified)
        {
            Ok(Value::Closure(Closure {
                function_name: qualified,
                semantic_function: None,
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
                    if let Ok(v) = call_object_property_setter_with_outputs(
                        Value::Object(obj.clone()),
                        &field,
                        rhs.clone(),
                        1,
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
                if class_defines_member_subsasgn(&cls) {
                    call_object_member_subsasgn(Value::Object(obj), field, rhs).await
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
            call_object_member_subsasgn(Value::HandleObject(handle), field, rhs).await
        }
        Value::Struct(mut st) => {
            if let Some(oldv) = st.fields.get(&field) {
                on_write(oldv, &rhs);
            }
            st.fields.insert(field, rhs);
            Ok(Value::Struct(st))
        }
        Value::Cell(ca) => crate::ops::cells::assign_cell_member(ca, field, rhs, on_write),
        Value::Num(0.0) if allow_init => {
            let mut st = StructValue::new();
            st.fields.insert(field, rhs);
            Ok(Value::Struct(st))
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
