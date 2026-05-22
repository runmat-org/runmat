use crate::call::shared::{
    call_object_member_subsasgn, call_object_member_subsref,
    call_object_property_getter_with_outputs, call_object_property_setter_with_outputs,
    class_defines_member_subsasgn, class_defines_member_subsref, external_qualified_display_name,
};
use crate::interpreter::errors::mex;
use runmat_builtins::{self, Access, Closure, StructValue, Value};
use runmat_runtime::RuntimeError;

const IDENT_PROPERTY_PRIVATE_ACCESS: &str = "RunMat:PropertyPrivateAccess";

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
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
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
            bound_function: None,
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
                bound_function: None,
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
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
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
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
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
        Value::Cell(ca) => crate::ops::cells::assign_cell_member(ca, field, rhs, on_write),
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
        || text.contains("invalid axes handle")
}

fn is_possible_graphics_handle_value(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v > 0.0,
        Value::Int(i) => i.to_f64() > 0.0,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{load_member, load_static_member, store_member};
    use runmat_builtins::{
        get_static_property_value, register_class, Access, ClassDef, MethodDef, ObjectInstance,
        PropertyDef, Value,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_class_name(prefix: &str) -> String {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", prefix, id)
    }

    #[test]
    fn load_static_member_resolves_inherited_static_property_value() {
        let parent_name = unique_class_name("vm_static_parent");
        let child_name = unique_class_name("vm_static_child");

        let mut parent_properties = HashMap::new();
        parent_properties.insert(
            "version".to_string(),
            PropertyDef {
                name: "version".to_string(),
                is_static: true,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: Some(Value::Num(1.0)),
            },
        );
        register_class(ClassDef {
            name: parent_name.clone(),
            parent: None,
            properties: parent_properties,
            methods: HashMap::new(),
        });
        register_class(ClassDef {
            name: child_name.clone(),
            parent: Some(parent_name.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        runmat_builtins::set_static_property_value(&parent_name, "version", Value::Num(3.0));
        let value = load_static_member(&child_name, "version")
            .expect("inherited static property should resolve through parent metadata owner");
        assert_eq!(value, Value::Num(3.0));
    }

    #[test]
    fn store_member_updates_inherited_static_property_owner_slot() {
        let parent_name = unique_class_name("vm_store_static_parent");
        let child_name = unique_class_name("vm_store_static_child");

        let mut parent_properties = HashMap::new();
        parent_properties.insert(
            "version".to_string(),
            PropertyDef {
                name: "version".to_string(),
                is_static: true,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: Some(Value::Num(1.0)),
            },
        );
        register_class(ClassDef {
            name: parent_name.clone(),
            parent: None,
            properties: parent_properties,
            methods: HashMap::new(),
        });
        register_class(ClassDef {
            name: child_name.clone(),
            parent: Some(parent_name.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let out = futures::executor::block_on(store_member(
            Value::ClassRef(child_name.clone()),
            "version".to_string(),
            Value::Num(9.0),
            false,
            |_old, _new| {},
        ))
        .expect("storing inherited static property via child class ref should succeed");
        assert_eq!(out, Value::ClassRef(child_name));
        assert_eq!(
            get_static_property_value(&parent_name, "version"),
            Some(Value::Num(9.0))
        );
    }

    #[test]
    fn load_static_member_resolves_inherited_static_method() {
        let parent_name = unique_class_name("vm_method_parent");
        let child_name = unique_class_name("vm_method_child");

        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            "build".to_string(),
            MethodDef {
                name: "build".to_string(),
                is_static: true,
                access: Access::Public,
                function_name: "build_impl".to_string(),
                implicit_class_argument: None,
            },
        );
        register_class(ClassDef {
            name: parent_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        register_class(ClassDef {
            name: child_name.clone(),
            parent: Some(parent_name.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let value = load_static_member(&child_name, "build")
            .expect("inherited static method should resolve through parent metadata");
        let Value::Closure(closure) = value else {
            panic!("expected static method lookup to return closure");
        };
        assert_eq!(closure.function_name, "build_impl");
    }

    #[test]
    fn load_member_uses_inherited_subsref_for_missing_property() {
        let parent_name = unique_class_name("vm_subsref_parent");
        let child_name = unique_class_name("vm_subsref_child");

        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            "subsref".to_string(),
            MethodDef {
                name: "subsref".to_string(),
                is_static: false,
                access: Access::Public,
                function_name: "OverIdx.subsref".to_string(),
                implicit_class_argument: None,
            },
        );
        register_class(ClassDef {
            name: parent_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        register_class(ClassDef {
            name: child_name.clone(),
            parent: Some(parent_name),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let obj = Value::Object(ObjectInstance::new(child_name));
        let value = futures::executor::block_on(load_member(obj, "missing".to_string(), false))
            .expect("missing member should dispatch to inherited subsref");
        assert_eq!(value, Value::Num(77.0));
    }

    #[test]
    fn store_member_uses_inherited_subsasgn_for_missing_property() {
        let parent_name = unique_class_name("vm_subsasgn_parent");
        let child_name = unique_class_name("vm_subsasgn_child");

        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            "subsasgn".to_string(),
            MethodDef {
                name: "subsasgn".to_string(),
                is_static: false,
                access: Access::Public,
                function_name: "OverIdx.subsasgn".to_string(),
                implicit_class_argument: None,
            },
        );
        register_class(ClassDef {
            name: parent_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        register_class(ClassDef {
            name: child_name.clone(),
            parent: Some(parent_name),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let out = futures::executor::block_on(store_member(
            Value::Object(ObjectInstance::new(child_name)),
            "missing".to_string(),
            Value::Num(13.0),
            false,
            |_old, _new| {},
        ))
        .expect("missing member store should dispatch to inherited subsasgn");
        let Value::Object(obj) = out else {
            panic!("expected object result from inherited subsasgn");
        };
        assert_eq!(obj.properties.get("missing"), Some(&Value::Num(13.0)));
    }
}
