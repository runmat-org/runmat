use crate::call::shared::{
    call_object_member_subsasgn, call_object_member_subsref,
    call_object_property_getter_with_outputs, call_object_property_setter_with_outputs,
    class_defines_member_subsasgn, class_defines_member_subsref, external_qualified_display_name,
    ObjectIndexOp,
};
use crate::interpreter::errors::mex;
use runmat_builtins::{self, Closure, StructValue, Tensor, Value};
use runmat_runtime::RuntimeError;

const IDENT_PROPERTY_PRIVATE_ACCESS: &str = "RunMat:PropertyPrivateAccess";
const IDENT_PROPERTY_READ_ONLY: &str = "RunMat:PropertyReadOnly";

fn caller_has_internal_class_access(
    caller_function_name: Option<&str>,
    class_name: &str,
) -> bool {
    if let Some(caller_name) = caller_function_name {
        if let Some((caller_class, _)) = caller_name.rsplit_once('.') {
            if !caller_class.is_empty()
                && runmat_builtins::get_class(caller_class).is_some()
                && (runmat_builtins::is_class_or_subclass(caller_class, class_name)
                    || runmat_builtins::is_class_or_subclass(class_name, caller_class))
            {
                return true;
            }
        }
    }
    caller_class_for_function(caller_function_name).is_some_and(|caller_class| {
        runmat_builtins::is_class_or_subclass(&caller_class, class_name)
            || runmat_builtins::is_class_or_subclass(class_name, &caller_class)
    })
}

fn caller_is_index_overload(
    caller_function_name: Option<&str>,
    class_name: &str,
    op: ObjectIndexOp,
) -> bool {
    let Some(caller) = caller_function_name else {
        return false;
    };
    let method_name = op.protocol_name();
    if caller == method_name {
        return true;
    }
    if let Some((method, owner)) = runmat_builtins::lookup_method(class_name, method_name) {
        if caller == method.function_name {
            return true;
        }
        if caller == format!("{owner}.{method_name}") {
            return true;
        }
    }
    if let Some((caller_class, caller_method)) = caller.rsplit_once('.') {
        if caller_method == method_name
            && runmat_builtins::is_class_or_subclass(class_name, caller_class)
        {
            return true;
        }
    }
    if let Some(caller_class) = caller_class_for_function(Some(caller)) {
        if let Some((method, _owner)) = runmat_builtins::lookup_method(&caller_class, method_name) {
            if method.function_name == caller
                && (runmat_builtins::is_class_or_subclass(class_name, &caller_class)
                    || runmat_builtins::is_class_or_subclass(&caller_class, class_name))
            {
                return true;
            }
        }
    }
    false
}

fn caller_class_for_function(caller_function_name: Option<&str>) -> Option<String> {
    let caller_function_name = caller_function_name?;
    if runmat_builtins::get_class(caller_function_name).is_some() {
        return Some(caller_function_name.to_string());
    }
    if let Some(owner) = runmat_builtins::class_names().into_iter().find(|class_name| {
        runmat_builtins::get_class(class_name).is_some_and(|class_def| {
            class_def
                .methods
                .values()
                .any(|method| method.function_name == caller_function_name)
        })
    }) {
        return Some(owner);
    }
    if let Some((class_name, method_name)) = caller_function_name.rsplit_once('.') {
        if !class_name.is_empty()
            && !method_name.is_empty()
            && runmat_builtins::get_class(class_name).is_some()
        {
            return Some(class_name.to_string());
        }
    }
    None
}

fn access_permitted(
    owner: &str,
    access: &runmat_builtins::Access,
    caller_function_name: Option<&str>,
) -> bool {
    match access {
        runmat_builtins::Access::Public => true,
        runmat_builtins::Access::Private => {
            caller_class_for_function(caller_function_name).as_deref() == Some(owner)
        }
        runmat_builtins::Access::Protected => caller_class_for_function(caller_function_name)
            .is_some_and(|caller_class| runmat_builtins::is_class_or_subclass(&caller_class, owner)),
    }
}

pub async fn load_member(
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(obj) => {
            if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                if class_defines_member_subsref(&cls)
                    && !caller_is_index_overload(
                        caller_function_name,
                        &obj.class_name,
                        ObjectIndexOp::Subsref,
                    )
                    && !caller_has_internal_class_access(caller_function_name, &obj.class_name)
                {
                    return call_object_member_subsref(Value::Object(obj), field).await;
                }
            }
            if let Some((p, owner)) = runmat_builtins::lookup_property(&obj.class_name, &field) {
                if p.is_static {
                    return Err(mex(
                        "RunMat:PropertyStaticAccess",
                        &format!(
                            "Property '{}' is static; use classref('{}').{}",
                            field, obj.class_name, field
                        ),
                    ));
                }
                if !access_permitted(&owner, &p.get_access, caller_function_name)
                {
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
            } else if let Some((p2, owner)) =
                runmat_builtins::lookup_property(&obj.class_name, &field)
            {
                if !access_permitted(&owner, &p2.get_access, caller_function_name)
                {
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
                }
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
            if let Some(cls) = runmat_builtins::get_class(&handle.class_name) {
                if class_defines_member_subsref(&cls)
                    && !caller_is_index_overload(
                        caller_function_name,
                        &handle.class_name,
                        ObjectIndexOp::Subsref,
                    )
                    && !caller_has_internal_class_access(caller_function_name, &handle.class_name)
                {
                    return call_object_member_subsref(Value::HandleObject(handle), field).await;
                }
            }
            runmat_runtime::call_builtin_async_with_outputs(
                "getfield",
                &[Value::HandleObject(handle), Value::String(field)],
                1,
            )
            .await
        }
        Value::Listener(listener) => {
            runmat_runtime::call_builtin_async_with_outputs(
                "getfield",
                &[Value::Listener(listener), Value::String(field)],
                1,
            )
            .await
        }
        Value::ClassRef(cls) => load_static_member(&cls, &field, caller_function_name),
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
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    load_member(base, name, allow_init, caller_function_name).await
}

pub fn load_static_member(
    cls: &str,
    field: &str,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    if let Some((p, owner)) = runmat_builtins::lookup_property(cls, field) {
        if !p.is_static {
            return Err(mex(
                "RunMat:PropertyStaticAccess",
                &format!("Property '{}' is not static", field),
            ));
        }
        if !access_permitted(&owner, &p.get_access, caller_function_name)
        {
            return Err(mex(
                IDENT_PROPERTY_PRIVATE_ACCESS,
                &format!("Property '{}' is private", field),
            ));
        }
        if let Some(v) = runmat_builtins::get_static_property_value(&owner, field) {
            Ok(v)
        } else if let Some(v) = &p.default_value {
            Ok(v.clone())
        } else {
            Ok(Value::Tensor(Tensor::new(vec![], vec![0, 0]).expect("empty tensor")))
        }
    } else if let Some((m, _owner)) = runmat_builtins::lookup_method(cls, field) {
        if !m.is_static {
            return Err(mex(
                "RunMat:MethodStaticAccess",
                &format!("Method '{}' is not static", field),
            ));
        }
        Ok(Value::Closure(Closure {
            function_name: m.function_name,
            bound_function: None,
            captures: vec![],
        }))
    } else if runmat_builtins::class_has_enumeration_member(cls, field) {
        let mut value = runmat_builtins::ObjectInstance::new(cls.to_string());
        value
            .properties
            .insert("__enum_member__".to_string(), Value::String(field.to_string()));
        Ok(Value::Object(value))
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
    caller_function_name: Option<&str>,
    mut on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    match base {
        Value::Object(mut obj) => {
            if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                if class_defines_member_subsasgn(&cls)
                    && !caller_is_index_overload(
                        caller_function_name,
                        &obj.class_name,
                        ObjectIndexOp::Subsasgn,
                    )
                    && !caller_has_internal_class_access(caller_function_name, &obj.class_name)
                {
                    return call_object_member_subsasgn(Value::Object(obj), field, rhs).await;
                }
            }
            if let Some((p, owner)) = runmat_builtins::lookup_property(&obj.class_name, &field) {
                if p.is_static {
                    return Err(mex(
                        "RunMat:PropertyStaticAccess",
                        &format!(
                            "Property '{}' is static; use classref('{}').{}",
                            field, obj.class_name, field
                        ),
                    ));
                }
                if p.is_constant {
                    return Err(mex(
                        IDENT_PROPERTY_READ_ONLY,
                        &format!("Property '{}' is constant", field),
                    ));
                }
                if !access_permitted(&owner, &p.set_access, caller_function_name)
                {
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
                    return Err(mex(
                        "RunMat:PropertyStaticAccess",
                        &format!("Property '{}' is not static", field),
                    ));
                }
                if p.is_constant {
                    return Err(mex(
                        IDENT_PROPERTY_READ_ONLY,
                        &format!("Property '{}' is constant", field),
                    ));
                }
                if !access_permitted(&owner, &p.set_access, caller_function_name)
                {
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
            if let Some(cls) = runmat_builtins::get_class(&handle.class_name) {
                if class_defines_member_subsasgn(&cls)
                    && !caller_is_index_overload(
                        caller_function_name,
                        &handle.class_name,
                        ObjectIndexOp::Subsasgn,
                    )
                    && !caller_has_internal_class_access(caller_function_name, &handle.class_name)
                {
                    return call_object_member_subsasgn(Value::HandleObject(handle), field, rhs)
                        .await;
                }
            }
            runmat_runtime::call_builtin_async_with_outputs(
                "setfield",
                &[Value::HandleObject(handle), Value::String(field), rhs],
                1,
            )
            .await
        }
        Value::Listener(listener) => {
            runmat_runtime::call_builtin_async_with_outputs(
                "setfield",
                &[Value::Listener(listener), Value::String(field), rhs],
                1,
            )
            .await
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
    caller_function_name: Option<&str>,
    on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    store_member(base, name, rhs, allow_init, caller_function_name, on_write).await
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
                is_constant: false,
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
        let value = load_static_member(&child_name, "version", None)
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
                is_constant: false,
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
            None,
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
                is_abstract: false,
                is_sealed: false,
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

        let value = load_static_member(&child_name, "build", None)
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
                is_abstract: false,
                is_sealed: false,
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
        let value =
            futures::executor::block_on(load_member(obj, "missing".to_string(), false, None))
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
                is_abstract: false,
                is_sealed: false,
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
            None,
            |_old, _new| {},
        ))
        .expect("missing member store should dispatch to inherited subsasgn");
        let Value::Object(obj) = out else {
            panic!("expected object result from inherited subsasgn");
        };
        assert_eq!(obj.properties.get("missing"), Some(&Value::Num(13.0)));
    }
}
