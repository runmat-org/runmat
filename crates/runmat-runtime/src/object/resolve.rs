//! Executor-neutral member and static-member resolution.
//!
//! This module owns MATLAB struct/object/handle member access, class access
//! control, dependent and dynamic properties, `subsref`/`subsasgn` overloads,
//! graphics-handle compatibility, and canonical GC-barrier stores. Executors
//! provide only their current function identity for access checks and attach
//! their own frame/source context to returned runtime errors.

use crate::builtins::introspection::dynamicprops;
use crate::call::identity::external_qualified_display_name;
use crate::object::dispatch::{
    call_object_property_getter_with_outputs, call_object_property_setter_with_outputs,
};
use crate::object::indexing::ObjectIndexOp;
use crate::RuntimeError;
use runmat_types::ClassIdentity;
use runmat_value::{Closure, StructValue, Tensor, Value};

const IDENT_PROPERTY_PRIVATE_ACCESS: &str = "RunMat:PropertyPrivateAccess";
const IDENT_PROPERTY_READ_ONLY: &str = "RunMat:PropertyReadOnly";

fn mex(identifier: &str, message: &str) -> RuntimeError {
    crate::runtime_error::semantic_error(identifier, message)
}

fn caller_class_for_function(caller_function_name: Option<&str>) -> Option<ClassIdentity> {
    caller_function_name
        .and_then(crate::class_registry::caller_method_for_function)
        .map(|(class, _)| class)
}

fn access_permitted(
    owner: &ClassIdentity,
    access: &runmat_types::MemberAccess,
    caller_function_name: Option<&str>,
) -> bool {
    match access {
        runmat_types::MemberAccess::Public => true,
        runmat_types::MemberAccess::Private => {
            caller_class_for_function(caller_function_name).as_ref() == Some(owner)
        }
        runmat_types::MemberAccess::Protected => caller_class_for_function(caller_function_name)
            .is_some_and(|caller_class| {
                crate::class_registry::is_class_or_subclass(&caller_class, owner)
            }),
    }
}

pub async fn load_member(
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    load_member_with_context(None, base, field, allow_init, caller_function_name).await
}

pub async fn load_member_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    let mut values =
        read_member_sequence_with_context(context, base, field, allow_init, caller_function_name)
            .await?
            .resolve(
                runmat_types::SequenceUse::RequireSingle,
                crate::sequence::SequenceResolutionContext::default(),
            )?;
    values
        .pop()
        .ok_or_else(|| RuntimeError::from("member read did not produce one value"))
}

pub async fn read_member_sequence_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<crate::sequence::ValueSequence, RuntimeError> {
    match base {
        Value::StructArray(array) => crate::aggregate::structure::gather_member(array, &field),
        Value::ObjectArray(array) => {
            let mut values = Vec::with_capacity(array.len());
            for value in array.into_data() {
                let mut selected = Box::pin(read_member_sequence_with_context(
                    context,
                    value,
                    field.clone(),
                    allow_init,
                    caller_function_name,
                ))
                .await?
                .resolve(
                    runmat_types::SequenceUse::RequireSingle,
                    crate::sequence::SequenceResolutionContext::default(),
                )?;
                values.push(selected.pop().expect("single sequence resolution"));
            }
            Ok(crate::sequence::ValueSequence::comma_separated(values))
        }
        base => {
            load_scalar_member_with_context(context, base, field, allow_init, caller_function_name)
                .await
                .map(crate::sequence::ValueSequence::single)
        }
    }
}

async fn load_scalar_member_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    if let Some(result) = context
        .and_then(|context| crate::parallel::introspection::load_member(context, &base, &field))
    {
        return result;
    }
    match base {
        Value::Object(obj) => {
            let base = Value::Object(obj.clone());
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let crate::object::protocol::ProtocolResolution::Method(method) =
                crate::object::protocol::resolve_object_protocol(
                    &base,
                    crate::object::protocol::ObjectProtocol::Subsref,
                    &access,
                )?
            {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
                );
                return crate::object::protocol::invoke_resolved_object_protocol(
                    &crate::object::protocol::ProtocolResolution::Method(method),
                    base,
                    path,
                    1,
                )
                .await;
            }
            if let Some((p, owner)) = crate::class_registry::lookup_property(
                &obj.class_name,
                &runmat_types::MemberName::from(field.as_str()),
            ) {
                if p.is_static {
                    return Err(mex(
                        "RunMat:PropertyStaticAccess",
                        &format!(
                            "Property '{}' is static; use classref('{}').{}",
                            field, obj.class_name, field
                        ),
                    ));
                }
                if !access_permitted(&owner, &p.get_access, caller_function_name) {
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
                }
                if p.is_dependent {
                    let getter = crate::object_property_getter_name(&field);
                    if crate::class_registry::lookup_method(
                        &obj.class_name,
                        &runmat_types::MethodName::from(getter.as_str()),
                    )
                    .is_some()
                    {
                        return call_object_property_getter_with_outputs(
                            Value::Object(obj.clone()),
                            &field,
                            1,
                        )
                        .await;
                    }
                }
            }
            if let Some(v) = dynamicprops::dynamic_property_read(&obj, &field)? {
                Ok(v)
            } else if let Some(v) = obj.properties.get(&field) {
                Ok(v.clone())
            } else if let Some((p2, owner)) = crate::class_registry::lookup_property(
                &obj.class_name,
                &runmat_types::MemberName::from(field.as_str()),
            ) {
                if !access_permitted(&owner, &p2.get_access, caller_function_name) {
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
            } else if crate::class_registry::get_class(&obj.class_name).is_some() {
                Err(format!(
                    "Undefined property '{}' for class {}",
                    field, obj.class_name
                )
                .into())
            } else {
                Err(format!("Unknown class {}", obj.class_name).into())
            }
        }
        Value::HandleObject(handle) => {
            let base = Value::HandleObject(handle.clone());
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let crate::object::protocol::ProtocolResolution::Method(method) =
                crate::object::protocol::resolve_object_protocol(
                    &base,
                    crate::object::protocol::ObjectProtocol::Subsref,
                    &access,
                )?
            {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
                );
                return crate::object::protocol::invoke_resolved_object_protocol(
                    &crate::object::protocol::ProtocolResolution::Method(method),
                    base,
                    path,
                    1,
                )
                .await;
            }
            crate::builtins::structs::core::getfield::get_member_value(
                Value::HandleObject(handle),
                &field,
            )
            .await
        }
        Value::Listener(listener) => {
            crate::builtins::structs::core::getfield::get_member_value(
                Value::Listener(listener),
                &field,
            )
            .await
        }
        Value::Foreign(reference) => crate::foreign::load_foreign_member(reference, field).await,
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
        Value::StructArray(_) | Value::ObjectArray(_) => Err(mex(
            "MemberSequenceInvariant",
            "aggregate member reads must pass through sequence resolution",
        )),
        Value::Cell(_) => Err(mex("LoadMember", "LoadMember on non-object")),
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
                    let cell = runmat_value::CellArray::new(values, rows, 1)
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

pub async fn load_member_dynamic_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    name: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    load_member_with_context(context, base, name, allow_init, caller_function_name).await
}

pub fn load_static_member(
    cls: &ClassIdentity,
    field: &str,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    if let Some((p, owner)) =
        crate::class_registry::lookup_property(cls, &runmat_types::MemberName::from(field))
    {
        if !p.is_static {
            return Err(mex(
                "RunMat:PropertyStaticAccess",
                &format!("Property '{}' is not static", field),
            ));
        }
        if !access_permitted(&owner, &p.get_access, caller_function_name) {
            return Err(mex(
                IDENT_PROPERTY_PRIVATE_ACCESS,
                &format!("Property '{}' is private", field),
            ));
        }
        if let Some(v) = crate::class_registry::get_static_property_value(&owner, field) {
            Ok(v)
        } else if let Some(v) = &p.default_value {
            Ok(v.clone())
        } else {
            Ok(Value::Tensor(
                Tensor::new(vec![], vec![0, 0]).expect("empty tensor"),
            ))
        }
    } else if let Some((m, _owner)) =
        crate::class_registry::lookup_method(cls, &runmat_types::MethodName::from(field))
    {
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
    } else if crate::class_registry::class_has_enumeration_member(cls, field) {
        let mut value = runmat_value::ObjectInstance::new(cls.clone());
        value.properties.insert(
            "__enum_member__".to_string(),
            Value::String(field.to_string()),
        );
        Ok(Value::Object(value))
    } else {
        let qualified = external_qualified_display_name(cls.display_name(), field);
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
            let base = Value::Object(obj.clone());
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let crate::object::protocol::ProtocolResolution::Method(method) =
                crate::object::protocol::resolve_object_protocol(
                    &base,
                    crate::object::protocol::ObjectProtocol::Subsasgn,
                    &access,
                )?
            {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
                );
                return crate::object::protocol::invoke_resolved_object_assignment(
                    &method,
                    base,
                    path,
                    vec![rhs],
                )
                .await;
            }
            if let Some((p, owner)) = crate::class_registry::lookup_property(
                &obj.class_name,
                &runmat_types::MemberName::from(field.as_str()),
            ) {
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
                if !access_permitted(&owner, &p.set_access, caller_function_name) {
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
                }
                if p.is_dependent {
                    let setter = crate::object_property_setter_name(&field);
                    if crate::class_registry::lookup_method(
                        &obj.class_name,
                        &runmat_types::MethodName::from(setter.as_str()),
                    )
                    .is_some()
                    {
                        return call_object_property_setter_with_outputs(
                            Value::Object(obj.clone()),
                            &field,
                            rhs.clone(),
                            1,
                        )
                        .await;
                    }
                }
                if let Some(oldv) = obj.properties.get(&field) {
                    on_write(oldv, &rhs);
                }
                obj.properties.insert(field, rhs);
                Ok(Value::Object(obj))
            } else if dynamicprops::dynamic_property_exists(&obj, &field) {
                if let Some(oldv) = obj.properties.get(&field) {
                    on_write(oldv, &rhs);
                }
                dynamicprops::dynamic_property_assign(&mut obj, &field, rhs)?;
                Ok(Value::Object(obj))
            } else if crate::class_registry::get_class(&obj.class_name).is_some() {
                Err(format!("Undefined property '{}' for class {}", field, obj.class_name).into())
            } else {
                Err(format!("Unknown class {}", obj.class_name).into())
            }
        }
        Value::ClassRef(cls) => {
            if let Some((p, owner)) = crate::class_registry::lookup_property(
                &cls,
                &runmat_types::MemberName::from(field.as_str()),
            ) {
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
                if !access_permitted(&owner, &p.set_access, caller_function_name) {
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
                }
                crate::class_registry::set_static_property_value_in_owner(&owner, &field, rhs)?;
                Ok(Value::ClassRef(cls))
            } else {
                Err(format!("Unknown property '{}' on class {}", field, cls).into())
            }
        }
        Value::HandleObject(handle) => {
            let base = Value::HandleObject(handle.clone());
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let crate::object::protocol::ProtocolResolution::Method(method) =
                crate::object::protocol::resolve_object_protocol(
                    &base,
                    crate::object::protocol::ObjectProtocol::Subsasgn,
                    &access,
                )?
            {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
                );
                return crate::object::protocol::invoke_resolved_object_assignment(
                    &method,
                    base,
                    path,
                    vec![rhs],
                )
                .await;
            }
            crate::call_builtin_async_with_outputs(
                "setfield",
                &[Value::HandleObject(handle), Value::String(field), rhs],
                1,
            )
            .await
        }
        Value::Listener(listener) => {
            crate::call_builtin_async_with_outputs(
                "setfield",
                &[Value::Listener(listener), Value::String(field), rhs],
                1,
            )
            .await
        }
        Value::Foreign(reference) => {
            crate::foreign::store_foreign_member(reference, field, rhs).await
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
        Value::StructArray(_) => Err(mex(
            "RunMat:CommaSeparatedListAssignmentArity",
            "simple member assignment cannot target multiple structure elements; use a comma-separated destination list",
        )),
        Value::Cell(_) => Err(mex("StoreMember", "StoreMember on non-object")),
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

pub fn member_sequence_cardinality(base: &Value) -> Result<usize, RuntimeError> {
    match base {
        Value::Struct(_) | Value::Object(_) | Value::HandleObject(_) => Ok(1),
        Value::StructArray(array) => Ok(array.len()),
        Value::ObjectArray(array) => Ok(array.len()),
        _ => Err(mex(
            "RunMat:CommaSeparatedListDestination",
            "comma-separated member assignment requires a structure or object aggregate",
        )),
    }
}

pub async fn store_member_sequence_traced(
    base: Value,
    field: String,
    values: Vec<Value>,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    match base {
        Value::StructArray(array) => crate::aggregate::structure::assign_member_values(
            array,
            field,
            values,
            runmat_gc::gc_record_write,
        ),
        Value::ObjectArray(array) => {
            if values.len() != array.len() {
                return Err(mex(
                    "RunMat:CommaSeparatedListAssignmentArity",
                    &format!(
                        "member assignment requires exactly one value per destination element (expected {}, received {})",
                        array.len(),
                        values.len()
                    ),
                ));
            }
            let aggregate = Value::ObjectArray(array);
            let resolution = crate::object::dispatch::resolve_object_index_protocol(
                &aggregate,
                ObjectIndexOp::Subsasgn,
                caller_function_name,
            )?;
            if let crate::object::protocol::ProtocolResolution::Method(method) = resolution {
                return crate::object::protocol::invoke_resolved_object_assignment(
                    &method,
                    aggregate,
                    crate::object::indexing::ObjectSubscriptPath::single(
                        crate::object::indexing::ObjectSubscript::member(field),
                    ),
                    values,
                )
                .await;
            }
            let Value::ObjectArray(array) = aggregate else {
                unreachable!("aggregate was constructed as an object array")
            };
            let class_name = array.class_name().clone();
            let shape = array.shape().to_vec();
            let mut updated = Vec::with_capacity(values.len());
            for (element, value) in array.into_data().into_iter().zip(values) {
                updated.push(
                    store_member_traced(element, field.clone(), value, false, caller_function_name)
                        .await?,
                );
            }
            runmat_value::ObjectArray::new(class_name, updated, shape)
                .map(Value::ObjectArray)
                .map_err(RuntimeError::from)
        }
        base if values.len() == 1 => {
            let value = values.into_iter().next().expect("validated singleton");
            store_member_traced(base, field, value, false, caller_function_name).await
        }
        _ => Err(mex(
            "RunMat:CommaSeparatedListAssignmentArity",
            "member assignment requires exactly one value per destination element",
        )),
    }
}

/// Store a member while applying the canonical GC write barrier.
pub async fn store_member_traced(
    base: Value,
    field: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    store_member(
        base,
        field,
        rhs,
        allow_init,
        caller_function_name,
        runmat_gc::gc_record_write,
    )
    .await
}

/// Store a dynamically named member while applying the canonical GC barrier.
pub async fn store_member_dynamic_traced(
    base: Value,
    name: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    store_member_dynamic(
        base,
        name,
        rhs,
        allow_init,
        caller_function_name,
        runmat_gc::gc_record_write,
    )
    .await
}

async fn load_graphics_member(base: Value, field: &str) -> Result<Value, RuntimeError> {
    crate::call_builtin_async("get", &[base, Value::String(field.to_string())]).await
}

async fn store_graphics_member(
    base: Value,
    field: &str,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    crate::call_builtin_async(
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
        Value::Num(v) => v.is_finite() && *v >= 0.0,
        Value::Int(i) => i.try_to_u64().is_some(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        is_possible_graphics_handle_value, load_member, load_static_member,
        read_member_sequence_with_context, store_member,
    };
    use runmat_types::{ClassIdentity, MemberAccess};
    use runmat_value::{IntValue, ObjectArray, ObjectInstance, Value};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn graphics_handle_predicate_preserves_wide_integer_sign() {
        assert!(is_possible_graphics_handle_value(&Value::Int(
            IntValue::U64(u64::MAX)
        )));
        assert!(!is_possible_graphics_handle_value(&Value::Int(
            IntValue::I64(-1)
        )));
    }

    fn unique_class_name(prefix: &str) -> ClassIdentity {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        ClassIdentity::from(format!("{}_{}", prefix, id))
    }

    #[test]
    fn object_array_member_access_returns_comma_separated_values_in_order() {
        let values = ["first", "second"]
            .into_iter()
            .map(|name| {
                let mut object = ObjectInstance::new("matlab.unittest.TestResult");
                object
                    .properties
                    .insert("Name".into(), Value::String(name.into()));
                Value::Object(object)
            })
            .collect();
        let array =
            Value::ObjectArray(ObjectArray::row("matlab.unittest.TestResult", values).unwrap());
        let loaded = futures::executor::block_on(read_member_sequence_with_context(
            None,
            array,
            "Name".into(),
            false,
            None,
        ))
        .unwrap()
        .resolve(
            runmat_types::SequenceUse::ExpandAll,
            crate::sequence::SequenceResolutionContext::default(),
        )
        .unwrap();
        assert_eq!(
            loaded,
            vec![
                Value::String("first".into()),
                Value::String("second".into())
            ]
        );
    }

    #[test]
    fn load_static_member_resolves_inherited_static_property_value() {
        let parent_name = unique_class_name("vm_static_parent");
        let child_name = unique_class_name("vm_static_child");

        let mut parent_properties = HashMap::new();
        parent_properties.insert(
            "version".into(),
            crate::class_registry::RuntimeProperty {
                name: "version".into(),
                is_static: true,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: Some(Value::Num(1.0)),
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: parent_name.clone(),
            parent: None,
            properties: parent_properties,
            methods: HashMap::new(),
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: child_name.clone(),
            parent: Some(parent_name.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        crate::class_registry::set_static_property_value(&parent_name, "version", Value::Num(3.0));
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
            "version".into(),
            crate::class_registry::RuntimeProperty {
                name: "version".into(),
                is_static: true,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: Some(Value::Num(1.0)),
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: parent_name.clone(),
            parent: None,
            properties: parent_properties,
            methods: HashMap::new(),
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
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
            crate::class_registry::get_static_property_value(&parent_name, "version"),
            Some(Value::Num(9.0))
        );
    }

    #[test]
    fn load_static_member_resolves_inherited_static_method() {
        let parent_name = unique_class_name("vm_method_parent");
        let child_name = unique_class_name("vm_method_child");

        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            "build".into(),
            crate::class_registry::RuntimeMethod {
                name: "build".into(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "build_impl".into(),
                implicit_class_argument: None,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: parent_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
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
            "subsref".into(),
            crate::class_registry::RuntimeMethod {
                name: "subsref".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "OverIdx.subsref".into(),
                implicit_class_argument: None,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: parent_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
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
            "subsasgn".into(),
            crate::class_registry::RuntimeMethod {
                name: "subsasgn".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "OverIdx.subsasgn".into(),
                implicit_class_argument: None,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: parent_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
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
