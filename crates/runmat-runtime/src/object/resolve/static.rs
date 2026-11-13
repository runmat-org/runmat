use super::*;

pub(super) fn load_static_member_with_caller(
    cls: &ClassIdentity,
    field: &str,
    caller_function_name: MemberCaller<'_>,
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
        let binding =
            crate::class_registry::lookup_bound_method(cls, &runmat_types::MethodName::from(field))
                .and_then(|method| method.bound_function);
        let source_file = binding
            .as_ref()
            .and_then(|binding| crate::user_functions::source_file_for(binding.function))
            .map(Into::into);
        Ok(Value::Closure(Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: m.function_name.into(),
                binding,
                source_role: runmat_types::FunctionSourceRole::Method,
                source_file,
                captures: vec![].into(),
            },
        )))
    } else if crate::class_registry::class_has_enumeration_member(cls, field) {
        Ok(enumeration_value_unbounded(cls, field))
    } else {
        let qualified = external_qualified_display_name(cls.display_name(), field);
        if let Some(builtin) = crate::dispatcher::observed_dispatch_builtin_name(&qualified) {
            Ok(Value::Closure(Closure::from_data(
                runmat_value::ClosureData {
                    lexical_environment: None,
                    function_name: builtin.0.into(),
                    binding: None,
                    source_role: runmat_types::FunctionSourceRole::Method,
                    source_file: None,
                    captures: vec![].into(),
                },
            )))
        } else {
            Err(format!("Unknown property '{}' on class {}", field, cls).into())
        }
    }
}

const ENUMERATION_MEMBER_PROPERTY: &str = "__enum_member__";
fn enumeration_value_unbounded(class: &ClassIdentity, member: &str) -> Value {
    let mut value = runmat_value::UnboundedObjectBuilder::new(class.clone());
    value.insert(
        ENUMERATION_MEMBER_PROPERTY.to_owned(),
        Value::string(member.to_owned()),
    );
    Value::Object(value.finish())
}
/// Decode the same carrier written by canonical static-member construction.
/// Consumers retain responsibility for their typed class/member contract.
pub(crate) fn enumeration_member_name(object: &runmat_value::ObjectInstance) -> Option<&str> {
    match object.property(ENUMERATION_MEMBER_PROPERTY) {
        Some(Value::String(name)) => name.as_str(),
        _ => None,
    }
}
/// Publish a registered member through the canonical original object allocator.
pub(crate) fn enumeration_value_in(
    runtime: Option<&crate::context::RuntimeContext>,
    class: &ClassIdentity,
    member: &str,
) -> crate::BuiltinResult<RootedValue> {
    crate::context::with_original_scope(runtime, || {
        if !crate::class_registry::class_has_enumeration_member(class, member) {
            return Err(
                crate::build_runtime_error("unregistered enumeration member")
                    .with_identifier("RunMat:EnumerationMember")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build(),
            );
        }
        let value = Value::String(crate::call::argument_text::copy_str(member, runtime)?.into());
        crate::execution::object::construct_object_from_borrowed_fields(
            runtime,
            class.display_name(),
            [(ENUMERATION_MEMBER_PROPERTY, &value)].into_iter(),
        )
    })
}

pub fn load_static_member(
    class: &ClassIdentity,
    field: &str,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    load_static_member_with_caller(class, field, MemberCaller::named(caller_function_name))
}
