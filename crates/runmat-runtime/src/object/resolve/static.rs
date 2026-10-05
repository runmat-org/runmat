use super::*;

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
