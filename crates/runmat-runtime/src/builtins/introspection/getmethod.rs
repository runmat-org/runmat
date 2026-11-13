#[cfg(test)]
use runmat_builtins::{
    BuiltinExtensionMode, BuiltinIntegerAuditKind, GETMETHOD_EXTENSIONS, GETMETHOD_INTEGER_AUDIT,
};
use runmat_builtins::{
    GETMETHOD_DESCRIPTOR, GETMETHOD_ERROR_METHOD_PRIVATE, GETMETHOD_ERROR_NAME_INVALID,
    GETMETHOD_ERROR_RECEIVER_UNSUPPORTED, GETMETHOD_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;
pub(crate) fn dispatch_getmethod(obj: Value, name: String) -> crate::BuiltinResult<Value> {
    crate::compatibility::ensure_builtin_extension_enabled(&GETMETHOD_EXTENSION, "getmethod")?;

    fn ensure_method_accessible(
        class_name: &runmat_types::ClassIdentity,
        method_name: &str,
    ) -> crate::BuiltinResult<()> {
        let method_identity = runmat_types::MethodName::from(method_name);
        let Some((method, owner)) =
            crate::class_registry::lookup_method(class_name, &method_identity)
        else {
            return Ok(());
        };
        let caller_class = crate::class_access_context();
        let access_allowed = match method.access {
            runmat_types::MemberAccess::Public => true,
            runmat_types::MemberAccess::Private => caller_class.as_deref() == Some(&owner),
            runmat_types::MemberAccess::Protected => caller_class
                .as_deref()
                .is_some_and(|caller| crate::class_registry::is_class_or_subclass(caller, &owner)),
        };
        if access_allowed {
            return Ok(());
        }
        Err(crate::runtime_descriptor_error_with_detail(
            "getmethod",
            &GETMETHOD_ERROR_METHOD_PRIVATE,
            format!("{}.{}", class_name, method_name),
        ))
    }

    let method_name = name.trim();
    if method_name.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "getmethod",
            &GETMETHOD_ERROR_NAME_INVALID,
        ));
    }
    let caller_scope = crate::class_access_context()
        .map(|identity| Value::string(identity.display_name().to_owned()))
        .unwrap_or_else(|| Value::string(String::new()));
    match obj {
        Value::Object(o) => {
            ensure_method_accessible(o.class_name(), method_name)?;
            if let Some((resolved, _owner)) = crate::class_registry::lookup_method(
                o.class_name(),
                &runmat_types::MethodName::from(method_name),
            ) {
                let registered = crate::class_registry::lookup_bound_method(
                    o.class_name(),
                    &runmat_types::MethodName::from(method_name),
                )
                .and_then(|method| method.bound_function);
                let bound_function =
                    registered
                        .as_ref()
                        .map(|binding| binding.function)
                        .or_else(|| {
                            crate::user_functions::resolve_semantic_function_by_name(
                                &resolved.function_name,
                            )
                        });
                return Ok(Value::Closure(runmat_value::Closure::from_data(
                    runmat_value::ClosureData {
                        lexical_environment: None,
                        function_name: resolved.function_name.clone().into(),
                        binding: registered.or_else(|| {
                            bound_function.map(crate::user_functions::bound_function_binding_for)
                        }),
                        source_role: runmat_types::FunctionSourceRole::Method,
                        source_file: bound_function
                            .and_then(crate::user_functions::source_file_for)
                            .map(Into::into),
                        captures: vec![Value::Object(o)].into(),
                    },
                )));
            }
            Ok(Value::Closure(runmat_value::Closure::from_data(
                runmat_value::ClosureData {
                    lexical_environment: None,
                    function_name: crate::CALL_BOUND_METHOD_BUILTIN_NAME.into(),
                    binding: None,
                    source_role: runmat_types::FunctionSourceRole::Method,
                    source_file: None,
                    captures: vec![
                        Value::Object(o),
                        Value::string(method_name.to_string()),
                        caller_scope.clone(),
                    ]
                    .into(),
                },
            )))
        }
        Value::HandleObject(h) => {
            ensure_method_accessible(&h.class_name, method_name)?;
            if let Some((resolved, _owner)) = crate::class_registry::lookup_method(
                &h.class_name,
                &runmat_types::MethodName::from(method_name),
            ) {
                let registered = crate::class_registry::lookup_bound_method(
                    &h.class_name,
                    &runmat_types::MethodName::from(method_name),
                )
                .and_then(|method| method.bound_function);
                let bound_function =
                    registered
                        .as_ref()
                        .map(|binding| binding.function)
                        .or_else(|| {
                            crate::user_functions::resolve_semantic_function_by_name(
                                &resolved.function_name,
                            )
                        });
                return Ok(Value::Closure(runmat_value::Closure::from_data(
                    runmat_value::ClosureData {
                        lexical_environment: None,
                        function_name: resolved.function_name.clone().into(),
                        binding: registered.or_else(|| {
                            bound_function.map(crate::user_functions::bound_function_binding_for)
                        }),
                        source_role: runmat_types::FunctionSourceRole::Method,
                        source_file: bound_function
                            .and_then(crate::user_functions::source_file_for)
                            .map(Into::into),
                        captures: vec![Value::HandleObject(h)].into(),
                    },
                )));
            }
            Ok(Value::Closure(runmat_value::Closure::from_data(
                runmat_value::ClosureData {
                    lexical_environment: None,
                    function_name: crate::CALL_BOUND_METHOD_BUILTIN_NAME.into(),
                    binding: None,
                    source_role: runmat_types::FunctionSourceRole::Method,
                    source_file: None,
                    captures: vec![
                        Value::HandleObject(h),
                        Value::string(method_name.to_string()),
                        caller_scope,
                    ]
                    .into(),
                },
            )))
        }
        Value::ClassRef(cls) => {
            ensure_method_accessible(&cls, method_name)?;
            let method = runmat_types::MethodName::from(method_name);
            if let Some((declaration, _)) = crate::class_registry::lookup_method(&cls, &method) {
                if declaration.is_static {
                    if let Some(binding) = crate::class_registry::lookup_bound_method(&cls, &method)
                        .and_then(|method| method.bound_function)
                    {
                        let function = binding.function;
                        return Ok(Value::Closure(runmat_value::Closure::from_data(
                            runmat_value::ClosureData {
                                lexical_environment: None,
                                function_name: declaration.function_name.into(),
                                binding: Some(binding),
                                source_role: runmat_types::FunctionSourceRole::Method,
                                source_file: crate::user_functions::source_file_for(function)
                                    .map(Into::into),
                                captures: Vec::new().into(),
                            },
                        )));
                    }
                }
            }
            crate::builtins::introspection::function_handle_text::dispatch_str2func(Value::string(
                format!("@{cls}.{method_name}"),
            ))
        }
        other => Err(crate::runtime_descriptor_error_with_detail(
            "getmethod",
            &GETMETHOD_ERROR_RECEIVER_UNSUPPORTED,
            format!("{other:?}"),
        )),
    }
}

pub async fn getmethod_builtin(obj: Value, name: Value) -> crate::BuiltinResult<Value> {
    crate::compatibility::ensure_builtin_extension_enabled(&GETMETHOD_EXTENSION, "getmethod")?;
    let name = match name {
        Value::String(runmat_value::StringElement::Text(name)) => name.as_str().to_owned(),
        Value::CharArray(chars) if chars.rows == 1 => {
            super::text::decode_utf16_with_error(chars.data(), || {
                crate::runtime_descriptor_error_with_detail(
                    "getmethod",
                    &GETMETHOD_ERROR_NAME_INVALID,
                    "method name is not valid UTF-16 text",
                )
            })?
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned()
        }
        other => {
            return Err(crate::runtime_descriptor_error_with_detail(
                "getmethod",
                &GETMETHOD_ERROR_NAME_INVALID,
                format!("method name must be host text, got {other:?}"),
            ))
        }
    };
    dispatch_getmethod(obj, name)
}

#[runtime_builtin(
    name = "getmethod",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::getmethod"
)]
pub async fn getmethod_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &GETMETHOD_DESCRIPTOR,
        &runmat_builtins::GETMETHOD_ERROR_OUTPUTS,
        "getmethod",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "getmethod",
            &runmat_builtins::GETMETHOD_ERROR_ARITY,
        ));
    }
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
    };
    let policy = HostAllocationReceipt::policy(runtime.as_ref()).map_err(|cause| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:Placement:ResourceLimit",
            "getmethod: allocation authority unavailable",
            cause,
        )
    })?;
    if policy == HostAllocationPolicy::Managed {
        return Err(crate::build_runtime_error("getmethod: closure captures and dispatch metadata have no certified finite-resource bound").with_builtin("getmethod").with_identifier("RunMat:Placement:InsufficientResourceEvidence").build());
    }
    let future = async {
        let result = getmethod_builtin(args[0].clone(), args[1].clone()).await;
        drop(args);
        result
    };
    match runtime.as_ref() {
        Some(context) => context.scope(future).await,
        None => future.await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callable_is_registered_as_a_runmat_only_extension() {
        assert_eq!(GETMETHOD_EXTENSIONS, [GETMETHOD_EXTENSION]);
        assert_eq!(GETMETHOD_EXTENSION.mode, BuiltinExtensionMode::RunMatOnly);
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = futures::executor::block_on(getmethod_builtin(
            Value::ClassRef("Example".into()),
            Value::String("method".into()),
        ))
        .expect_err("strict compatibility rejects RunMat getmethod");
        assert_eq!(error.identifier(), GETMETHOD_EXTENSION.error_identifier);
    }

    #[test]
    fn integer_audit_rejects_all_integer_receivers_and_resident_numeric() {
        assert_eq!(
            GETMETHOD_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for value in [
            Value::Int(runmat_value::IntValue::I8(1)),
            Value::Int(runmat_value::IntValue::I16(1)),
            Value::Int(runmat_value::IntValue::I32(1)),
            Value::Int(runmat_value::IntValue::I64(1)),
            Value::Int(runmat_value::IntValue::U8(1)),
            Value::Int(runmat_value::IntValue::U16(1)),
            Value::Int(runmat_value::IntValue::U32(1)),
            Value::Int(runmat_value::IntValue::U64(1)),
        ] {
            let error = futures::executor::block_on(getmethod_builtin(
                value,
                Value::String("method".into()),
            ))
            .expect_err("integer receiver");
            assert_eq!(
                error.identifier(),
                GETMETHOD_ERROR_RECEIVER_UNSUPPORTED.identifier
            );
        }
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("valid unregistered resident fixture")
                .handle,
        );
        let error = futures::executor::block_on(getmethod_builtin(
            resident,
            Value::String("method".into()),
        ))
        .expect_err("resident receiver");
        assert_eq!(
            error.identifier(),
            GETMETHOD_ERROR_RECEIVER_UNSUPPORTED.identifier
        );

        let name_error = futures::executor::block_on(getmethod_builtin(
            Value::ClassRef("Example".into()),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        ))
        .expect_err("integer method name rejects without text conversion");
        assert_eq!(
            name_error.identifier(),
            GETMETHOD_ERROR_NAME_INVALID.identifier
        );
    }
}
