#[cfg(test)]
use runmat_builtins::BuiltinIntegerAuditKind;
pub use runmat_builtins::{
    FUNC2STR_DESCRIPTOR, FUNC2STR_ERROR_HANDLE_TYPE_INVALID, FUNC2STR_ERROR_TOO_MANY_OUTPUTS,
    FUNC2STR_INTEGER_AUDIT, STR2FUNC_DESCRIPTOR, STR2FUNC_ERROR_NAME_INVALID,
    STR2FUNC_ERROR_NAME_SHAPE_INVALID, STR2FUNC_ERROR_NAME_TYPE_INVALID,
    STR2FUNC_ERROR_TOO_MANY_OUTPUTS, STR2FUNC_INTEGER_AUDIT,
};
use runmat_value::Value;

pub(crate) fn dispatch_str2func(value: Value) -> crate::BuiltinResult<Value> {
    fn normalize_handle_name(text: &str) -> Option<String> {
        let trimmed = text.trim();
        let name = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
        (!name.is_empty()).then(|| name.to_string())
    }

    let name = match value {
        Value::String(runmat_value::StringElement::Text(text)) => normalize_handle_name(&text),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            normalize_handle_name(
                sa.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string"),
            )
        }
        Value::StringArray(_) => {
            return Err(crate::runtime_descriptor_error_with_detail(
                "str2func",
                &STR2FUNC_ERROR_NAME_SHAPE_INVALID,
                "string array must be scalar",
            ))
        }
        Value::CharArray(ca) if ca.rows == 1 && ca.shape().iter().skip(2).all(|n| *n == 1) => {
            let text = super::text::decode_utf16_with_error(ca.data(), || {
                crate::runtime_descriptor_error_with_detail(
                    "str2func",
                    &STR2FUNC_ERROR_NAME_INVALID,
                    "function name is not valid UTF-16 text",
                )
            })?;
            normalize_handle_name(&text)
        }
        Value::CharArray(_) => {
            return Err(crate::runtime_descriptor_error_with_detail(
                "str2func",
                &STR2FUNC_ERROR_NAME_SHAPE_INVALID,
                "char array must be a row vector",
            ))
        }
        other => {
            return Err(crate::runtime_descriptor_error_with_detail(
                "str2func",
                &STR2FUNC_ERROR_NAME_TYPE_INVALID,
                format!("got {other:?}"),
            ))
        }
    }
    .ok_or_else(|| crate::runtime_descriptor_error("str2func", &STR2FUNC_ERROR_NAME_INVALID))?;

    if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(&name) {
        Ok(crate::user_functions::bound_function_handle(name, function))
    } else if crate::is_well_formed_qualified_name(&name) {
        Ok(Value::external_function_handle(name))
    } else {
        Ok(Value::function_handle(name))
    }
}

#[runmat_macros::runtime_builtin(
    name = "str2func",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::function_handle_text"
)]
pub fn str2func_builtin_registered(value: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &STR2FUNC_DESCRIPTOR,
        &STR2FUNC_ERROR_TOO_MANY_OUTPUTS,
        "str2func",
    )?;
    dispatch_str2func(value)
}

pub(crate) fn dispatch_func2str(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => Ok(Value::String(
            runmat_value::StringElement::Text(name.into_text_owner()),
        )),
        Value::BoundFunctionHandle(handle) => Ok(Value::String(runmat_value::StringElement::Text(
            handle.name.clone(),
        ))),
        Value::Closure(closure) => Ok(Value::String(runmat_value::StringElement::Text(
            closure.function_name.clone(),
        ))),
        Value::ScopedFunctionHandle(_) => Ok(Value::from("<invocation-scoped callable>")),
        other => Err(crate::runtime_descriptor_error_with_detail(
            "func2str",
            &FUNC2STR_ERROR_HANDLE_TYPE_INVALID,
            format!("got {other:?}"),
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "func2str",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::function_handle_text"
)]
pub fn func2str_builtin_registered(value: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &FUNC2STR_DESCRIPTOR,
        &FUNC2STR_ERROR_TOO_MANY_OUTPUTS,
        "func2str",
    )?;
    dispatch_func2str(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn str2func_rejects_character_pages_and_excess_outputs() {
        let pages =
            runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
                .unwrap();
        let error = dispatch_str2func(Value::CharArray(pages)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Str2FuncNameShapeInvalid"));
        let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
        let error = str2func_builtin_registered(Value::string("sin")).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:str2func:TooManyOutputs"));
    }

    #[test]
    fn func2str_returns_existing_name_owner_without_resolving() {
        for value in [
            Value::function_handle("unresolved_text_target".into()),
            Value::external_function_handle("pkg.unresolved_target".into()),
            Value::method_function_handle("Class.method".into()),
        ] {
            let (expected, backing) = match &value {
                Value::FunctionHandle(name)
                | Value::ExternalFunctionHandle(name)
                | Value::MethodFunctionHandle(name) => {
                    (name.name().to_owned(), name.name().as_ptr())
                }
                _ => unreachable!(),
            };
            let result = dispatch_func2str(value).unwrap();
            let Value::String(runmat_value::StringElement::Text(text)) = result else {
                panic!("string scalar expected")
            };
            assert_eq!(text.as_str(), expected);
            assert_eq!(text.as_str().as_ptr(), backing);
        }
    }

    #[test]
    fn func2str_rejects_excess_outputs_before_input_inspection() {
        let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
        let error = func2str_builtin_registered(Value::Num(1.0)).unwrap_err();
        assert_eq!(
            error.identifier(),
            FUNC2STR_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }

    fn unregistered_resident_value() -> Value {
        Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("valid unregistered resident fixture")
                .handle,
        )
    }

    #[test]
    fn str2func_integer_audit_is_inapplicable() {
        assert_eq!(
            STR2FUNC_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert!(STR2FUNC_INTEGER_AUDIT.canonical_builtin.is_none());
    }

    #[test]
    fn str2func_rejects_integer_and_resident_values_before_provider_access() {
        for value in [
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                    .expect("integer tensor"),
            ),
            unregistered_resident_value(),
        ] {
            let error = dispatch_str2func(value).expect_err("non-text must reject");
            assert_eq!(error.identifier(), Some("RunMat:Str2FuncNameTypeInvalid"));
        }
    }

    #[test]
    fn func2str_integer_audit_is_inapplicable() {
        assert_eq!(
            FUNC2STR_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert!(FUNC2STR_INTEGER_AUDIT.canonical_builtin.is_none());
    }

    #[test]
    fn func2str_rejects_integer_and_resident_values_before_provider_access() {
        let integers = [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ];
        for value in integers.into_iter().map(Value::Int).chain([
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                    .expect("integer tensor"),
            ),
            unregistered_resident_value(),
        ]) {
            let error = dispatch_func2str(value).expect_err("non-handle rejection");
            assert_eq!(
                error.identifier(),
                FUNC2STR_ERROR_HANDLE_TYPE_INVALID.identifier
            );
            assert!(!error.message().to_ascii_lowercase().contains("provider"));
        }
    }
}
