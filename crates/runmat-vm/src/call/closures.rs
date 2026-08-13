use crate::interpreter::stack::{pop_args, pop_value};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

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
    stack.push(runmat_runtime::call::closures::closure_value(
        func_name, captures,
    ));
    Ok(())
}

pub fn create_semantic_closure(
    stack: &mut Vec<Value>,
    function: runmat_hir::FunctionId,
    display_name: String,
    capture_count: usize,
) -> Result<(), RuntimeError> {
    let mut captures = Vec::with_capacity(capture_count);
    for _ in 0..capture_count {
        captures.push(pop_value(stack)?);
    }
    captures.reverse();
    stack.push(runmat_runtime::call::closures::semantic_closure_value(
        function,
        display_name,
        captures,
    ));
    Ok(())
}

pub fn collect_method_args(
    stack: &mut Vec<Value>,
    arg_count: usize,
) -> Result<(Value, Vec<Value>), RuntimeError> {
    let args = pop_args(stack, arg_count)?;
    let base = pop_value(stack)?;
    Ok((base, args))
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_hir::{CallableFallbackPolicy, CallableIdentity, MethodId};
    use runmat_runtime::call::closures::load_method_closure;
    use runmat_runtime::object::dispatch::call_method_or_member_index_with_outputs;
    use runmat_types::MemberAccess;
    use runmat_value::Value;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn classref_external_method_uses_external_boundary_semantic_resolution() {
        let class_name = "ClassRefExternalMethodResolutionTest".to_string();
        let resolved_name = format!("{class_name}.remote_inc");
        runmat_runtime::class_registry::register_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: class_name.clone(),
                parent: None,
                properties: HashMap::new(),
                methods: HashMap::new(),
            },
        );
        let _resolver_guard =
            runmat_runtime::user_functions::install_semantic_function_resolver(Some(Arc::new(
                move |name| (name == resolved_name).then_some(7331),
            )));
        let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(
            Some(Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 7331);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(2.0)]);
                Box::pin(async { Ok(Value::Num(3.0)) })
            })),
        );
        let value = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef(class_name),
            CallableIdentity::Method(MethodId("remote_inc".to_string())),
            vec![Value::Num(2.0)],
            1,
            None,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect("classref external call should resolve through semantic resolver");
        assert_eq!(value, Value::Num(3.0));
    }

    #[test]
    fn classref_external_method_without_resolver_remains_unresolved() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::Method(MethodId("sqrt".to_string())),
            vec![Value::Num(9.0)],
            1,
            None,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("classref external call should not fallback to builtin name resolution");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn method_member_call_rejects_identity_without_method_name() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::AnonymousFunction(runmat_hir::FunctionId(12)),
            vec![Value::Num(9.0)],
            1,
            None,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("anonymous identity should not be used for method/member call");
        assert_eq!(err.identifier(), Some("RunMat:MethodCallCalleeInvalid"));
    }

    #[test]
    fn method_member_call_rejects_imported_identity_with_identifier() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::Imported(runmat_hir::DefPath {
                package: runmat_hir::PackageName("Point".to_string()),
                module: runmat_hir::QualifiedName(vec![
                    runmat_hir::SymbolName("Point".to_string()),
                    runmat_hir::SymbolName("origin".to_string()),
                ]),
                item: vec![runmat_hir::DefPathSegment::Function(
                    runmat_hir::SymbolName("origin".to_string()),
                )],
            }),
            vec![Value::Num(9.0)],
            1,
            None,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("imported identity should not be used for method/member call");
        assert_eq!(err.identifier(), Some("RunMat:MethodCallCalleeInvalid"));
    }

    #[test]
    fn method_member_call_rejects_multisegment_external_identity_with_identifier() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::ExternalName(runmat_hir::QualifiedName(vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("remote".to_string()),
            ])),
            vec![Value::Num(9.0)],
            1,
            None,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("multi-segment external identity should not be used for method/member call");
        assert_eq!(err.identifier(), Some("RunMat:MethodCallCalleeInvalid"));
    }

    #[test]
    fn method_member_call_rejects_whitespace_method_identity_with_identifier() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::Method(MethodId("   ".to_string())),
            vec![Value::Num(9.0)],
            1,
            None,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("whitespace method identity should not be used for method/member call");
        assert_eq!(err.identifier(), Some("RunMat:MethodCallCalleeInvalid"));
    }

    #[test]
    fn method_member_call_rejects_whitespace_single_segment_external_identity_with_identifier() {
        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef("Point".to_string()),
            CallableIdentity::ExternalName(runmat_hir::QualifiedName(vec![
                runmat_hir::SymbolName("   ".to_string()),
            ])),
            vec![Value::Num(9.0)],
            1,
            None,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err(
            "whitespace single-segment external identity should not be used for method/member call",
        );
        assert_eq!(err.identifier(), Some("RunMat:MethodCallCalleeInvalid"));
    }

    #[test]
    fn classref_nonstatic_method_reports_identifier() {
        let class_name = "ClosureMethodNotStaticTest".to_string();
        let mut methods = HashMap::new();
        methods.insert(
            "inst".to_string(),
            runmat_runtime::class_registry::RuntimeMethod {
                name: "inst".to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "inst".to_string(),
                implicit_class_argument: None,
            },
        );
        runmat_runtime::class_registry::register_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: class_name.clone(),
                parent: None,
                properties: HashMap::new(),
                methods,
            },
        );

        let err = block_on(call_method_or_member_index_with_outputs(
            Value::ClassRef(class_name),
            CallableIdentity::Method(MethodId("inst".to_string())),
            vec![],
            1,
            None,
            CallableFallbackPolicy::ObjectDispatch,
        ))
        .expect_err("classref call to non-static method should fail");
        assert_eq!(err.identifier(), Some("RunMat:MethodNotStatic"));
    }

    #[test]
    fn load_method_unknown_static_method_reports_identifier() {
        let err = load_method_closure(
            Value::ClassRef("Point".to_string()),
            "definitely_missing_static_method".to_string(),
            None,
        )
        .expect_err("unknown static method should fail during method-handle load");
        assert_eq!(err.identifier(), Some("RunMat:UnknownStaticMethod"));
    }
}
