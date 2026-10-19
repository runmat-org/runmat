use super::*;

#[test]
fn repeated_anonymous_publication_preserves_session_catalog_dependency() {
    runmat_gc::gc_test_context(|| {
        let mut session = crate::RunMatSession::with_options(false, false).unwrap();
        let source = "f = @(x) x + 1; y = f(2); assert(y == 3);";
        let outcome = crate::execute_text_request_for_testing(&mut session, source).unwrap();
        assert!(outcome.error.is_none());
        let first = session.published_function_registry_for_testing();
        let outcome = crate::execute_text_request_for_testing(&mut session, source).unwrap();
        assert!(outcome.error.is_none());
        let second = session.published_function_registry_for_testing();
        let anonymous = first
            .names
            .keys()
            .find(|name| name.starts_with("anonymous#"))
            .expect("authored closure is represented in the session registry");
        assert_eq!(
            first.resolve_name(anonymous),
            second.resolve_name(anonymous)
        );
        assert_eq!(
            session_catalog_revision(&first).unwrap(),
            session_catalog_revision(&second).unwrap()
        );
        let mut cache = GenericNativeCache::default();
        cache
            .publish_session_registry(&runmat_vm::FunctionRegistry::default(), &first)
            .unwrap();
        let captured = cache
            .dependencies
            .snapshot([&DependencyKey::SessionCatalog]);
        cache.publish_session_registry(&first, &second).unwrap();
        assert!(captured.is_satisfied_by(&cache.dependencies.snapshot_all()));
    });
}

#[test]
fn completion_diagnostics_preserve_success_and_distinguish_failure_paths() {
    let key = CompilationKey {
        entry: EntryKey("opaque entry".into()),
        tier: CompilationTier::Generic,
    };
    assert_eq!(
        diagnostics::completed(&key, Some(Ok::<_, String>(7))),
        Some(7)
    );
    assert_eq!(
        diagnostics::completed::<i32>(&key, Some(Err("compiler stage failed".into()))),
        None
    );
    assert_eq!(diagnostics::completed::<i32>(&key, None), None);
}
