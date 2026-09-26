use super::*;

#[test]
fn repeated_anonymous_publication_changes_session_catalog_dependency() {
    runmat_gc::gc_test_context(|| {
        let mut session = crate::RunMatSession::with_options(false, false).unwrap();
        let source = "f = @(x) x + 1; y = f(2); assert(y == 3);";
        let first = session
            .compile_input(source)
            .unwrap()
            .bytecode
            .function_registry();
        let outcome = crate::execute_text_request_for_testing(&mut session, source).unwrap();
        assert!(outcome.error.is_none());
        let second = session
            .compile_input(source)
            .unwrap()
            .bytecode
            .function_registry();
        let anonymous = first
            .names
            .keys()
            .find(|name| name.starts_with("anonymous#"))
            .expect("authored closure is represented in the session registry");
        assert_ne!(
            first.resolve_name(anonymous),
            second.resolve_name(anonymous)
        );
        assert_ne!(
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
        assert!(!captured.is_satisfied_by(&cache.dependencies.snapshot_all()));
        // This characterizes the invalidation gate, not proof that the L03
        // witness depends on SessionCatalog or that stale code is safe to reuse.
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
