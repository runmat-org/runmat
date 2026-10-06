use super::*;

struct RestoredCwd(PathBuf);

impl Drop for RestoredCwd {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.0).expect("restore original current directory");
    }
}

fn execute_helper(session: &mut RunMatSession) -> crate::abi::RuntimeDiagnostic {
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "<dynamic-source-product>".into(),
            text: "helper();".into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let outcome = futures::executor::block_on(session.execute_request(request))
        .result
        .expect("public source invocation returns an outcome");
    outcome
        .diagnostics
        .into_iter()
        .find(|item| item.severity == crate::abi::DiagnosticSeverity::Error)
        .expect("the private companion reports its runtime failure")
}

#[test]
fn dynamic_cache_hit_preserves_private_companion_source_attribution() {
    let _serial = runmat_filesystem::provider_override_lock();
    let temporary = tempfile::TempDir::new().expect("isolated source root");
    std::fs::create_dir(temporary.path().join("private")).expect("private source directory");
    std::fs::write(
        temporary.path().join("helper.m"),
        "function y = helper()\n y = secret();\nend\n",
    )
    .expect("dynamic entry source");
    std::fs::write(
        temporary.path().join("private/secret.m"),
        "function y = secret()\n error('private source witness');\nend\n",
    )
    .expect("private companion source");
    let _cwd = RestoredCwd(std::env::current_dir().expect("capture current directory"));
    std::env::set_current_dir(temporary.path()).expect("select source root");

    let mut session = RunMatSession::with_options(false, false).expect("interpreter session");
    let first = execute_helper(&mut session);
    assert!(
        first.message.contains("private source witness"),
        "{first:?}"
    );
    assert!(
        first
            .callstack
            .iter()
            .any(|frame| frame.contains("private/secret.m")),
        "private source must own the failure: {first:?}"
    );
    let (registry, source_map) = {
        let cache = session.dynamic_function_cache.lock().expect("cache lock");
        assert_eq!(cache.len(), 1, "the entry was dynamically resolved");
        let entry = cache.values().next().expect("resolved helper");
        assert!(entry.source_map.entries().iter().any(|record| {
            record.relative_path.ends_with("private/secret.m")
                && record.text.contains("private source witness")
        }));
        (Arc::clone(&entry.registry), Arc::clone(&entry.source_map))
    };

    let second = execute_helper(&mut session);
    assert_eq!(second.code, first.code);
    assert_eq!(second.message, first.message);
    assert_eq!(second.callstack, first.callstack);
    let cache = session.dynamic_function_cache.lock().expect("cache lock");
    assert_eq!(cache.len(), 1);
    let entry = cache.values().next().expect("same dynamic entry");
    assert!(
        Arc::ptr_eq(&entry.registry, &registry),
        "registry cache hit"
    );
    assert!(
        Arc::ptr_eq(&entry.source_map, &source_map),
        "source-map cache hit"
    );
}
