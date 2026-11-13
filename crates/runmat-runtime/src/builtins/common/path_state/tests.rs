use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn join_and_split_round_trip() {
    let parts = vec!["/tmp/a".to_string(), "/tmp/b".to_string()];
    let joined = join_parts(&parts);
    assert_eq!(split_segments(&joined), parts);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn active_runtime_context_owns_path_and_generation() {
    let first = Arc::new(SearchPath::new("first".to_string()));
    let first_context = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_search_path(Arc::clone(&first));
    let _first_context = crate::context::RuntimeContextGuard::enter(first_context);
    assert_eq!(current_path_string(), "first");
    set_path_string("first").expect("test search path update");
    assert_eq!(first.generation(), 0, "no-op replacement is not a mutation");
    let updated = format!("first{PATH_LIST_SEPARATOR}added");
    set_path_string(&updated).expect("test search path update");
    assert_eq!(current_path_string(), updated);
    assert_eq!(first.generation(), 1);

    let second = Arc::new(SearchPath::new("second".to_string()));
    {
        let second_context = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_search_path(second);
        let _second_context = crate::context::RuntimeContextGuard::enter(second_context);
        assert_eq!(current_path_string(), "second");
    }
    assert_eq!(current_path_string(), updated);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn refresh_advances_revision_without_changing_path_text() {
    let path = SearchPath::new("functions".to_string());
    assert_eq!(path.refresh_generation().unwrap(), 1);
    assert_eq!(path.current_string(), "functions");
    assert_eq!(path.generation(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn user_segment_is_replaced_by_origin_not_equal_manual_text() {
    let path = SearchPath::new(String::new());
    path.install_factory_at_revision(
        vec![
            PathEntry::new("user".into(), PathOrigin::User),
            PathEntry::new("env".into(), PathOrigin::Environment),
            PathEntry::new("install".into(), PathOrigin::Installation),
        ],
        0,
    );
    let mut edited = path.snapshot();
    edited.push(PathEntry::new("user".into(), PathOrigin::Manual));
    path.replace_entries(edited);
    assert!(path.replace_user_if_newer(1, Some("new-user".into())));
    assert_eq!(
        path.current_string(),
        join_parts(&[
            "new-user".into(),
            "env".into(),
            "install".into(),
            "user".into(),
        ])
    );
    path.restore_factory();
    assert_eq!(
        path.current_string(),
        join_parts(&["new-user".into(), "env".into(), "install".into()])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn full_path_replacement_explicitly_discards_user_provenance() {
    let path = SearchPath::new(String::new());
    path.install_factory_at_revision(vec![PathEntry::new("user".into(), PathOrigin::User)], 0);
    path.replace("user");
    assert_eq!(path.snapshot()[0].origin, PathOrigin::Manual);
    assert!(path.replace_user_if_newer(1, Some("new-user".into())));
    assert_eq!(
        path.current_string(),
        join_parts(&["new-user".into(), "user".into()])
    );
    assert!(
        !path.replace_user_if_newer(1, Some("stale-user".into())),
        "an older application setting cannot overwrite a committed revision"
    );
    assert_eq!(
        path.current_string(),
        join_parts(&["new-user".into(), "user".into()])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn stale_equal_revision_rebinds_only_user_entry_and_retains_manual_path() {
    let path = SearchPath::new(String::new());
    path.install_factory_at_revision(
        vec![
            PathEntry::new("old/user".into(), PathOrigin::User),
            PathEntry::new("environment".into(), PathOrigin::Environment),
        ],
        7,
    );
    path.replace_entries(vec![
        PathEntry::new("old/user".into(), PathOrigin::User),
        PathEntry::new("manual".into(), PathOrigin::Manual),
    ]);
    path.mark_user_authority_stale();
    assert!(!path.user_authority_current());
    assert!(path.rehydrate_user(7, Some("new/user".into())));
    assert!(path.user_authority_current());
    assert_eq!(
        path.snapshot(),
        vec![
            PathEntry::new("new/user".into(), PathOrigin::User),
            PathEntry::new("manual".into(), PathOrigin::Manual),
        ]
    );
    path.restore_factory();
    assert_eq!(
        path.snapshot(),
        vec![
            PathEntry::new("new/user".into(), PathOrigin::User),
            PathEntry::new("environment".into(), PathOrigin::Environment),
        ]
    );
    path.mark_user_authority_stale();
    assert!(!path.rehydrate_user(6, Some("older".into())));
    assert!(!path.user_authority_current());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn initial_hydration_retains_explicit_manual_entries_without_making_them_factory() {
    let path = SearchPath::new("host-configured".into());
    assert!(path.install_captured_factory(
        ResolvedUserPath {
            presentation: "user".into(),
            search_segment: Some("user".into()),
        },
        5,
        "environment",
        Some("installation".into()),
    ));
    assert_eq!(
        path.current_string(),
        join_parts(&[
            "host-configured".into(),
            "user".into(),
            "environment".into(),
            "installation".into(),
        ])
    );
    path.restore_factory();
    assert_eq!(
        path.current_string(),
        join_parts(&["user".into(), "environment".into(), "installation".into()])
    );
    path.mark_user_authority_stale();
    assert!(!path.replace_user_if_newer(4, Some("old".into())));
    assert!(!path.user_authority_current());
    assert!(!path.install_captured_factory(
        ResolvedUserPath {
            presentation: "old".into(),
            search_segment: Some("old".into()),
        },
        4,
        "environment",
        None,
    ));
    assert!(!path.user_authority_current());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn stale_user_authority_rejects_source_search_before_touching_filesystem() {
    let never_published = SearchPath::new("manual".into());
    never_published.mark_user_authority_stale();
    assert!(!never_published.source_authority_current());
    let path = Arc::new(SearchPath::new("manual".into()));
    path.install_factory_at_revision(vec![PathEntry::new("user".into(), PathOrigin::User)], 1);
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_search_path(path.clone());
    let _active = crate::context::RuntimeContextGuard::enter(runtime);
    assert!(source_authority_current());
    path.mark_user_authority_stale();
    assert!(!source_authority_current());
    let error = crate::builtins::common::path_search::search_directories("lookup")
        .expect_err("stale user path must stop source lookup");
    assert!(error.to_string().contains("requires rehydration"));
    let error = error.into_runtime_error("lookup");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(matches!(
        error
            .source
            .as_ref()
            .unwrap()
            .downcast_ref::<crate::execution::ExecutionServiceError>(),
        Some(crate::execution::ExecutionServiceError::Infrastructure(_))
    ));
}
