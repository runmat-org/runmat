use super::*;
use crate::builtins::common::path_state::SearchPath;
use crate::context::{
    IsolatedMemoryPreferenceService, RuntimeServiceFuture, RuntimeServicePorts,
    RuntimeUserPathService, UserPathSnapshot,
};
use runmat_filesystem as fs;
use std::{rc::Rc, sync::Arc};

fn context(
    search: Arc<SearchPath>,
    settings: Rc<IsolatedMemoryPreferenceService>,
) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_search_path(search)
        .with_service_ports(RuntimeServicePorts::default().with_user_path(settings))
}

fn text(value: &Value) -> String {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .expect("userpath produces a character row")
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setting_survives_a_later_context_and_restoration_uses_captured_layers() {
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = fs::provider_override_lock();
    let (first, second) = ("/first", "/second");
    let _first = fs::replace_provider(Arc::new(fs::MemoryFsProvider::with_current_dir(first)));
    let settings = Rc::new(IsolatedMemoryPreferenceService::new());
    let first_path = Arc::new(SearchPath::new(String::new()));
    let first_context = context(first_path.clone(), settings.clone());
    let value = futures::executor::block_on(run(
        Some(first_context),
        crate::builtins::common::test_support::builtin_arguments(vec![super::super::char_value(
            "relative/MiXeD",
        )]),
    ))
    .expect("commit setting");
    assert_eq!(text(&value), "relative/MiXeD");
    assert_eq!(
        first_path.current_string(),
        std::path::Path::new(first)
            .join("relative/MiXeD")
            .to_str()
            .expect("test path")
    );

    let _second = fs::replace_provider(Arc::new(fs::MemoryFsProvider::with_current_dir(second)));
    fs::set_current_dir(second).expect("select second virtual current directory");
    let later_path = Arc::new(SearchPath::new(String::new()));
    let later_context = context(later_path.clone(), settings);
    let query = futures::executor::block_on(run(
        Some(later_context.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("later context reads application setting");
    assert_eq!(text(&query), "relative/MiXeD");
    let later_segment = std::path::Path::new(second)
        .join("relative/MiXeD")
        .to_str()
        .expect("test path")
        .to_owned();
    assert_eq!(later_path.current_string(), later_segment);
    later_path.install_captured_factory(
        crate::builtins::common::path_state::ResolvedUserPath {
            presentation: "relative/MiXeD".into(),
            search_segment: Some(later_segment.clone()),
        },
        1,
        "environment",
        Some("installation".into()),
    );
    later_path.replace_entries(vec![crate::builtins::common::path_state::PathEntry::new(
        "manual".into(),
        crate::builtins::common::path_state::PathOrigin::Manual,
    )]);
    let restored = restore(
        Some(later_context),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("restore captured layers");
    assert_eq!(
        text(&restored),
        format!(
            "{later_segment}{}environment{}installation",
            crate::builtins::common::path_state::PATH_LIST_SEPARATOR,
            crate::builtins::common::path_state::PATH_LIST_SEPARATOR
        )
    );
}

struct UncertainCommit;

struct RejectedCommit;

impl RuntimeUserPathService for UncertainCommit {
    fn read(
        &self,
        _runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        Box::pin(async {
            Ok(UserPathSnapshot {
                setting: UserPathSetting::Default,
                revision: 0,
            })
        })
    }

    fn commit(
        &self,
        _runtime: RuntimeContext,
        _mutation: UserPathMutation,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        Box::pin(async {
            Err(UserPathServiceError::Storage(Box::new(
                PreferenceServiceError::CommitIndeterminate(
                    build_runtime_error("commit outcome lost").build(),
                ),
            )))
        })
    }
}

impl RuntimeUserPathService for RejectedCommit {
    fn read(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        UncertainCommit.read(runtime)
    }

    fn commit(
        &self,
        _runtime: RuntimeContext,
        _mutation: UserPathMutation,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        Box::pin(async {
            Err(UserPathServiceError::Storage(Box::new(
                PreferenceServiceError::InvalidTransaction("rejected before commit".into()),
            )))
        })
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn uncertain_commit_invalidates_source_search_until_rehydration() {
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = fs::provider_override_lock();
    let cwd = "/selected";
    let _selected = fs::replace_provider(Arc::new(fs::MemoryFsProvider::with_current_dir(cwd)));
    let changed = "/new";
    let search = Arc::new(SearchPath::new(String::new()));
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_search_path(search.clone())
        .with_service_ports(
            RuntimeServicePorts::default().with_user_path(Rc::new(UncertainCommit)),
        );
    let error = futures::executor::block_on(run(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![super::super::char_value(
            changed,
        )]),
    ))
    .expect_err("indeterminate storage outcome");
    assert_eq!(error.identifier(), Some("RunMat:userpath:Storage"));
    assert!(!search.user_authority_current());
    let _active = crate::context::RuntimeContextGuard::enter(runtime);
    assert!(
        crate::builtins::common::path_search::search_directories("lookup")
            .expect_err("stale setting must block source lookup")
            .to_string()
            .contains("requires rehydration")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_path_list_rejects_before_context_or_provider_capture() {
    let invalid = format!(
        "first{}second",
        crate::builtins::common::path_state::PATH_LIST_SEPARATOR
    );
    let error = futures::executor::block_on(run(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![super::super::char_value(
            &invalid,
        )]),
    ))
    .expect_err("one user path cannot contain path-list separators");
    assert_eq!(error.identifier(), Some("RunMat:userpath:InvalidPath"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn definitive_rejection_preserves_current_user_path_authority() {
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = fs::provider_override_lock();
    let cwd = "/selected";
    let _selected = fs::replace_provider(Arc::new(fs::MemoryFsProvider::with_current_dir(cwd)));
    let changed = "/new";
    let search = Arc::new(SearchPath::new(String::new()));
    search.install_captured_factory(
        crate::builtins::common::path_state::ResolvedUserPath {
            presentation: "old".into(),
            search_segment: Some("old".into()),
        },
        1,
        "",
        None,
    );
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_search_path(search.clone())
        .with_service_ports(RuntimeServicePorts::default().with_user_path(Rc::new(RejectedCommit)));
    let error = futures::executor::block_on(run(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![super::super::char_value(
            changed,
        )]),
    ))
    .expect_err("definitive rejection");
    assert_eq!(error.identifier(), Some("RunMat:userpath:Storage"));
    assert!(search.user_authority_current());
    assert_eq!(search.current_string(), "old");
    let _active = crate::context::RuntimeContextGuard::enter(runtime);
    assert!(crate::builtins::common::path_search::search_directories("lookup").is_ok());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn partial_path_is_gated_before_filesystem_or_application_settings() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let rejected = futures::executor::block_on(run(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![super::super::char_value(
            "relative/toolbox",
        )]),
    ))
    .expect_err("R2026a rejects a partial path before context capture");
    assert_eq!(
        rejected.identifier(),
        USERPATH_PARTIAL_PATH_EXTENSION.error_identifier
    );
    #[cfg(windows)]
    let absolute = "C:\\installed\\toolbox";
    #[cfg(not(windows))]
    let absolute = "/installed/toolbox";
    let absolute_result = futures::executor::block_on(run(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![super::super::char_value(
            absolute,
        )]),
    ))
    .expect_err("absolute paths reach the missing service");
    assert_eq!(
        absolute_result.identifier(),
        Some("RunMat:RuntimeContext:CapabilityUnavailable")
    );
    drop(_strict);
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    let admitted = futures::executor::block_on(run(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![super::super::char_value(
            "relative/toolbox",
        )]),
    ))
    .expect_err("RunMat mode admits the control before service lookup");
    assert_eq!(
        admitted.identifier(),
        Some("RunMat:RuntimeContext:CapabilityUnavailable")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn restore_return_is_extension_but_statement_is_admitted_before_mutation() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let search = Arc::new(SearchPath::new(String::new()));
    search.replace_entries(vec![crate::builtins::common::path_state::PathEntry::new(
        "manual".into(),
        crate::builtins::common::path_state::PathOrigin::Manual,
    )]);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_search_path(search.clone());
    {
        let _output = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let rejected = restore(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("MATLAB-compatible form has no requested result");
        assert_eq!(
            rejected.identifier(),
            RESTOREDEFAULTPATH_RETURN_EXTENSION.error_identifier
        );
        assert_eq!(search.current_string(), "manual");
    }
    {
        let _output = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let rejected = restore(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("a statement cannot restore an uncaptured user-path factory");
        assert_eq!(
            rejected.identifier(),
            Some("RunMat:restoredefaultpath:StaleUserPath")
        );
        assert_eq!(search.current_string(), "manual");
    }
    assert!(search.install_captured_factory(
        crate::builtins::common::path_state::ResolvedUserPath {
            presentation: String::new(),
            search_segment: None,
        },
        0,
        "factory",
        None,
    ));
    assert!(search.user_authority_current());
    search.replace_entries(vec![crate::builtins::common::path_state::PathEntry::new(
        "manual".into(),
        crate::builtins::common::path_state::PathOrigin::Manual,
    )]);
    {
        let _output = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        restore(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect("statement restores the path");
        assert_eq!(search.current_string(), "factory");
    }
    drop(_strict);
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    let _output = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let restored = restore(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("RunMat mode may request the restored path");
    assert_eq!(text(&restored), "factory");
}
