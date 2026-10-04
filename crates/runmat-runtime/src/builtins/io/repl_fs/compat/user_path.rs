//! Public userpath controls and immediate ordered search-path publication.

use runmat_value::Value;

use crate::builtins::common::path_state::CapturedUserPathAuthority;
use crate::context::{
    PreferenceServiceError, RuntimeContext, UserPathMutation, UserPathServiceError, UserPathSetting,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

enum Request {
    Read,
    Change(UserPathMutation),
}

pub(super) async fn run(runtime: Option<RuntimeContext>, args: Vec<Value>) -> BuiltinResult<Value> {
    let request = decode(args)?;
    if let Request::Change(UserPathMutation::Set(ref text)) = request {
        CapturedUserPathAuthority::validate_custom_text(text).map_err(path_error)?;
    }
    let runtime = runtime.ok_or_else(unavailable)?;
    let service = runtime
        .service_ports()
        .user_path()
        .cloned()
        .ok_or_else(unavailable)?;
    let search = runtime.search_path().cloned().ok_or_else(unavailable)?;
    let authority = CapturedUserPathAuthority::capture().map_err(authority_error)?;
    // A known invalid path must fail before persistent storage is touched.
    if let Request::Change(UserPathMutation::Set(ref text)) = request {
        authority.resolve_custom(text).map_err(path_error)?;
    }
    let resolution_runtime = runtime.clone();
    let changed = matches!(&request, Request::Change(_));
    let explicit_reset = matches!(&request, Request::Change(UserPathMutation::Reset));
    let snapshot = match request {
        Request::Read => service.read(runtime).await.map_err(service_error)?,
        Request::Change(mutation) => {
            // This is the last local authority check before the application
            // store authorizes its independent durable transaction.
            authority.check_current().map_err(authority_error)?;
            match service.commit(runtime, mutation).await {
                Ok(snapshot) => snapshot,
                Err(source) => {
                    // A definitive rejection leaves the old path valid; only
                    // an indeterminate authorized commit can invalidate it.
                    if matches!(
                        &source,
                        UserPathServiceError::Storage(source)
                            if matches!(source.as_ref(), PreferenceServiceError::CommitIndeterminate(_))
                    ) {
                        search.mark_user_authority_stale();
                    }
                    return Err(service_error(source));
                }
            }
        }
    };
    if let Err(source) = authority.check_current() {
        if changed {
            search.mark_user_authority_stale();
            return Err(committed_but_local_path_unavailable(
                snapshot.revision,
                source,
            ));
        }
        return Err(authority_error(source));
    }
    if explicit_reset && !matches!(&snapshot.setting, UserPathSetting::Default) {
        search.mark_user_authority_stale();
        return Err(build_runtime_error(
            "userpath: application settings service returned a non-default value for reset",
        )
        .with_builtin("userpath")
        .with_identifier("RunMat:userpath:Storage")
        .build());
    }
    let mut cooperation = resolution_runtime.cpu_cooperation();
    if let Err(source) = cooperation.checkpoint(0).await {
        if changed {
            search.mark_user_authority_stale();
            return Err(committed_but_local_cancellation(snapshot.revision, source));
        }
        return Err(source.into_runtime_error());
    }
    let resolution = async {
        if explicit_reset {
            authority.resolve_after_reset().await
        } else {
            authority.resolve(&snapshot.setting).await
        }
    };
    let resolved = match cooperation.await_cancellable(resolution).await {
        Ok(Ok(resolved)) => resolved,
        Ok(Err(source)) if changed => {
            search.mark_user_authority_stale();
            return Err(committed_but_local_path_unavailable(
                snapshot.revision,
                source,
            ));
        }
        Ok(Err(source)) => return Err(path_error(source)),
        Err(source) if changed => {
            search.mark_user_authority_stale();
            return Err(committed_but_local_cancellation(snapshot.revision, source));
        }
        Err(source) => return Err(source.into_runtime_error()),
    };
    if let Err(source) = authority.check_current() {
        if changed {
            search.mark_user_authority_stale();
            return Err(committed_but_local_path_unavailable(
                snapshot.revision,
                source,
            ));
        }
        return Err(authority_error(source));
    }
    if search.user_authority_current() {
        search.replace_user_if_newer(snapshot.revision, resolved.search_segment);
    } else if !search.rehydrate_user(snapshot.revision, resolved.search_segment) {
        return Err(build_runtime_error(
            "userpath: application setting read returned an older revision",
        )
        .with_builtin("userpath")
        .with_identifier("RunMat:userpath:StaleRevision")
        .build());
    }
    Ok(super::char_value(&resolved.presentation))
}

pub(super) fn restore(runtime: Option<RuntimeContext>, args: Vec<Value>) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(
            build_runtime_error("restoredefaultpath: too many output arguments")
                .with_builtin("restoredefaultpath")
                .with_identifier("RunMat:restoredefaultpath:TooManyOutputs")
                .build(),
        );
    }
    if !args.is_empty() {
        return Err(
            build_runtime_error("restoredefaultpath: too many input arguments")
                .with_builtin("restoredefaultpath")
                .with_identifier("RunMat:restoredefaultpath:InvalidCall")
                .build(),
        );
    }
    let runtime = runtime.ok_or_else(restore_unavailable)?;
    let search = runtime.search_path().ok_or_else(restore_unavailable)?;
    if !search.user_authority_current() {
        return Err(build_runtime_error(
            "restoredefaultpath: user-path setting must be rehydrated before source resolution",
        )
        .with_builtin("restoredefaultpath")
        .with_identifier("RunMat:restoredefaultpath:StaleUserPath")
        .build());
    }
    search.restore_factory();
    Ok(super::char_value(&search.current_string()))
}

fn decode(args: Vec<Value>) -> BuiltinResult<Request> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(build_runtime_error("userpath: too many output arguments")
            .with_builtin("userpath")
            .with_identifier("RunMat:userpath:TooManyOutputs")
            .build());
    }
    let mut args = args.into_iter();
    let Some(value) = args.next() else {
        return Ok(Request::Read);
    };
    if args.next().is_some() {
        return Err(build_runtime_error("userpath: too many input arguments")
            .with_builtin("userpath")
            .with_identifier("RunMat:userpath:InvalidCall")
            .build());
    }
    let text =
        crate::builtins::common::validation::value_text_scalar_string(&value).ok_or_else(|| {
            build_runtime_error("userpath: path or command must be host text")
                .with_builtin("userpath")
                .with_identifier("RunMat:userpath:InvalidPath")
                .build()
        })?;
    let mutation = if text.eq_ignore_ascii_case("reset") {
        UserPathMutation::Reset
    } else if text.eq_ignore_ascii_case("clear") {
        UserPathMutation::Clear
    } else {
        UserPathMutation::Set(text)
    };
    Ok(Request::Change(mutation))
}

fn unavailable() -> RuntimeError {
    build_runtime_error("userpath: application settings service is unavailable")
        .with_builtin("userpath")
        .with_identifier("RunMat:RuntimeContext:CapabilityUnavailable")
        .build()
}

fn restore_unavailable() -> RuntimeError {
    build_runtime_error("restoredefaultpath: ordered search-path service is unavailable")
        .with_builtin("restoredefaultpath")
        .with_identifier("RunMat:RuntimeContext:CapabilityUnavailable")
        .build()
}

fn authority_error(source: std::io::Error) -> RuntimeError {
    build_runtime_error(format!("userpath: filesystem authority failed: {source}"))
        .with_builtin("userpath")
        .with_identifier("RunMat:userpath:FilesystemAuthority")
        .with_source(source)
        .build()
}

fn path_error(source: std::io::Error) -> RuntimeError {
    if source.kind() != std::io::ErrorKind::InvalidInput {
        return authority_error(source);
    }
    build_runtime_error(format!("userpath: invalid path: {source}"))
        .with_builtin("userpath")
        .with_identifier("RunMat:userpath:InvalidPath")
        .with_source(source)
        .build()
}

fn committed_but_local_path_unavailable(revision: u64, source: std::io::Error) -> RuntimeError {
    build_runtime_error(format!(
        "userpath: application setting committed at revision {revision}, but local path authority is unavailable: {source}"
    ))
    .with_builtin("userpath")
    .with_identifier("RunMat:userpath:CommittedButLocalPathUnavailable")
    .with_source(source)
    .build()
}

fn committed_but_local_cancellation(
    revision: u64,
    source: crate::execution::ExecutionServiceError,
) -> RuntimeError {
    build_runtime_error(format!(
        "userpath: application setting committed at revision {revision}, but local publication was cancelled"
    ))
    .with_builtin("userpath")
    .with_identifier("RunMat:userpath:CommittedButLocalPathUnavailable")
    .with_source(source)
    .build()
}

fn service_error(source: UserPathServiceError) -> RuntimeError {
    match source {
        UserPathServiceError::Execution(source) => source.into_runtime_error(),
        source => build_runtime_error(format!("userpath: application setting failed: {source}"))
            .with_builtin("userpath")
            .with_identifier("RunMat:userpath:Storage")
            .with_source(source)
            .build(),
    }
}

#[cfg(test)]
mod tests {
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
        let _serial = fs::provider_override_lock();
        #[cfg(windows)]
        let (first, second) = ("C:\\first", "C:\\second");
        #[cfg(not(windows))]
        let (first, second) = ("/first", "/second");
        let _first = fs::replace_provider(Arc::new(fs::MemoryFsProvider::with_current_dir(first)));
        let settings = Rc::new(IsolatedMemoryPreferenceService::new());
        let first_path = Arc::new(SearchPath::new(String::new()));
        let first_context = context(first_path.clone(), settings.clone());
        let value = futures::executor::block_on(run(
            Some(first_context),
            vec![super::super::char_value("relative/MiXeD")],
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

        let _second =
            fs::replace_provider(Arc::new(fs::MemoryFsProvider::with_current_dir(second)));
        fs::set_current_dir(second).expect("select second virtual current directory");
        let later_path = Arc::new(SearchPath::new(String::new()));
        let later_context = context(later_path.clone(), settings);
        let query = futures::executor::block_on(run(Some(later_context.clone()), vec![]))
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
        let restored = restore(Some(later_context), vec![]).expect("restore captured layers");
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
        let _serial = fs::provider_override_lock();
        #[cfg(windows)]
        let cwd = "C:\\selected";
        #[cfg(not(windows))]
        let cwd = "/selected";
        let _selected = fs::replace_provider(Arc::new(fs::MemoryFsProvider::with_current_dir(cwd)));
        let search = Arc::new(SearchPath::new(String::new()));
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_search_path(search.clone())
                .with_service_ports(
                    RuntimeServicePorts::default().with_user_path(Rc::new(UncertainCommit)),
                );
        let error = futures::executor::block_on(run(
            Some(runtime.clone()),
            vec![super::super::char_value("relative/path")],
        ))
        .expect_err("indeterminate storage outcome");
        assert_eq!(error.identifier(), Some("RunMat:userpath:Storage"));
        assert!(!search.user_authority_current());
        let _active = crate::context::RuntimeContextGuard::enter(runtime);
        assert!(
            crate::builtins::common::path_search::search_directories("lookup")
                .expect_err("stale setting must block source lookup")
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
        let error =
            futures::executor::block_on(run(None, vec![super::super::char_value(&invalid)]))
                .expect_err("one user path cannot contain path-list separators");
        assert_eq!(error.identifier(), Some("RunMat:userpath:InvalidPath"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn definitive_rejection_preserves_current_user_path_authority() {
        let _serial = fs::provider_override_lock();
        #[cfg(windows)]
        let cwd = "C:\\selected";
        #[cfg(not(windows))]
        let cwd = "/selected";
        let _selected = fs::replace_provider(Arc::new(fs::MemoryFsProvider::with_current_dir(cwd)));
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
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_search_path(search.clone())
                .with_service_ports(
                    RuntimeServicePorts::default().with_user_path(Rc::new(RejectedCommit)),
                );
        let error = futures::executor::block_on(run(
            Some(runtime.clone()),
            vec![super::super::char_value("new")],
        ))
        .expect_err("definitive rejection");
        assert_eq!(error.identifier(), Some("RunMat:userpath:Storage"));
        assert!(search.user_authority_current());
        assert_eq!(search.current_string(), "old");
        let _active = crate::context::RuntimeContextGuard::enter(runtime);
        assert!(crate::builtins::common::path_search::search_directories("lookup").is_ok());
    }
}
