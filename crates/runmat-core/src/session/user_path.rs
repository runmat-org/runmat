//! Await application user-path hydration exactly once before source resolution.

use super::*;
use runmat_runtime::context::UserPathServiceError;

impl RunMatSession {
    pub(super) async fn ensure_user_path_hydrated(&mut self) -> std::result::Result<(), RunError> {
        let search_path = self.runtime_context.search_path().cloned().ok_or_else(|| {
            RunError::Runtime(
                build_runtime_error("session has no ordered search-path owner")
                    .with_identifier("RunMat:RuntimeContext:CapabilityUnavailable")
                    .build(),
            )
        })?;
        if self.user_path_hydrated && search_path.user_authority_current() {
            return Ok(());
        }
        let rehydrating = self.user_path_hydrated;
        // A stale local interpretation must be rebound to the current
        // filesystem namespace. The original factory's environment and
        // installation layers remain captured at session creation.
        let fresh_authority = if rehydrating {
            Some(
                runmat_runtime::builtins::common::path_state::CapturedUserPathAuthority::capture()
                    .map_err(authority_error)?,
            )
        } else {
            None
        };
        let authority = fresh_authority
            .as_ref()
            .unwrap_or(&self.startup_user_path_authority);
        let runtime = self.runtime_context.clone();
        let service = runtime
            .service_ports()
            .user_path()
            .cloned()
            .ok_or_else(|| {
                RunError::Runtime(
                    build_runtime_error(
                        "source resolution requires an application user-path settings service",
                    )
                    .with_identifier("RunMat:RuntimeContext:CapabilityUnavailable")
                    .build(),
                )
            })?;
        authority.check_current().map_err(authority_error)?;
        let snapshot = service.read(runtime.clone()).await.map_err(service_error)?;
        authority.check_current().map_err(authority_error)?;
        let mut cooperation = runtime.cpu_cooperation();
        cooperation
            .checkpoint(0)
            .await
            .map_err(|source| RunError::Runtime(source.into_runtime_error()))?;
        let user = cooperation
            .await_cancellable(authority.resolve(&snapshot.setting))
            .await
            .map_err(|source| RunError::Runtime(source.into_runtime_error()))?
            .map_err(authority_error)?;
        authority.check_current().map_err(authority_error)?;
        if rehydrating {
            if !search_path.rehydrate_user(snapshot.revision, user.search_segment) {
                return Err(RunError::Runtime(
                    build_runtime_error(
                        "application user-path read returned an older revision than the session has observed",
                    )
                    .with_identifier("RunMat:userpath:StaleRevision")
                    .build(),
                ));
            }
        } else {
            let installation = self
                .startup_user_path_authority
                .installation_root()
                .map_err(authority_error)?;
            if !search_path.install_captured_factory(
                user,
                snapshot.revision,
                &self.startup_environment_path,
                installation,
            ) {
                return Err(RunError::Runtime(
                    build_runtime_error(
                        "application user-path read returned an older revision than the session has observed",
                    )
                    .with_identifier("RunMat:userpath:StaleRevision")
                    .build(),
                ));
            }
        }
        self.user_path_hydrated = true;
        Ok(())
    }
}

fn authority_error(source: std::io::Error) -> RunError {
    RunError::Runtime(
        build_runtime_error(format!("user-path startup authority failed: {source}"))
            .with_identifier("RunMat:userpath:FilesystemAuthority")
            .with_source(source)
            .build(),
    )
}

fn service_error(source: UserPathServiceError) -> RunError {
    match source {
        UserPathServiceError::Execution(source) => RunError::Runtime(source.into_runtime_error()),
        source => RunError::Runtime(
            build_runtime_error(format!("application user-path setting failed: {source}"))
                .with_identifier("RunMat:userpath:Storage")
                .with_source(source)
                .build(),
        ),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use runmat_runtime::context::IsolatedMemoryPreferenceService;
    use std::{rc::Rc, sync::Arc};

    fn install_settings(
        session: &mut RunMatSession,
        settings: Rc<IsolatedMemoryPreferenceService>,
    ) {
        let ports = session
            .runtime_context
            .service_ports()
            .clone()
            .with_user_path(settings);
        session.runtime_context = session.runtime_context.clone().with_service_ports(ports);
    }

    fn execute(session: &mut RunMatSession, source: &str) -> std::result::Result<(), RunError> {
        let request = crate::abi::ExecutionRequest::for_source(
            crate::abi::SourceInput::Text {
                name: "<userpath-test>".into(),
                text: source.into(),
            },
            session.compat_mode(),
            crate::abi::HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        futures::executor::block_on(session.execute_request(request))
            .result
            .map(|_| ())
    }

    #[test]
    fn later_core_session_hydrates_exact_setting_before_execution_and_restores_its_factory() {
        let _serial = runmat_filesystem::provider_override_lock();
        let (first, second) = ("/selected", "/later");
        let _first = runmat_filesystem::replace_provider(Arc::new(
            runmat_filesystem::MemoryFsProvider::with_current_dir(first),
        ));
        let settings = Rc::new(IsolatedMemoryPreferenceService::new());
        let mut first_session = RunMatSession::with_options(false, false).expect("first session");
        first_session.set_compat_mode(runmat_parser::CompatMode::RunMat);
        install_settings(&mut first_session, settings.clone());
        execute(&mut first_session, "userpath('relative/MiXeD');")
            .expect("first session commits the setting");
        drop(first_session);

        let _second = runmat_filesystem::replace_provider(Arc::new(
            runmat_filesystem::MemoryFsProvider::with_current_dir(second),
        ));
        runmat_filesystem::set_current_dir(second)
            .expect("select second virtual current directory");
        let mut later_session = RunMatSession::with_options(false, false).expect("later session");
        install_settings(&mut later_session, settings);
        execute(
            &mut later_session,
            "assert(strcmp(userpath(), 'relative/MiXeD'));",
        )
        .expect("hydration precedes source execution");
        let search = later_session
            .runtime_context
            .search_path()
            .expect("search path");
        let expected = std::path::Path::new(second)
            .join("relative/MiXeD")
            .to_str()
            .expect("test path")
            .to_owned();
        assert!(search.current_string().starts_with(&expected));
        execute(
            &mut later_session,
            "path('temporary_manual_entry'); restoredefaultpath();",
        )
        .expect("restore captured factory");
        let restored = later_session
            .runtime_context
            .search_path()
            .expect("search path")
            .current_string();
        assert!(restored.starts_with(&expected));
        assert!(!restored.contains("temporary_manual_entry"));
    }
}
