use std::rc::Rc;

use runmat_runtime::context::{RuntimePreferenceService, RuntimeUserPathService};

use super::RunMatSession;

impl RunMatSession {
    /// Replace the host-supplied application settings authority before first
    /// source resolution. Both preference and user-path ports receive the
    /// same owner; no project filesystem or implicit session fallback is used.
    pub fn install_application_settings<T>(&mut self, service: Rc<T>) -> anyhow::Result<()>
    where
        T: RuntimePreferenceService + RuntimeUserPathService + 'static,
    {
        if self.user_path_hydrated || self.is_executing {
            anyhow::bail!("application settings must be installed before first execution");
        }
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_preferences(service.clone())
            .with_user_path(service);
        self.runtime_context = self.runtime_context.clone().with_service_ports(ports);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::context::IsolatedMemoryPreferenceService;

    fn execute(session: &mut RunMatSession, source: &str) {
        let request = crate::abi::ExecutionRequest::for_source(
            crate::abi::SourceInput::Text {
                name: "<application-settings-test>".into(),
                text: source.into(),
            },
            session.compat_mode(),
            crate::abi::HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        let outcome = futures::executor::block_on(session.execute_request(request))
            .result
            .expect("public engine execution");
        assert!(
            outcome
                .diagnostics
                .iter()
                .all(|diagnostic| diagnostic.severity != crate::abi::DiagnosticSeverity::Error),
            "unexpected runtime diagnostics: {:?}",
            outcome.diagnostics
        );
    }

    #[test]
    fn isolated_authority_serves_both_ports_across_real_sessions_and_hydration() {
        let _serial = runmat_filesystem::provider_override_lock();
        let root = "/settings-example";
        let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
            runmat_filesystem::MemoryFsProvider::with_current_dir(root),
        ));
        runmat_filesystem::set_current_dir(root).expect("select fixture current directory");
        let shared = Rc::new(IsolatedMemoryPreferenceService::new());

        let mut first = RunMatSession::with_options(false, false).expect("first session");
        first.set_compat_mode(runmat_parser::CompatMode::RunMat);
        assert!(!first.user_path_hydrated);
        first.install_application_settings(shared.clone()).unwrap();
        assert!(first
            .runtime_context
            .service_ports()
            .preferences()
            .is_some());
        assert!(first.runtime_context.service_ports().user_path().is_some());
        execute(
            &mut first,
            "setpref('Example', 'Value', 7); userpath('relative/MiXeD');",
        );
        assert!(first.user_path_hydrated);
        assert!(first
            .install_application_settings(Rc::new(IsolatedMemoryPreferenceService::new()))
            .is_err());
        execute(&mut first, "assert(getpref('Example', 'Value') == 7); assert(strcmp(userpath(), 'relative/MiXeD'));");
        drop(first);

        let mut later = RunMatSession::with_options(false, false).expect("later session");
        later.set_compat_mode(runmat_parser::CompatMode::RunMat);
        later.install_application_settings(shared).unwrap();
        assert!(!later.user_path_hydrated);
        execute(&mut later, "assert(getpref('Example', 'Value') == 7); assert(strcmp(userpath(), 'relative/MiXeD'));");
        assert!(later.user_path_hydrated);
        let expected = std::path::Path::new(root).join("relative/MiXeD");
        assert!(later
            .runtime_context
            .search_path()
            .unwrap()
            .current_string()
            .starts_with(expected.to_str().unwrap()));
        drop(later);

        let mut independent =
            RunMatSession::with_options(false, false).expect("independent session");
        independent.set_compat_mode(runmat_parser::CompatMode::RunMat);
        independent
            .install_application_settings(Rc::new(IsolatedMemoryPreferenceService::new()))
            .unwrap();
        execute(&mut independent, "assert(ispref('Example', 'Value') == 0); assert(strcmp(userpath(), 'relative/MiXeD') == 0);");
        assert!(independent.user_path_hydrated);
    }
}
