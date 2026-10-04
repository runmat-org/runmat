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

    #[test]
    fn one_isolated_authority_replaces_both_ports_before_hydration_only() {
        runmat_gc::gc_test_context(|| {
            let mut session = RunMatSession::with_options(false, false).unwrap();
            let service = Rc::new(IsolatedMemoryPreferenceService::new());
            session.install_application_settings(service).unwrap();
            assert!(session
                .runtime_context
                .service_ports()
                .preferences()
                .is_some());
            assert!(session
                .runtime_context
                .service_ports()
                .user_path()
                .is_some());
            session.user_path_hydrated = true;
            assert!(session
                .install_application_settings(Rc::new(IsolatedMemoryPreferenceService::new()))
                .is_err());
        });
    }
}
