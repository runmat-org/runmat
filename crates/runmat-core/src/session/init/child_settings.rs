use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};

/// A test child keeps its own execution, workspace, cancellation, search-path, and
/// task owners while using the exact embedding application's settings ports.
/// Missing caller authority stays missing; it never triggers discovery in a
/// different host account or an anonymous browser namespace.
pub(super) fn inherit_caller_settings(
    mut child: RuntimeServicePorts,
    caller: &RuntimeContext,
) -> RuntimeServicePorts {
    let ports = caller.service_ports();
    if let Some(preferences) = ports.preferences() {
        child = child.with_preferences(preferences.clone());
    }
    if let Some(user_path) = ports.user_path() {
        child = child.with_user_path(user_path.clone());
    }
    child
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::context::IsolatedMemoryPreferenceService;
    use runmat_runtime::execution::RuntimeExecutionService;
    use std::rc::Rc;

    #[test]
    fn inherits_only_exact_application_ports_and_preserves_absence() {
        let bare = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let child = inherit_caller_settings(RuntimeServicePorts::default(), &bare);
        assert!(child.preferences().is_none());
        assert!(child.user_path().is_none());

        let settings = Rc::new(IsolatedMemoryPreferenceService::new());
        let caller = bare.with_service_ports(
            RuntimeServicePorts::default()
                .with_preferences(settings.clone())
                .with_user_path(settings),
        );
        let child = inherit_caller_settings(RuntimeServicePorts::default(), &caller);
        assert!(Rc::ptr_eq(
            caller.service_ports().preferences().unwrap(),
            child.preferences().unwrap()
        ));
        assert!(Rc::ptr_eq(
            caller.service_ports().user_path().unwrap(),
            child.user_path().unwrap()
        ));
        assert!(child.local_tasks().is_none());
        assert!(child.workspace().is_none());
    }
}
