use super::*;
use crate::context::{RuntimeServicePorts, RuntimeWorkspaceService};
use crate::execution::RuntimeExecutionService;
use runmat_execution::profiling::{
    current_command_profiling, CommandProfilingRoot, DeferredWorkCounts,
};
use runmat_value::Value;

struct EmptyWorkspace;

impl RuntimeWorkspaceService for EmptyWorkspace {
    fn assign(&self, _: &str, _: Value) -> Result<(), crate::RuntimeError> {
        Err(crate::runtime_error::semantic_error(
            "RunMat:GraphicsCallbackTest:UnexpectedWorkspaceMutation",
            "graphics callback profiling witness does not admit workspace assignment",
        ))
    }

    fn clear(&self) -> Result<(), crate::RuntimeError> {
        Err(crate::runtime_error::semantic_error(
            "RunMat:GraphicsCallbackTest:UnexpectedWorkspaceMutation",
            "graphics callback profiling witness does not admit workspace clear",
        ))
    }

    fn remove(&self, _: &str) -> Result<(), crate::RuntimeError> {
        Err(crate::runtime_error::semantic_error(
            "RunMat:GraphicsCallbackTest:UnexpectedWorkspaceMutation",
            "graphics callback profiling witness does not admit workspace removal",
        ))
    }

    fn lookup(&self, _: &str) -> Option<Value> {
        None
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        Vec::new()
    }

    fn global_names(&self) -> Vec<String> {
        Vec::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn host_send_and_request_keep_registration_context_but_neither_command_origin() {
    for request in [false, true] {
        let registration_command = CommandProfilingRoot::new();
        let draining_command = CommandProfilingRoot::new();
        let workspace: Rc<dyn RuntimeWorkspaceService> = Rc::new(EmptyWorkspace);
        let selected = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_workspace(workspace.clone()))
            .with_command_profiling(registration_command.receipt());
        let selected_identity = selected.local_identity();
        let draining = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_command_profiling(draining_command.receipt());
        let queue = RuntimeGraphicsCallbackQueue::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        );
        let called = Rc::new(Cell::new(0));
        let callback: Rc<GraphicsCallbackResponder> = {
            let workspace = workspace.clone();
            let called = called.clone();
            Rc::new(move |_| {
                let workspace = workspace.clone();
                let called = called.clone();
                Box::pin(async move {
                    let active = crate::context::scope::active_runtime_context()
                        .expect("registration semantic context selected");
                    assert_eq!(active.local_identity(), selected_identity);
                    assert!(Rc::ptr_eq(
                        active.service_ports().workspace().expect("exact workspace"),
                        &workspace,
                    ));
                    assert!(
                        current_command_profiling().is_none(),
                        "host event has neither installation nor draining command origin",
                    );
                    called.set(called.get() + 1);
                    Ok(GraphicsCallbackResult::Completed)
                })
            })
        };
        let (registration, ingress) = {
            let _installation = selected.enter();
            queue
                .register_response(selected, 7, GraphicsCallbackKind::PanActionPre, callback)
                .expect("register under the first command")
        };
        assert!(
            queue
                .0
                .registrations
                .borrow()
                .values()
                .all(|registration| registration.selected.command_profiling().is_none()),
            "the stored semantic context cannot retain the registration command",
        );
        let event = GraphicsCallbackHostEvent {
            figure: 7,
            kind: GraphicsCallbackKind::PanActionPre,
            position: None,
        };
        let _draining = draining.enter();
        let ticket = if request {
            Some(ingress.request(event).expect("host request"))
        } else {
            ingress.send(event).expect("host send");
            None
        };
        futures::executor::block_on(queue.process()).expect("drain host event");
        if let Some(ticket) = ticket {
            assert_eq!(
                futures::executor::block_on(ticket).expect("host reply"),
                GraphicsCallbackResult::Completed,
            );
        }
        assert_eq!(called.get(), 1);
        assert!(current_command_profiling().is_some(), "drainer restored");
        drop(registration);
        assert_eq!(
            registration_command.seal().deferred_work,
            Some(DeferredWorkCounts::default()),
        );
        assert_eq!(
            draining_command.seal().deferred_work,
            Some(DeferredWorkCounts::default()),
        );
    }
}

#[cfg(all(feature = "plot-core", not(target_arch = "wasm32")))]
#[test]
fn property_replacement_retains_semantic_owner_without_either_installation_command() {
    use crate::builtins::plotting::state::{
        close_figure, figure_graphics_receiver, reset_plot_state, select_figure, FigureHandle,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

    let _plot_guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    let figure = FigureHandle::from(38);
    select_figure(figure);
    let receiver = figure_graphics_receiver(figure)
        .expect("issue receiver")
        .expect("live figure");
    let base = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let queue = RuntimeGraphicsCallbackQueue::new_for_selected(
        &base,
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(2).unwrap(),
    );
    let ports = base
        .service_ports()
        .clone()
        .with_graphics_callback_queue(queue.downgrade());
    let selected = base.with_service_ports(ports);
    let identity = selected.local_identity();
    let first = CommandProfilingRoot::new();
    let second = CommandProfilingRoot::new();
    let callback: Rc<GraphicsCallbackResponder> =
        Rc::new(|_| Box::pin(async { Ok(GraphicsCallbackResult::Completed) }));
    let kind = GraphicsCallbackKind::PanActionPre;
    queue
        .replace_graphics_property_response(
            selected.clone().with_command_profiling(first.receipt()),
            receiver.clone(),
            kind,
            callback.clone(),
        )
        .expect("initial property");
    queue
        .replace_graphics_property_response(
            selected.with_command_profiling(second.receipt()),
            receiver.clone(),
            kind,
            callback,
        )
        .expect("replacement property");
    let registrations = queue.0.registrations.borrow();
    assert_eq!(registrations.len(), 1);
    let retained = registrations.values().next().unwrap();
    assert_eq!(retained.selected.local_identity(), identity);
    assert!(retained.selected.command_profiling().is_none());
    drop(registrations);
    assert!(queue.clear_graphics_property_response(&receiver, kind));
    close_figure(Some(figure)).expect("close figure");
    assert_eq!(
        first.seal().deferred_work,
        Some(DeferredWorkCounts::default())
    );
    assert_eq!(
        second.seal().deferred_work,
        Some(DeferredWorkCounts::default())
    );
}
