use std::rc::Rc;

use runmat_runtime::context::{RuntimeGraphicsPresentationService, RuntimeHostService};

use super::RunMatSession;

impl RunMatSession {
    pub(super) fn retire_graphics_host(&self) {
        if let Some(service) = self.runtime_context.service_ports().graphics_presentation() {
            service.retire();
        }
        if let Some(callbacks) = &self.graphics_callbacks {
            callbacks.close();
        }
    }
    /// Join an explicitly supplied local graphics host. Absence never gains
    /// ambient native authority, including process/remote worker construction.
    pub fn attach_graphics_host(
        &mut self,
        host: &SessionGraphicsHost,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        match host {
            SessionGraphicsHost::Unavailable => {}
            #[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
            SessionGraphicsHost::Native { opener, gui } => {
                let queue = runmat_runtime::context::RuntimeGraphicsCallbackQueue::new_for_selected(
                    &self.runtime_context,
                    gui.maximum_pending_callbacks,
                    gui.maximum_callback_registrations,
                );
                let presentation: Rc<dyn RuntimeGraphicsPresentationService> =
                    Rc::new(runmat_runtime::context::NativeGraphicsAttachment::new(
                        opener.clone(),
                        queue.downgrade(),
                        gui.maximum_windows,
                        gui.maximum_snapshot_bytes,
                        runmat_runtime::context::NativeWindowConfig {
                            width: gui.width,
                            height: gui.height,
                            vsync: gui.vsync,
                            maximized: gui.maximized,
                            ..Default::default()
                        },
                    )?);
                let ports = self
                    .runtime_context
                    .service_ports()
                    .clone()
                    .with_graphics_callback_queue(queue.downgrade())
                    .with_graphics_presentation(Rc::clone(&presentation));
                let staged = self.runtime_context.clone().with_service_ports(ports);
                if let Err(error) = queue.install_retained_graphics_properties(&staged) {
                    presentation.retire();
                    queue.close();
                    return Err(error);
                }
                let previous_presentation = self
                    .runtime_context
                    .service_ports()
                    .graphics_presentation()
                    .cloned();
                let previous_queue = self.graphics_callbacks.replace(queue);
                self.runtime_context = staged;
                if let Some(previous) = previous_presentation {
                    previous.retire();
                }
                if let Some(previous) = previous_queue {
                    previous.close();
                }
            }
        }
        Ok(())
    }

    /// Explicit host flush for command/idle boundaries, not an invented frame
    /// acknowledgment. The selected service owns the actual operation.
    pub async fn flush_graphics(&self) -> Result<(), runmat_runtime::RuntimeError> {
        use runmat_runtime::context::{
            GraphicsCallbackPolicy, GraphicsFlushRate, GraphicsFlushRequest,
        };
        if let Some(service) = self.runtime_context.service_ports().graphics_presentation() {
            let operation = service.flush(GraphicsFlushRequest {
                rate: GraphicsFlushRate::Immediate,
                callbacks: GraphicsCallbackPolicy::Process,
            });
            #[cfg(not(target_arch = "wasm32"))]
            self.local_tasks.participate_foreground(operation).await??;
            #[cfg(target_arch = "wasm32")]
            operation.await?;
        }
        Ok(())
    }
    /// Install this session's application host, including document interaction.
    ///
    /// The host owns console forwarding, warnings and interactive input as well
    /// as document requests. Other runtime service ports remain unchanged. This
    /// does not grant Desktop capability; the embedding application separately
    /// declares the capabilities it actually supplies.
    pub fn install_host_service(&mut self, service: Rc<dyn RuntimeHostService>) {
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_host(service);
        self.runtime_context = self.runtime_context.clone().with_service_ports(ports);
    }

    /// Install the owning application's frame-presentation and callback-pump
    /// service. A host must supply actual completion receipts; merely having
    /// a plot registry or numeric figure handle does not grant this port.
    pub fn install_graphics_presentation_service(
        &mut self,
        service: Rc<dyn RuntimeGraphicsPresentationService>,
    ) {
        if let Some(previous) = self.runtime_context.service_ports().graphics_presentation() {
            previous.retire();
        }
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_graphics_presentation(service);
        self.runtime_context = self.runtime_context.clone().with_service_ports(ports);
    }

    /// Retain the callback queue at the session owner, outside the selected
    /// service port and any callback registration's originating context.
    pub fn install_graphics_callback_queue(
        &mut self,
        maximum_pending_events: std::num::NonZeroUsize,
        maximum_registrations: std::num::NonZeroUsize,
    ) -> runmat_runtime::context::RuntimeGraphicsCallbackQueue {
        let queue = runmat_runtime::context::RuntimeGraphicsCallbackQueue::new_for_selected(
            &self.runtime_context,
            maximum_pending_events,
            maximum_registrations,
        );
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_graphics_callback_queue(queue.downgrade());
        self.runtime_context = self.runtime_context.clone().with_service_ports(ports);
        let previous = self.graphics_callbacks.replace(queue.clone());
        if let Some(previous) = previous {
            previous.close();
        }
        queue
    }

    pub fn graphics_callback_queue(
        &self,
    ) -> Option<&runmat_runtime::context::RuntimeGraphicsCallbackQueue> {
        self.graphics_callbacks.as_ref()
    }
}

/// Local, nonserialized production attachment. GUI builds still require an
/// actual OS-main-thread opener; selecting a feature alone grants nothing.
#[derive(Clone, Default)]
pub enum SessionGraphicsHost {
    #[default]
    Unavailable,
    #[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
    Native {
        opener: runmat_runtime::context::NativeWindowOpenPort,
        gui: runmat_config::runtime::GuiConfig,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::{
        context::{HostInteraction, RuntimeServiceFuture},
        interaction::InteractionResponse,
        RuntimeError,
    };

    struct Host;
    struct Graphics(Rc<std::cell::Cell<usize>>);

    impl runmat_runtime::context::RuntimeGraphicsPresentationService for Graphics {
        fn retire(&self) {
            self.0.set(self.0.get() + 1);
        }
        fn flush(
            &self,
            _: runmat_runtime::context::GraphicsFlushRequest,
        ) -> std::pin::Pin<Box<dyn runmat_runtime::context::RuntimeGraphicsFlushOperation>>
        {
            panic!("installing a service must not request a frame")
        }
    }
    fn same_service<T: ?Sized>(before: Option<&Rc<T>>, after: Option<&Rc<T>>) {
        match (before, after) {
            (Some(before), Some(after)) => assert!(Rc::ptr_eq(before, after)),
            (None, None) => {}
            _ => panic!("host installation changed another service port"),
        }
    }
    impl RuntimeHostService for Host {
        fn console(
            &self,
            _: runmat_runtime::console::ConsoleStream,
            _: runmat_runtime::console::ConsoleText,
        ) {
            panic!("installation must not emit console output")
        }
        fn warning(&self, _: runmat_runtime::warning_store::RuntimeWarning) {
            panic!("installation must not emit warnings")
        }
        fn interact(
            &self,
            _: HostInteraction,
        ) -> RuntimeServiceFuture<Result<InteractionResponse, RuntimeError>> {
            panic!("installation must not request input")
        }
    }

    #[test]
    fn host_installation_preserves_existing_session_service_identities() {
        runmat_gc::gc_test_context(|| {
            let mut session = RunMatSession::with_options(false, false).unwrap();
            let before = session.runtime_context.service_ports().clone();
            let host: Rc<dyn RuntimeHostService> = Rc::new(Host);
            session.install_host_service(host.clone());
            let after = session.runtime_context.service_ports();
            assert!(Rc::ptr_eq(after.host().unwrap(), &host));
            same_service(before.call(), after.call());
            same_service(before.acceleration(), after.acceleration());
            same_service(before.placement(), after.placement());
            same_service(before.spmd(), after.spmd());
            same_service(before.distributed(), after.distributed());
        });
    }

    #[test]
    fn graphics_installation_retains_selected_port_without_entering_host_effects() {
        runmat_gc::gc_test_context(|| {
            let mut session = RunMatSession::with_options(false, false).unwrap();
            let before = session.runtime_context.service_ports().clone();
            let retired = Rc::new(std::cell::Cell::new(0));
            let service: Rc<dyn RuntimeGraphicsPresentationService> =
                Rc::new(Graphics(retired.clone()));
            session.install_graphics_presentation_service(service.clone());
            let after = session.runtime_context.service_ports();
            assert!(Rc::ptr_eq(after.graphics_presentation().unwrap(), &service));
            same_service(before.host(), after.host());
            same_service(before.call(), after.call());
            same_service(before.placement(), after.placement());
            same_service(before.distributed(), after.distributed());
            assert_eq!(
                retired.get(),
                0,
                "installation must not retire the new attachment"
            );
            let capacity = std::num::NonZeroUsize::new(2).unwrap();
            let queue = session.install_graphics_callback_queue(capacity, capacity);
            drop(session);
            assert!(queue.is_closed());
            assert_eq!(
                retired.get(),
                1,
                "session drop retires even externally retained services"
            );
        });
    }

    #[test]
    fn replacing_or_dropping_session_retires_external_callback_queue_clones() {
        runmat_gc::gc_test_context(|| {
            let mut session = RunMatSession::with_options(false, false).unwrap();
            let capacity = std::num::NonZeroUsize::new(2).unwrap();
            let first = session.install_graphics_callback_queue(capacity, capacity);
            let external = first.clone();
            let second = session.install_graphics_callback_queue(capacity, capacity);
            assert!(first.is_closed());
            assert!(external.is_closed());
            assert!(!second.is_closed());
            drop(session);
            assert!(second.is_closed());
        });
    }

    #[test]
    fn session_composes_one_base_workspace_and_revokes_it_on_drop() {
        runmat_gc::gc_test_context(|| {
            let session = RunMatSession::with_options(false, false).unwrap();
            let ports = session.runtime_context.service_ports();
            let current = ports.workspace().expect("current workspace service");
            let owner = ports
                .base_workspace()
                .expect("base workspace authority")
                .owner();

            current
                .assign("answer", runmat_value::Value::Num(42.0))
                .unwrap();
            assert_eq!(
                owner.lookup("answer").unwrap(),
                Some(runmat_value::Value::Num(42.0))
            );

            drop(session);
            assert!(matches!(
                owner.snapshot(),
                Err(runmat_runtime::workspace::BaseWorkspaceError::Closed)
            ));
        });
    }
}
