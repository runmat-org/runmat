use std::rc::Rc;

use runmat_runtime::context::RuntimeHostService;

use super::RunMatSession;

impl RunMatSession {
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
    fn same_service<T: ?Sized>(before: Option<&Rc<T>>, after: Option<&Rc<T>>) {
        match (before, after) {
            (Some(before), Some(after)) => assert!(Rc::ptr_eq(before, after)),
            (None, None) => {}
            _ => panic!("host installation changed another service port"),
        }
    }
    impl RuntimeHostService for Host {
        fn console(&self, _: runmat_runtime::console::ConsoleStream, _: String) {
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
}
