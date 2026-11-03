use super::*;
use crate::context::{RuntimeContext, RuntimeServicePorts};
use std::{cell::Cell, rc::Rc};

struct DenyingHost(Rc<Cell<u32>>);
impl AsyncFeaRunAdmission for DenyingHost {
    fn admit(
        &self,
        request: FeaRunRequest,
    ) -> RuntimeServiceFuture<Result<Box<dyn AsyncFeaRunLease>, FeaRunDenied>> {
        assert_eq!(request.site, super::super::FeaRunSite::Local);
        assert_eq!(request.kind, FeaRunKind::Solve);
        assert_eq!(
            request.study_identity,
            runmat_execution::Digest::sha256(b"exact-domain-root").to_string()
        );
        self.0.set(self.0.get() + 1);
        Box::pin(async { Err(FeaRunDenied::SignInRequired) })
    }
}

#[test]
fn async_domain_admission_denies_before_polling_any_mesh_or_solver_effect() {
    let admitted = Rc::new(Cell::new(0));
    let effects = Cell::new(0);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_fea_run_admission(Rc::new(DenyingHost(admitted.clone()))),
        );
    let _entered = runtime.enter();
    let result = futures::executor::block_on(run(
        FeaRunKind::Solve,
        runmat_execution::Digest::sha256(b"exact-domain-root"),
        async {
            effects.set(effects.get() + 1);
            Ok::<_, FeaRunExecutionError>(42)
        },
        |failure| failure,
        |result, lease_error| result.and(Err(lease_error)),
    ));
    assert!(matches!(
        result,
        Err(FeaRunExecutionError::Denied(FeaRunDenied::SignInRequired))
    ));
    assert_eq!(admitted.get(), 1);
    assert_eq!(effects.get(), 0);
}

#[test]
fn no_async_host_keeps_the_existing_domain_gate_and_does_not_precharge_scripts() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let _entered = runtime.enter();
    let result = futures::executor::block_on(run(
        FeaRunKind::Solve,
        runmat_execution::Digest::sha256(b"domain"),
        async { Ok::<_, FeaRunExecutionError>(42) },
        |failure| failure,
        |result, lease_error| result.and(Err(lease_error)),
    ));
    assert_eq!(result.unwrap(), 42);
}
