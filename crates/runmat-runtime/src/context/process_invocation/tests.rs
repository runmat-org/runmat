use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_process_host::{ProcessHostError, ProcessHostResult};

use super::*;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeProcessService,
    RuntimeServiceFuture, RuntimeServicePorts,
};

struct CountedProcess(Rc<Cell<usize>>);

impl RuntimeProcessService for CountedProcess {
    fn output_complete(
        &self,
        _command: HostCommand,
    ) -> RuntimeServiceFuture<ProcessHostResult<CompleteProcessOutput>> {
        let calls = self.0.clone();
        Box::pin(async move {
            calls.set(calls.get() + 1);
            Err(ProcessHostError::Protocol("unexpected process poll".into()))
        })
    }

    fn output_complete_with_input(
        &self,
        _command: HostCommand,
        _input: Vec<u8>,
    ) -> RuntimeServiceFuture<ProcessHostResult<CompleteProcessOutput>> {
        self.output_complete(HostCommand::new("unused"))
    }
}

struct Finite;

#[test]
fn process_environment_policies_select_session_values_and_explicit_overrides() {
    use runmat_process_host::environment::{EnvironmentAllowlist, EnvironmentPolicy};

    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    runtime
        .install_environment_snapshot(vec![
            ("KEPT".into(), "session".into()),
            ("OVERRIDE".into(), "original".into()),
        ])
        .unwrap();
    for (policy, inherits_kept) in [
        (EnvironmentPolicy::Inherit, true),
        (EnvironmentPolicy::InheritExcept(vec!["KEPT".into()]), false),
        (EnvironmentPolicy::Clear, false),
        (
            EnvironmentPolicy::Allow(EnvironmentAllowlist::new(["KEPT".into()])),
            true,
        ),
    ] {
        let mut command = HostCommand::new("unused");
        command.environment_policy = policy;
        command
            .environment
            .insert("OVERRIDE".into(), "explicit".into());
        select_environment(&runtime, &mut command);
        assert_eq!(command.environment_policy, EnvironmentPolicy::Clear);
        assert_eq!(command.environment.get("OVERRIDE").unwrap(), "explicit");
        assert_eq!(
            command.environment.get("KEPT").map(String::as_str),
            inherits_kept.then_some("session")
        );
        assert_eq!(command.environment.len(), 1 + usize::from(inherits_kept));
    }
    assert_eq!(runtime.environment_snapshot()[1].1, "original");
}

impl RuntimeParallelService for Finite {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: Some(1),
            ..Default::default()
        }
    }
}

#[test]
fn finite_budget_rejects_both_output_forms_before_process_call() {
    let calls = Rc::new(Cell::new(0));
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_process(Rc::new(CountedProcess(calls.clone())))
                .with_parallel(Rc::new(Finite)),
        );
    for input in [None, Some(vec![1, 2, 3])] {
        let error = futures::executor::block_on(output_complete(
            Some(&runtime),
            HostCommand::new("/bin/sh"),
            input,
            None,
        ))
        .expect_err("unbounded output is inadmissible under finite quota");
        assert!(matches!(error, ProcessInvocationError::ResourceEvidence));
    }
    assert_eq!(calls.get(), 0);
}

#[test]
fn already_cancelled_invocation_never_polls_owned_process() {
    let _serial = runmat_filesystem::provider_override_lock();
    let _native =
        runmat_filesystem::replace_provider(Arc::new(runmat_filesystem::NativeFsProvider));
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let calls = Rc::new(Cell::new(0));
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_process(Rc::new(CountedProcess(calls.clone()))),
    );
    let error = futures::executor::block_on(output_complete(
        Some(&runtime),
        HostCommand::new("/bin/sh"),
        Some(vec![1, 2, 3]),
        None,
    ))
    .expect_err("pre-cancelled invocation must not start the process");
    assert!(matches!(
        error,
        ProcessInvocationError::Execution(ExecutionServiceError::Cancelled)
    ));
    assert_eq!(calls.get(), 0);
}
