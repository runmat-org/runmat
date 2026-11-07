//! Actual original portable outputs across process attempts and cold durable attachment.
use runmat_core::{ExecutableSource, RunMatSession};
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use runmat_execution_runner_native::supervisor::{
    run_local_supervisor, LocalSupervisorClient, LocalSupervisorConfig, SupervisorPaths,
};
use runmat_execution_runner_native::{NativeExecutionConfig, NativeExecutionService};
use runmat_process_host::HiddenMode;
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, DurableJobOptions, RuntimeExecutionServices,
};
use runmat_runtime::execution::{ExecutionServiceError, RuntimeExecutionService};
use runmat_value::Value;
use std::collections::BTreeSet;
use std::time::Duration;
use std::{cell::RefCell, rc::Rc};
#[path = "output_process_conformance/cancellation.rs"]
mod cancellation;
#[path = "output_process_conformance/durable_poll.rs"]
mod durable_poll;
#[path = "output_process_conformance/environment.rs"]
mod environment;
#[path = "supervisor_process_conformance/fixture.rs"]
mod fixture;
#[path = "output_process_conformance/handles.rs"]
mod handles;
#[path = "output_process_conformance/live_prefix.rs"]
mod live_prefix;
#[path = "output_process_conformance/nested_console.rs"]
mod nested_console;
#[path = "output_process_conformance/parent_loss.rs"]
mod parent_loss;
#[path = "output_process_conformance/portable_units.rs"]
mod portable_units;
#[path = "output_process_conformance/publisher_exit.rs"]
mod publisher_exit;

#[path = "support/native_host.rs"]
mod native_host;

fn main() {
    native_host::run(run);
}

fn run() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        if std::env::args().any(|arg| arg == environment::PRODUCER_MARKER) {
            environment::produce().await;
        } else if std::env::args().any(|arg| arg == HiddenMode::ExecutionWorker.marker()) {
            runmat_execution_runner_native::run_worker_stdio().await.unwrap();
        } else if std::env::args().any(|arg| arg == HiddenMode::ExecutionDriver.marker()) {
            use runmat_execution_runner_native::supervisor::{
                BatchDriverInvocation, ProgramBatchExecution,
                execute_and_complete_graph_batch, execute_and_complete_program_batch,
            };
            type DriverFuture = std::pin::Pin<Box<dyn std::future::Future<Output =
                runmat_execution_runner_native::NativeExecutionResult<ProgramBatchExecution>>>>;
            let (exit_marker, execution): (_, DriverFuture) = match
                runmat_execution_runner_native::supervisor::prepare_batch_driver().unwrap()
            {
                BatchDriverInvocation::Program { attempt, submission, lease, .. } => (
                    attempt.job_directory.join("test-provider-exit"),
                    Box::pin(execute_and_complete_program_batch(*attempt, *submission, *lease)),
                ),
                BatchDriverInvocation::GraphTask { attempt, submission, lease, .. } => (
                    attempt.job_directory.join("test-provider-exit"),
                    Box::pin(execute_and_complete_graph_batch(*attempt, *submission, *lease)),
                ),
                BatchDriverInvocation::Script { .. } | BatchDriverInvocation::Graph { .. } =>
                    panic!("original output witness requires an admitted task authority"),
            };
            // Abrupt publisher loss is independent of whether its numerical
            // execution yields to the async executor.
            let _publisher_exit = publisher_exit::Observer::start(exit_marker);
            execution.await.unwrap();
        } else if std::env::args().any(|arg| arg == HiddenMode::LocalSupervisor.marker()) {
            let config = LocalSupervisorConfig::for_current_executable().unwrap();
            let root = config.paths.root.clone();
            tokio::select! {
                result = run_local_supervisor(config) => result.unwrap(),
                _ = async { while !root.join("test-stop").exists() { tokio::time::sleep(Duration::from_millis(10)).await; } } => {}
            }
            std::fs::write(root.join("test-stopped"), b"stopped").unwrap();
        } else {
            witness().await;
        }
    });
}

#[derive(Default)]
struct CapturingService {
    serial: RuntimeExecutionService,
    original: RefCell<Option<(FutureHandle, DeferredCall)>>,
}
impl RuntimeExecutionServices for CapturingService {
    fn scope_id(&self) -> ExecutionScopeId {
        self.serial.scope_id()
    }
    fn requires_program_capture(&self) -> bool {
        true
    }
    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        let original = call.clone();
        let future = self.serial.create_future(call)?;
        *self.original.borrow_mut() = Some((future.clone(), original));
        Ok(future)
    }
    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        self.serial.spawn(future)
    }
    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError> {
        self.serial.begin_await(value)
    }
    fn complete_future(
        &self,
        future: runmat_runtime::execution::FutureCompletionRef<'_>,
        result: Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.serial.complete_future(future, result)
    }
    fn cancel(
        &self,
        value: &Value,
        reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        self.serial.cancel(value, reason)
    }
    fn drain_scope(&self, reason: CancellationReason) {
        self.serial.drain_scope(reason)
    }
    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        self.serial.current_pool()
    }
    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        self.serial.ensure_pool(request)
    }
    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError> {
        self.serial.close_pool(pool)
    }
    fn inspect_future(
        &self,
        future: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.serial.inspect_future(future)
    }
    fn inspect_task(
        &self,
        task: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.serial.inspect_task(task)
    }
    fn claim_next_task_result(
        &self,
        tasks: &[runmat_value::SharedTaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        self.serial.claim_next_task_result(tasks)
    }
    fn mark_task_result_read(&self, task: &TaskHandle) -> Result<(), ExecutionServiceError> {
        self.serial.mark_task_result_read(task)
    }
}

/// Keep the compiler's selected source issuer alive until the consumer accepts
/// the actual call; an envelope alone is not a deferred execution authority.
struct CapturedCall {
    call: DeferredCall,
    envelope: runmat_execution::ExecutableUnitEnvelope,
    _session: RunMatSession,
    _unit: runmat_core::ExecutableUnit,
    runtime: runmat_runtime::context::RuntimeContext,
}
impl std::ops::Deref for CapturedCall {
    type Target = DeferredCall;
    fn deref(&self) -> &DeferredCall {
        &self.call
    }
}
impl std::ops::DerefMut for CapturedCall {
    fn deref_mut(&mut self) -> &mut DeferredCall {
        &mut self.call
    }
}

async fn call(requested_outputs: usize) -> CapturedCall {
    let mut call = call_source(requested_outputs,
        "function [first, second] = original_output_values(); fprintf(1,'first'); fprintf(2,'middle'); clc; fprintf(1,'last'); first = 13; second = 29; end").await;
    call.console = Some(runmat_execution::console::ConsoleLimits {
        max_records: 8,
        max_text_bytes: 1000,
    });
    call
}

async fn call_source(requested_outputs: usize, source: &str) -> CapturedCall {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::default());
    session.install_execution_services(service.clone());
    let source = format!("{source}\nfunction f = fixture_capture(); f = parfeval(@original_output_values, {requested_outputs}); end");
    let unit = session
        .compile_executable_unit(
            ExecutableSource::new("original-output-owner", "original_output_values.m", source),
            None,
        )
        .await
        .unwrap();
    let envelope = unit
        .portable_envelope_for(Some("original_output_values"))
        .unwrap();
    let result = session
        .invoke_executable(
            &unit,
            runmat_core::ProcedureInvocation {
                target: runmat_core::ProcedureTarget::Function("fixture_capture".into()),
                arguments: vec![].into(),
                requested_outputs: 1,
            },
            &runmat_core::InvocationControl::default(),
        )
        .await
        .unwrap();
    assert_eq!(result.as_slice().len(), 1);
    let (_future, call) = service
        .original
        .borrow_mut()
        .take()
        .expect("compiler-selected deferred call");
    // parfeval has reserved its actual serial task before returning the
    // handle wrapper; the recorded call retains that selected compiler basis.
    let runtime = session.execution_runtime_context(unit.revision().program_revision.clone());
    CapturedCall {
        call,
        envelope,
        _session: session,
        _unit: unit,
        runtime,
    }
}

async fn await_in_context(
    service: &NativeExecutionService,
    value: Value,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError> {
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut value = value;
        loop {
            match service.begin_await_in_context(value, runtime)? {
                AwaitAction::Completed(result) => return Ok(result),
                AwaitAction::Pending(pending) => {
                    if let Some(wait) = service.wait_pending(&pending, runtime)? {
                        wait.await?;
                    } else {
                        tokio::task::yield_now().await;
                    }
                    value = pending;
                }
                other => panic!("original task await returned {other:?}"),
            }
        }
    })
    .await
    .expect("original task await did not finish")
}

async fn witness() {
    fixture::verify_unwind().await;
    let namespace = fixture::SupervisorFixture::new("original-output-authority");
    std::env::set_var("RUNMAT_EXECUTION_STATE_DIR", namespace.root());
    let config = NativeExecutionConfig::for_current_executable().unwrap();
    parent_loss::verify(&config).await;
    let service = NativeExecutionService::new(config.clone()).unwrap();
    handles::verify_caller(&service).await;
    portable_units::verify(&config).await;
    nested_console::verify(&config).await;
    environment::verify_producer_exit(&config).await;
    for requested in [0, 1, 2] {
        let original_call = call(requested).await;
        let future = service.create_future(original_call.call.clone()).unwrap();
        handles::reject_altered_future(&service, &future);
        let task = service.spawn(&future).unwrap();
        handles::reject_altered_task(&service, &task);
        let value = await_in_context(
            &service,
            Value::Task(task.clone().into()),
            &original_call.runtime,
        )
        .await
        .unwrap();
        assert_eq!(
            value.as_slice(),
            &vec![Value::Num(13.0), Value::Num(29.0)][..requested]
        );
        service
            .cancel(
                &Value::Future(future.clone().into()),
                runmat_execution::CancellationReason::User,
            )
            .unwrap();
        let AwaitAction::Completed(retained) = service
            .begin_await_in_context(Value::Future(future.clone().into()), &original_call.runtime)
            .unwrap()
        else {
            panic!("late cancellation changed accepted completion");
        };
        assert_eq!(retained, value);
        let durable_original = call(requested).await;
        let job = service
            .submit_job(
                durable_original.call.clone(),
                DurableJobOptions {
                    idempotency_key: Some(format!("original-outputs-{requested}")),
                    retention_millis:
                        runmat_execution_runner_native::supervisor::MIN_RETENTION_MILLIS,
                    resources: ResourceRequest {
                        cpu_millicores: config.worker_cpu_millicores,
                        memory_bytes: config.worker_memory_bytes,
                        scratch_bytes: config.worker_scratch_bytes,
                        max_wall_millis: 30_000,
                        // Include the bounded console-subtree frame, as the
                        // native driver's ordinary output reservation does.
                        max_artifact_bytes: u64::from(config.max_message_bytes),
                        max_egress_bytes: 0,
                        max_relay_bytes: 0,
                        accelerators: vec![],
                        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
                    },
                },
            )
            .unwrap();
        for mutation in 0..3 {
            let mut altered = job.clone();
            match mutation {
                0 => altered.outputs.requested_outputs += 1,
                1 => altered.generation += 1,
                _ => altered.run_id = runmat_execution::RunId::derive(&[b"foreign-runtime-job"]),
            }
            assert!(service
                .cancel(
                    &Value::Job(altered.into()),
                    runmat_execution::CancellationReason::Quota
                )
                .is_err());
        }
        let result = await_in_context(
            &service,
            Value::Job(job.clone().into()),
            &durable_original.runtime,
        )
        .await
        .unwrap();
        assert_eq!(
            result.as_slice(),
            &vec![Value::Num(13.0), Value::Num(29.0)][..requested]
        );
        let client = LocalSupervisorClient::for_current_executable().unwrap();
        let cold = client.attach(job.id, 0, 0).await.unwrap();
        assert_eq!(cold.record.handle, job);
        let evidence = cold
            .execution
            .expect("durable original task evidence must survive cold attachment");
        assert!(!evidence.attempts.is_empty());
        for attempt in &evidence.attempts {
            let transcript = attempt
                .terminal
                .as_ref()
                .unwrap()
                .console
                .as_ref()
                .expect("original console retained in cold attempt");
            transcript.validate().unwrap();
            assert!(transcript.source_receipt().is_some());
            assert_eq!(
                transcript.completion(),
                runmat_execution::console::ConsoleCompletion::Complete
            );
            assert_eq!(
                transcript
                    .records()
                    .iter()
                    .map(|record| (record.stream, record.text.as_ref()))
                    .collect::<Vec<_>>(),
                vec![
                    (runmat_execution::console::ConsoleStream::Stdout, "first"),
                    (runmat_execution::console::ConsoleStream::Stderr, "middle"),
                    (runmat_execution::console::ConsoleStream::ClearScreen, ""),
                    (runmat_execution::console::ConsoleStream::Stdout, "last"),
                ]
            );
        }
        let request = materialized_request(
            &durable_original,
            job.outputs.clone(),
            runmat_execution_artifact::ProgramTarget::portable(format!(
                "{}-{}-interpreter-bytecode-v2",
                std::env::consts::ARCH,
                std::env::consts::OS
            )),
        );
        request
            .validate_response(
                &cold
                    .response
                    .expect("original response must survive cold attachment"),
            )
            .unwrap();
        assert!(cold.resources.is_some());
        service
            .cancel(
                &Value::Job(job.clone().into()),
                runmat_execution::CancellationReason::Quota,
            )
            .unwrap();
        let AwaitAction::Completed(retained) = service
            .begin_await_in_context(Value::Job(job.clone().into()), &durable_original.runtime)
            .unwrap()
        else {
            panic!("late cancellation changed durable accepted completion");
        };
        assert_eq!(retained, result);
    }
    durable_poll::verify(&service, &config).await;
    live_prefix::verify(&service, &config, namespace.root()).await;
    drop(service);
    println!("original zero/one/multiple outputs passed isolated execution, durable await and cold task evidence");
}

fn materialized_request(
    call: &DeferredCall,
    outputs: runmat_execution::OutputContract,
    target: runmat_execution_artifact::ProgramTarget,
) -> runmat_execution_artifact::GraphExecutionRequest {
    let (_, request, _) = runmat_vm::materialize_deferred_graph_call(
        call,
        outputs,
        target,
        runmat_execution::value::ValueLimits::default(),
    )
    .unwrap();
    request
}
