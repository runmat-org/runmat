//! Actual original portable outputs across process attempts and cold durable attachment.
use runmat_core::{ExecutableSource, RunMatSession};
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution_runner_native::supervisor::{
    run_local_supervisor, LocalSupervisorClient, LocalSupervisorConfig, SupervisorPaths,
};
use runmat_execution_runner_native::{NativeExecutionConfig, NativeExecutionService};
use runmat_process_host::HiddenMode;
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, DurableJobOptions, RuntimeExecutionServices,
};
use runmat_value::Value;
use std::collections::BTreeSet;
use std::time::Duration;
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
#[path = "output_process_conformance/parent_loss.rs"]
mod parent_loss;
#[path = "output_process_conformance/portable_units.rs"]
mod portable_units;

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        if std::env::args().any(|arg| arg == environment::PRODUCER_MARKER) {
            environment::produce().await;
        } else if std::env::args().any(|arg| arg == cancellation::WORKER_MARKER) {
            cancellation::blocked_worker().await;
        } else if std::env::args().any(|arg| arg == HiddenMode::ExecutionWorker.marker()) {
            runmat_execution_runner_native::run_worker_stdio().await.unwrap();
        } else if std::env::args().any(|arg| arg == HiddenMode::ExecutionDriver.marker()) {
            let runmat_execution_runner_native::supervisor::BatchDriverInvocation::Program { attempt, submission, lease, .. } = runmat_execution_runner_native::supervisor::prepare_batch_driver().unwrap() else { panic!("original output witness requires program authority"); };
            runmat_execution_runner_native::supervisor::execute_and_complete_program_batch(*attempt, *submission, *lease).await.unwrap();
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

async fn call(requested_outputs: usize) -> DeferredCall {
    let mut call = call_source(
        requested_outputs,
        "function [first, second] = original_output_values(); fprintf(1,'first'); fprintf(2,'middle'); clc; fprintf(1,'last'); first = 13; second = 29; end",
    ).await;
    call.console = Some(runmat_execution::console::ConsoleLimits {
        max_records: 8,
        max_text_bytes: 1000,
    });
    call
}

async fn call_source(requested_outputs: usize, source: &str) -> DeferredCall {
    let mut session = RunMatSession::with_options(false, false).unwrap();
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
    let function = usize::try_from(envelope.manifest.identity.entrypoint_function.0).unwrap();
    DeferredCall {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        function,
        arguments: vec![],
        requested_outputs,
        program_revision: Some(unit.revision().program_revision.clone()),
        program: Some(envelope.canonical_bytes().unwrap()),
    }
}

fn expected(requested_outputs: usize) -> Value {
    match requested_outputs {
        0 => Value::OutputList(vec![]),
        1 => Value::Num(13.0),
        2 => Value::OutputList(vec![Value::Num(13.0), Value::Num(29.0)]),
        _ => panic!("unexpected original output contract"),
    }
}

async fn witness() {
    fixture::verify_unwind().await;
    let namespace = fixture::SupervisorFixture::new("original-output-authority");
    std::env::set_var("RUNMAT_EXECUTION_STATE_DIR", namespace.root());
    let config = NativeExecutionConfig::for_current_executable().unwrap();
    parent_loss::verify(&config).await;
    let service = NativeExecutionService::new(config.clone()).unwrap();
    handles::verify_caller(&service).await;
    cancellation::verify(&config).await;
    portable_units::verify(&config).await;
    environment::verify_producer_exit(&config).await;
    for requested in [0, 1, 2] {
        let future = service.create_future(call(requested).await).unwrap();
        handles::reject_altered_future(&service, &future);
        let task = service.spawn(&future).unwrap();
        handles::reject_altered_task(&service, &task);
        let value = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                match service.begin_await(Value::Task(task.clone())).unwrap() {
                    AwaitAction::Pending(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                    AwaitAction::Completed(value) => break value,
                    other => panic!("spawned original task returned {other:?}"),
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(value, expected(requested));
        service
            .cancel(
                &Value::Future(future.clone()),
                runmat_execution::CancellationReason::User,
            )
            .unwrap();
        let AwaitAction::Completed(retained) =
            service.begin_await(Value::Future(future.clone())).unwrap()
        else {
            panic!("late cancellation changed accepted completion");
        };
        assert_eq!(retained, value);
        let job = service
            .submit_job(
                call(requested).await,
                DurableJobOptions {
                    idempotency_key: Some(format!("original-outputs-{requested}")),
                    retention_millis:
                        runmat_execution_runner_native::supervisor::MIN_RETENTION_MILLIS,
                    resources: ResourceRequest {
                        cpu_millicores: config.worker_cpu_millicores,
                        memory_bytes: config.worker_memory_bytes,
                        scratch_bytes: config.worker_scratch_bytes,
                        max_wall_millis: 30_000,
                        max_artifact_bytes: 1024 * 1024,
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
                    &Value::Job(altered),
                    runmat_execution::CancellationReason::Quota
                )
                .is_err());
        }
        assert_eq!(service.await_job(&job).unwrap(), expected(requested));
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
        let original = call(requested).await;
        let request = runmat_vm::materialize_deferred_call(
            &original,
            job.outputs.clone(),
            runmat_execution_artifact::ProgramTarget::portable(format!(
                "{}-{}-interpreter-bytecode-v1",
                std::env::consts::ARCH,
                std::env::consts::OS
            )),
        )
        .unwrap();
        cold.response
            .expect("original response must survive cold attachment")
            .validate_against(&request)
            .unwrap();
        assert!(cold.resources.is_some());
        service
            .cancel(
                &Value::Job(job.clone()),
                runmat_execution::CancellationReason::Quota,
            )
            .unwrap();
        assert_eq!(service.await_job(&job).unwrap(), expected(requested));
    }
    durable_poll::verify(&service, &config).await;
    drop(service);
    println!("original zero/one/multiple outputs passed isolated execution, durable await and cold task evidence");
}
