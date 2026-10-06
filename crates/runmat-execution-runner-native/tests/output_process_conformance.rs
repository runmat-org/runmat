//! Actual original portable outputs across process attempts and cold durable attachment.
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
use runmat_vm::{FunctionBytecode, FunctionRegistry, Instr};
use std::collections::{BTreeSet, HashMap};
use std::time::Duration;
#[path = "output_process_conformance/cancellation.rs"]
mod cancellation;
#[path = "output_process_conformance/durable_poll.rs"]
mod durable_poll;
#[path = "supervisor_process_conformance/fixture.rs"]
mod fixture;
#[path = "output_process_conformance/handles.rs"]
mod handles;
#[path = "output_process_conformance/portable_units.rs"]
mod portable_units;

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        if std::env::args().any(|arg| arg == cancellation::WORKER_MARKER) {
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

fn call(requested_outputs: usize) -> DeferredCall {
    let function = FunctionBytecode {
        display_name: "original_output_values".into(),
        instructions: vec![
            Instr::LoadConst(13.0),
            Instr::StoreVar(0),
            Instr::LoadConst(29.0),
            Instr::StoreVar(1),
            Instr::Return,
        ],
        var_count: 2,
        output_slots: vec![0, 1],
        ..Default::default()
    };
    let registry = FunctionRegistry::new(HashMap::from([(function.function, function)]));
    DeferredCall {
        function: 0,
        arguments: vec![],
        requested_outputs,
        program_revision: None,
        program: Some(serde_json::to_vec(&registry).unwrap()),
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
    let service = NativeExecutionService::new(config.clone()).unwrap();
    handles::verify_caller(&service).await;
    cancellation::verify(&config).await;
    portable_units::verify(&config).await;
    for requested in [0, 1, 2] {
        let future = service.create_future(call(requested)).unwrap();
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
                call(requested),
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
        let original = call(requested);
        let (recipe, artifact, arguments) = runmat_vm::materialize_deferred_call(
            &original,
            job.outputs.clone(),
            runmat_execution_artifact::ProgramTarget::portable(format!(
                "{}-{}-interpreter-bytecode-v1",
                std::env::consts::ARCH,
                std::env::consts::OS
            )),
        )
        .unwrap();
        let request = runmat_execution_artifact::ProgramExecutionRequest {
            schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe,
            artifact,
            function: 0,
            arguments,
            requested_outputs: requested as u16,
        };
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
