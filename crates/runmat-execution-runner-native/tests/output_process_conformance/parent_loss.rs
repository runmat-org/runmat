//! A real non-yielding worker must end when its original request pipe disappears.
use super::*;
use runmat_execution::console::{ConsoleBudget, ConsoleLimits, ConsoleStream};
use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::{PoolBackend, PoolId, ProgramExecutionAssignment, TaskId};
use runmat_execution_runner_native::WorkerProcessMessage;
use runmat_process_host::{
    environment::{EnvironmentAllowlist, EnvironmentPolicy},
    ipc::{private_control::PrivateControlListener, read_payload, write_payload, FrameLimits},
    HostCommand, StdioPolicy,
};
use tokio::io::AsyncReadExt;

async fn collect_output(mut reader: impl tokio::io::AsyncRead + Unpin) -> Vec<u8> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).await.unwrap();
    bytes
}

pub(super) async fn verify(config: &NativeExecutionConfig) {
    // The production launch for a placed Direct program: its exact reservation and console
    // controls. The source never yields, so only the worker's parent link can observe loss.
    let mut program = super::portable_units::request(
        "fprintf(1,'original before parent loss'); while true; first = 13; end",
        None,
        0,
    )
    .await;
    let console = ConsoleLimits {
        max_records: 4,
        max_text_bytes: 100,
    };
    program.console = Some(console);
    program.assignment = Some(ProgramExecutionAssignment {
        scope_id: ExecutionScopeId::derive(&[b"parent-loss-scope"]),
        pool_id: PoolId::derive(&[b"parent-loss-pool"]),
        task_id: TaskId::derive(&[b"parent-loss-task"]),
        attempt_id: AttemptId::derive(&[b"parent-loss-attempt"]),
        worker_id: WorkerId::derive(&[b"parent-loss-worker"]),
        backend: PoolBackend::LocalProcesses,
        resources: Default::default(),
    });
    let resources = ResourceRequest {
        cpu_millicores: config.worker_cpu_millicores,
        memory_bytes: config.worker_memory_bytes,
        scratch_bytes: config.worker_scratch_bytes,
        max_wall_millis: 30_000,
        max_artifact_bytes: 1024 * 1024,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: vec![],
        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    };
    let request =
        runmat_execution_artifact::ReservedProgramExecutionRequest::new(program, resources.clone())
            .unwrap();
    let mut command = HostCommand::new(&config.executable);
    command.arguments = config.worker_arguments.clone();
    // The test executable needs its parent's native loader/bootstrap paths.
    command.environment_policy = EnvironmentPolicy::Allow(EnvironmentAllowlist::platform_runtime());
    command.max_stderr_bytes = config.max_stderr_bytes;
    let listener = PrivateControlListener::bind().unwrap();
    command.stdio = StdioPolicy::PrivateControl(listener.endpoint().clone());
    let mut child = command.spawn().await.unwrap();
    let output = child.take_application_output().unwrap();
    let stdout = tokio::spawn(collect_output(output.stdout));
    let stderr = tokio::spawn(collect_output(output.stderr));
    let control = tokio::time::timeout(Duration::from_secs(30), listener.accept())
        .await
        .expect("worker private control connection")
        .unwrap();
    let mut reader = control.reader;
    let mut writer = control.writer;
    let limits = FrameLimits {
        max_message_bytes: config.max_message_bytes,
    };
    // Match the production reserved-console wire envelope and its finite grant.
    #[derive(serde::Serialize)]
    struct ConsoleLaunch<'a> {
        request: &'a runmat_execution_artifact::ReservedProgramExecutionRequest,
        console_budget: ConsoleBudget,
    }
    let launch = ConsoleLaunch {
        request: &request,
        console_budget: runmat_execution_runner_native::console_source_subgrant(
            &resources, console,
        )
        .unwrap(),
    };
    write_payload(&mut writer, &serde_json::to_vec(&launch).unwrap(), limits)
        .await
        .unwrap();
    let bytes = tokio::time::timeout(Duration::from_secs(30), read_payload(&mut reader, limits))
        .await
        .expect("original non-yielding source did not deliver its first write")
        .unwrap();
    let message: WorkerProcessMessage = serde_json::from_slice(&bytes).unwrap();
    let WorkerProcessMessage::Console { record } = message else {
        panic!("worker must stream the original write before entering its loop, got {message:?}");
    };
    assert_eq!(record.sequence, 1);
    assert_eq!(record.stream, ConsoleStream::Stdout);
    assert_eq!(record.text.as_ref(), "original before parent loss");
    let dropped = std::time::Instant::now();
    drop(writer);
    let exit = match tokio::time::timeout(Duration::from_secs(30), child.wait()).await {
        Ok(exit) => exit.unwrap(),
        Err(_) => {
            child.terminate_tree().await.unwrap();
            child.wait().await.unwrap();
            panic!("original worker survived loss of its parent pipe");
        }
    };
    assert!(!exit.success);
    assert_eq!(exit.code, Some(1));
    let stdout = stdout.await.unwrap();
    let stderr = stderr.await.unwrap();
    println!(
        "non-yielding original worker streamed its prefix and exited {:?} after parent loss; raw stdout={} bytes, stderr={} bytes",
        dropped.elapsed(), stdout.len(), stderr.len()
    );
}
