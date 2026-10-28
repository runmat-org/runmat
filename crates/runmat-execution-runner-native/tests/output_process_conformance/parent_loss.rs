//! A real non-yielding worker must end when its original request pipe disappears.
use super::*;
use runmat_execution::console::{ConsoleLimits, ConsoleStream};
use runmat_execution_runner_native::WorkerProcessMessage;
use runmat_process_host::{
    ipc::{read_payload, write_payload, FrameLimits},
    HostCommand,
};

pub(super) async fn verify(config: &NativeExecutionConfig) {
    // The production launch for a placed Direct program: its exact reservation and console
    // controls. The source never yields, so only the worker's parent link can observe loss.
    let mut program = super::portable_units::request(
        "fprintf(1,'original before parent loss'); while true; first = 13; end",
        None,
        0,
    )
    .await;
    program.console = Some(ConsoleLimits {
        max_records: 4,
        max_text_bytes: 100,
    });
    let request = runmat_execution_artifact::ReservedProgramExecutionRequest::new(
        program,
        ResourceRequest {
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
    )
    .unwrap();
    let mut command = HostCommand::new(&config.executable);
    command.arguments = config.worker_arguments.clone();
    command.max_stderr_bytes = config.max_stderr_bytes;
    let mut child = command.spawn().await.unwrap();
    let stdio = child.take_stdio().unwrap();
    let mut reader = stdio.stdout;
    let mut writer = stdio.stdin;
    let limits = FrameLimits {
        max_message_bytes: config.max_message_bytes,
    };
    write_payload(&mut writer, &serde_json::to_vec(&request).unwrap(), limits)
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
    println!(
        "non-yielding original worker streamed its prefix and exited {:?} after parent loss",
        dropped.elapsed()
    );
}
