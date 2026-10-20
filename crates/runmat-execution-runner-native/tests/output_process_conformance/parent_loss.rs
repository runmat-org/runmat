//! A real non-yielding worker must end when its original request pipe disappears.
use super::*;
use runmat_execution::console::{ConsoleLimits, ConsoleStream};
use runmat_execution_runner_native::WorkerProcessMessage;
use runmat_process_host::{
    ipc::{read_payload, write_payload, FrameLimits},
    HostCommand,
};

pub(super) async fn verify(config: &NativeExecutionConfig) {
    let mut call = call_source(1,
        "function [first, second] = original_output_values(); fprintf(1,'original before parent loss'); while true; first=13; end; second=29; end").await;
    call.console = Some(ConsoleLimits {
        max_records: 4,
        max_text_bytes: 100,
    });
    let request = materialized_request(
        &call,
        runmat_execution::OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        runmat_execution_artifact::ProgramTarget::portable("original-parent-loss"),
    );
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
    write_payload(
        &mut writer,
        &serde_json::to_vec(&runmat_execution_artifact::WorkerProgramRequest::Graph(
            request,
        ))
        .unwrap(),
        limits,
    )
    .await
    .unwrap();
    let bytes = tokio::time::timeout(Duration::from_secs(30), read_payload(&mut reader, limits))
        .await
        .expect("original non-yielding source did not deliver its first write")
        .unwrap();
    let WorkerProcessMessage::Console { record } = serde_json::from_slice(&bytes).unwrap() else {
        panic!("worker must stream the original write before entering its loop");
    };
    assert_eq!(record.sequence, 1);
    assert_eq!(record.stream, ConsoleStream::Stdout);
    assert_eq!(record.text.as_ref(), "original before parent loss");
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
    println!("non-yielding original worker streamed its prefix and exited on parent loss");
}
