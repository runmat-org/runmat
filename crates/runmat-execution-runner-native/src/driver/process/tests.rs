use std::sync::atomic::AtomicBool;
use std::time::Duration;

use tokio::io::AsyncWriteExt;

use super::*;
use crate::process_provider::exchange::read_payload_until_stopped;
use runmat_execution_runner::AttemptExecutionTermination;
use runmat_process_host::ipc::FrameLimits;

#[tokio::test]
async fn cancellation_polling_preserves_a_fragmented_frame() {
    let (mut writer, mut reader) = tokio::io::duplex(64);
    let payload = br#"{"message":"fragmented"}"#.to_vec();
    let length = u32::try_from(payload.len()).unwrap().to_be_bytes();
    let frame = [length.as_slice(), payload.as_slice()].concat();
    let sender = tokio::spawn(async move {
        for byte in frame {
            writer.write_all(&[byte]).await.unwrap();
            tokio::time::sleep(Duration::from_millis(12)).await;
        }
    });

    let cancelled = AtomicBool::new(false);
    let received = read_payload_until_stopped(
        &mut reader,
        FrameLimits {
            max_message_bytes: 1024,
        },
        || cancelled.load(std::sync::atomic::Ordering::Acquire),
    )
    .await
    .unwrap()
    .unwrap();

    sender.await.unwrap();
    assert_eq!(received, payload);
}

mod preflight;

#[cfg(unix)]
mod launcher;
