use std::rc::Rc;
use std::time::Duration;

use runmat_process_host::ipc::FrameLimits;

use super::{read_invocation_message, InvocationRead};
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use crate::foreign::python::isolation::{PythonHostMessage, PythonInvocationResult};

struct ObservedReader {
    inner: tokio::io::DuplexStream,
    consumed: Rc<std::cell::Cell<usize>>,
    changed: Rc<tokio::sync::Notify>,
}

impl tokio::io::AsyncRead for ObservedReader {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
        buffer: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let before = buffer.filled().len();
        let result = std::pin::Pin::new(&mut self.inner).poll_read(context, buffer);
        let count = buffer.filled().len() - before;
        if count != 0 {
            self.consumed.set(self.consumed.get() + count);
            self.changed.notify_one();
        }
        result
    }
}

async fn wait_for_consumed(
    consumed: &std::cell::Cell<usize>,
    changed: &tokio::sync::Notify,
    count: usize,
) {
    loop {
        let notification = changed.notified();
        if consumed.get() >= count {
            return;
        }
        notification.await;
    }
}

fn invocation_frame(request_id: u64, limits: FrameLimits) -> Vec<u8> {
    let message = PythonHostMessage::Invocation(PythonInvocationResult {
        request_id,
        outcome: Ok(Vec::new()),
    });
    let mut frame = Vec::new();
    runmat_process_host::ipc::write_payload_blocking(
        &mut frame,
        &serde_json::to_vec(&message).unwrap(),
        limits,
    )
    .unwrap();
    frame
}

#[tokio::test]
async fn fragmented_invocation_keeps_one_read_across_cancellation_polls() {
    use tokio::io::AsyncWriteExt;

    let limits = FrameLimits {
        max_message_bytes: 1024,
    };
    let first = invocation_frame(17, limits);
    let second = invocation_frame(18, limits);
    let split = 4 + (first.len() - 4) / 2;
    let (mut writer, inner) = tokio::io::duplex(128);
    let consumed = Rc::new(std::cell::Cell::new(0));
    let changed = Rc::new(tokio::sync::Notify::new());
    let mut reader = ObservedReader {
        inner,
        consumed: consumed.clone(),
        changed: changed.clone(),
    };
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let producer = async {
        writer.write_all(&first[..2]).await.unwrap();
        wait_for_consumed(&consumed, &changed, 2).await;
        // The production cancellation ticker fires while only a partial
        // header is owned; restarting read_exact here corrupts the next frame.
        tokio::time::sleep(Duration::from_millis(35)).await;
        writer.write_all(&first[2..split]).await.unwrap();
        wait_for_consumed(&consumed, &changed, split).await;
        tokio::time::sleep(Duration::from_millis(35)).await;
        writer.write_all(&first[split..]).await.unwrap();
        writer.write_all(&second).await.unwrap();
    };
    let consumer = async {
        for expected in [17, 18] {
            let InvocationRead::Message(PythonHostMessage::Invocation(result)) =
                read_invocation_message(&mut reader, limits, &context)
                    .await
                    .unwrap()
            else {
                panic!("expected the exact invocation frame");
            };
            assert_eq!(result.request_id, expected);
            assert_eq!(result.outcome, Ok(Vec::new()));
        }
    };
    tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(producer, consumer);
    })
    .await
    .expect("fragmented frames must complete without loss or duplication");
    assert_eq!(consumed.get(), first.len() + second.len());
}

#[tokio::test]
async fn cancelling_a_partial_invocation_read_releases_its_reader_borrow() {
    use tokio::io::AsyncWriteExt;

    let limits = FrameLimits {
        max_message_bytes: 1024,
    };
    let frame = invocation_frame(19, limits);
    let (mut writer, inner) = tokio::io::duplex(128);
    let consumed = Rc::new(std::cell::Cell::new(0));
    let changed = Rc::new(tokio::sync::Notify::new());
    let mut reader = ObservedReader {
        inner,
        consumed: consumed.clone(),
        changed: changed.clone(),
    };
    let cancellation = runmat_execution::CancellationSource::new();
    let context = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    // Stop inside a real payload, not before its header has been read.
    writer.write_all(&frame[..6]).await.unwrap();
    let cancel = async {
        wait_for_consumed(&consumed, &changed, 6).await;
        cancellation.cancel(runmat_execution::CancellationReason::User);
    };
    let read = async {
        assert!(matches!(
            read_invocation_message(&mut reader, limits, &context)
                .await
                .unwrap(),
            InvocationRead::Cancelled
        ));
    };
    tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(cancel, read);
    })
    .await
    .expect("cancellation must release an incomplete record");
    assert_eq!(consumed.get(), 6);
    // The production caller can now retire the child; do not reuse this
    // deliberately incomplete transport as if it were at a frame boundary.
    drop(reader);
}
