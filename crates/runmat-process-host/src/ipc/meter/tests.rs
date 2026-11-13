use super::*;
use crate::ipc::{read_payload, write_payload, FrameLimits};
use tokio::io::AsyncWriteExt;

#[tokio::test]
async fn framing_counts_actual_bytes_and_retains_partial_eof_reads() {
    let limits = FrameLimits {
        max_message_bytes: 128,
    };
    let (writer, reader) = tokio::io::duplex(128);
    let sent = IpcByteCounter::default();
    let received = IpcByteCounter::default();
    let mut writer = MeteredIo::new(writer, sent.clone());
    let mut reader = MeteredIo::new(reader, received.clone());
    write_payload(&mut writer, b"original", limits)
        .await
        .unwrap();
    assert_eq!(
        read_payload(&mut reader, limits).await.unwrap(),
        b"original"
    );
    assert_eq!(sent.bytes(), Some(12));
    assert_eq!(received.bytes(), Some(12));
    writer.write_all(&[0, 0, 0, 5, 1, 2]).await.unwrap();
    drop(writer);
    assert!(read_payload(&mut reader, limits).await.is_err());
    assert_eq!(sent.bytes(), Some(18));
    assert_eq!(received.bytes(), Some(18));
}

struct PartialWriter {
    first: bool,
}
impl AsyncWrite for PartialWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.first {
            self.first = false;
            Poll::Ready(Ok(buf.len().min(2)))
        } else {
            Poll::Ready(Err(io::Error::other("broken pipe")))
        }
    }
    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

#[tokio::test]
async fn failed_write_counts_only_the_committed_prefix() {
    let counter = IpcByteCounter::default();
    let mut writer = MeteredIo::new(PartialWriter { first: true }, counter.clone());
    assert!(writer.write_all(b"original").await.is_err());
    assert_eq!(counter.bytes(), Some(2));
}

#[tokio::test]
async fn overflow_drops_evidence_instead_of_claiming_a_clamped_total() {
    let counter = IpcByteCounter::default();
    counter.state.bytes.store(u64::MAX, Ordering::Release);
    let mut writer = MeteredIo::new(tokio::io::sink(), counter.clone());
    assert!(writer.write_all(b"x").await.is_err());
    assert_eq!(counter.bytes(), None);
}
