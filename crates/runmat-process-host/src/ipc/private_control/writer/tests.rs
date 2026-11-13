use std::future::Future;
use std::io::{self, Cursor, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
use std::task::Poll;
use std::time::Duration;

use super::*;

struct GateWriter {
    first: AtomicBool,
    started: Arc<Notify>,
    gate: Arc<(Mutex<bool>, Condvar)>,
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl Write for GateWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if !self.first.swap(true, Ordering::AcqRel) {
            self.started.notify_one();
            let (lock, wake) = &*self.gate;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
        }
        self.bytes.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn cancelling_a_reply_keeps_accepted_frames_whole_and_ordered_with_blocking_writes() {
    let limits = FrameLimits {
        max_message_bytes: 16,
    };
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let started = Arc::new(Notify::new());
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let writer = ControlFrameWriter::start(
        GateWriter {
            first: AtomicBool::new(false),
            started: started.clone(),
            gate: gate.clone(),
            bytes: bytes.clone(),
        },
        limits,
    )
    .unwrap();
    let first = {
        let writer = writer.clone();
        tokio::spawn(async move { writer.write_owned(vec![1]).await })
    };
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());

    let mut second = Box::pin(writer.write_owned(vec![2]));
    std::future::poll_fn(|context| {
        assert!(second.as_mut().poll(context).is_pending());
        Poll::Ready(())
    })
    .await;
    let (third_reply, third_result) = oneshot::channel();
    let third = {
        let writer = writer.clone();
        std::thread::spawn(move || {
            let _ = third_reply.send(writer.write_blocking(vec![3]));
        })
    };
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    tokio::time::timeout(Duration::from_secs(5), async {
        second.await.unwrap();
        third_result.await.unwrap().unwrap();
    })
    .await
    .unwrap();
    third.join().unwrap();

    let mut reader = Cursor::new(bytes.lock().unwrap().clone());
    for expected in [1, 2, 3] {
        assert_eq!(
            crate::ipc::read_payload_blocking(&mut reader, limits).unwrap(),
            vec![expected]
        );
    }
    assert_eq!(reader.position(), reader.get_ref().len() as u64);
}

struct FailedWriter;
impl Write for FailedWriter {
    fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "original writer failure",
        ))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn writer_failure_retains_one_original_cause_for_later_producers() {
    let writer = ControlFrameWriter::start(
        FailedWriter,
        FrameLimits {
            max_message_bytes: 16,
        },
    )
    .unwrap();
    let first = writer.write_owned(vec![1]).await.unwrap_err();
    let second = writer.write_blocking(vec![2]).unwrap_err();
    assert!(Arc::ptr_eq(&first.0, &second.0));
    assert!(matches!(first.0.as_ref(), ProcessHostError::Io(error)
        if error.kind() == io::ErrorKind::PermissionDenied));
}
