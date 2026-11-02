//! Drain on the original process reactor without spawning detached pipe readers.
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use runmat_process_host::{ChildOutput, ProcessOutputCaptureWriter};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::Notify;

use super::{RawOutputError, RawOutputJournal, RawOutputState, RawOutputStream, OUTPUT_PAGE_BYTES};

pub(crate) trait OutputDrainControl: Sync {
    fn raw_output_changed(&self);
    fn stop_for_raw_output(&self, error: Arc<RawOutputError>);
    fn raw_output_failure(&self) -> Option<Arc<RawOutputError>>;
}

#[derive(Default)]
pub(crate) struct OutputDrainLifetime {
    finished: AtomicBool,
    wake: Notify,
}

impl OutputDrainLifetime {
    pub(crate) fn finish(&self) {
        self.finished.store(true, Ordering::Release);
        self.wake.notify_waiters();
    }

    async fn wait(&self) {
        loop {
            let notified = self.wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.finished.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }
}

pub(crate) async fn drain_pipes(
    pipes: Option<ChildOutput>,
    journal: &Mutex<RawOutputJournal>,
    control: &(impl OutputDrainControl + ?Sized),
    lifetime: &OutputDrainLifetime,
) -> Result<(), Arc<RawOutputError>> {
    let Some(pipes) = pipes else { return Ok(()) };
    let drain = async {
        let (stdout, stderr) = tokio::join!(
            drain_stream(
                pipes.stdout,
                RawOutputStream::Stdout,
                None,
                journal,
                control
            ),
            drain_stream(
                pipes.stderr,
                RawOutputStream::Stderr,
                Some(pipes.stderr_diagnostics),
                journal,
                control
            ),
        );
        // Both original producers finish even when one fails first.
        combine_stream_outcomes(control.raw_output_failure(), [stdout, stderr])
    };
    tokio::pin!(drain);
    tokio::select! {
        result = &mut drain => result,
        _ = lifetime.wait() => {
            // The owner has completed its physical retirement attempt. Do not
            // keep an unverified tree's inherited descriptors alive forever.
            match tokio::time::timeout(Duration::from_secs(30), &mut drain).await {
                Ok(result) => result,
                Err(_) => {
                    let mut error = control.raw_output_failure().unwrap_or_else(|| Arc::new(
                        RawOutputError::Invalid("application streams did not finish after process retirement")
                    ));
                    let mut journal = journal.lock().expect("raw output journal poisoned");
                    if let Err(cleanup) = journal.fail_open_streams() {
                        error = Arc::new(RawOutputError::Cleanup {
                            primary: error,
                            cleanup: Box::new(cleanup),
                        });
                    }
                    control.stop_for_raw_output(Arc::clone(&error));
                    control.raw_output_changed();
                    Err(error)
                }
            }
        }
    }
}

fn combine_stream_outcomes(
    mut first: Option<Arc<RawOutputError>>,
    outcomes: [Result<(), Arc<RawOutputError>>; 2],
) -> Result<(), Arc<RawOutputError>> {
    for other in outcomes.into_iter().filter_map(Result::err) {
        first = Some(match first {
            Some(primary) if Arc::ptr_eq(&primary, &other) => primary,
            Some(primary) => Arc::new(RawOutputError::Streams { primary, other }),
            None => other,
        });
    }
    first.map_or(Ok(()), Err)
}

async fn drain_stream(
    mut pipe: impl AsyncRead + Unpin,
    stream: RawOutputStream,
    mut diagnostics: Option<ProcessOutputCaptureWriter>,
    journal: &Mutex<RawOutputJournal>,
    control: &(impl OutputDrainControl + ?Sized),
) -> Result<(), Arc<RawOutputError>> {
    let mut buffer = [0; OUTPUT_PAGE_BYTES];
    let mut failure = None;
    let outcome = loop {
        let count = match pipe.read(&mut buffer).await {
            Ok(count) => count,
            Err(error) => {
                let kind = error.kind();
                failure.get_or_insert_with(|| Arc::new(RawOutputError::Io(error)));
                break Err(kind);
            }
        };
        if count == 0 {
            break Ok(());
        }
        if let Some(capture) = diagnostics.as_mut() {
            if let Err(error) = capture.observe(&buffer[..count]) {
                failure.get_or_insert_with(|| {
                    Arc::new(RawOutputError::Io(std::io::Error::other(error)))
                });
                control.stop_for_raw_output(Arc::clone(
                    failure.as_ref().expect("capture failure retained"),
                ));
            }
        }
        if failure.is_none() {
            let appended = journal
                .lock()
                .expect("raw output journal poisoned")
                .append(stream, &buffer[..count]);
            if let Err(error) = appended {
                let error = Arc::new(error);
                control.stop_for_raw_output(Arc::clone(&error));
                failure = Some(error);
            }
        }
        control.raw_output_changed();
        // Continue reading after capture failure until the owner retires the
        // child. A full spool must not turn cancellation into a blocked write.
    };
    if let Some(capture) = diagnostics.as_mut() {
        capture.finish(outcome);
    }
    let state = if outcome.is_ok() && failure.is_none() {
        RawOutputState::EndOfStream
    } else {
        RawOutputState::ReadFailed
    };
    if let Err(cleanup) = journal
        .lock()
        .expect("raw output journal poisoned")
        .finish_stream(stream, state)
    {
        failure = Some(match failure.take() {
            Some(primary) => Arc::new(RawOutputError::Cleanup {
                primary,
                cleanup: Box::new(cleanup),
            }),
            None => Arc::new(cleanup),
        });
    }
    if let Some(error) = &failure {
        control.stop_for_raw_output(Arc::clone(error));
    }
    control.raw_output_changed();
    failure.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joining_streams_keeps_first_cause_and_independent_failure() {
        let first = Arc::new(RawOutputError::Invalid("first read failed"));
        let other = Arc::new(RawOutputError::Invalid("other publication failed"));
        let failure = combine_stream_outcomes(
            Some(Arc::clone(&first)),
            [Err(Arc::clone(&first)), Err(Arc::clone(&other))],
        )
        .unwrap_err();
        let RawOutputError::Streams {
            primary,
            other: secondary,
        } = failure.as_ref()
        else {
            panic!("independent stream failure must be retained");
        };
        assert!(Arc::ptr_eq(primary, &first));
        assert!(Arc::ptr_eq(secondary, &other));
        let single =
            combine_stream_outcomes(Some(Arc::clone(&first)), [Err(Arc::clone(&first)), Ok(())])
                .unwrap_err();
        assert!(Arc::ptr_eq(&single, &first));
        assert!(combine_stream_outcomes(None, [Ok(()), Ok(())]).is_ok());
    }
}
