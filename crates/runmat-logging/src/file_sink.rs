//! Owned native append sink. No runtime hook, exporter, logger or queue owner.
use std::fs::File;
use std::io::{self, Write};
use std::sync::{Arc, Mutex, MutexGuard};
use tracing_subscriber::fmt::MakeWriter;

pub(super) struct FileSink<W = File>(Arc<Mutex<State<W>>>);
struct State<W> {
    writer: W,
    failure: Option<io::ErrorKind>,
}

impl<W> Clone for FileSink<W> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<W: Write> FileSink<W> {
    pub(super) fn new(writer: W) -> Self {
        Self(Arc::new(Mutex::new(State {
            writer,
            failure: None,
        })))
    }
    pub(super) fn flush_result(&self) -> Result<(), String> {
        let mut state = self
            .0
            .lock()
            .map_err(|_| "Configured log file lock poisoned")?;
        if let Err(error) = state.writer.flush() {
            state.record_failure(error.kind());
        }
        match state.failure {
            Some(kind) => Err(format!(
                "Configured log file write/flush failed: {}",
                io::Error::from(kind)
            )),
            None => Ok(()),
        }
    }
}

impl<W> State<W> {
    fn record_failure(&mut self, kind: io::ErrorKind) {
        if self.failure.is_none() {
            self.failure = Some(kind);
            // Never recurse through tracing or echo a possibly private event.
            eprintln!(
                "RunMat: configured log file failed; command finalization will report the failure"
            );
        }
    }
}

pub(super) struct SinkWriter<'a, W = File>(MutexGuard<'a, State<W>>);
impl<'a, W: Write + 'a> MakeWriter<'a> for FileSink<W> {
    type Writer = SinkWriter<'a, W>;
    fn make_writer(&'a self) -> Self::Writer {
        SinkWriter(self.0.lock().unwrap_or_else(|poison| poison.into_inner()))
    }
}
impl<W: Write> Write for SinkWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some(kind) = self.0.failure {
            return Err(io::Error::from(kind));
        }
        let result = self.0.writer.write(bytes);
        if matches!(result, Ok(0)) && !bytes.is_empty() {
            self.0.record_failure(io::ErrorKind::WriteZero);
            return Err(io::ErrorKind::WriteZero.into());
        }
        if let Err(error) = &result {
            self.0.record_failure(error.kind());
        }
        result
    }
    fn flush(&mut self) -> io::Result<()> {
        let result = self.0.writer.flush();
        if let Err(error) = &result {
            self.0.record_failure(error.kind());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn one_sink_receives_span_events_and_bridged_records_and_latches_failures() {
        let path = std::env::temp_dir().join(format!(
            "runmat-log-file-{}-{}.log",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(b"retained-prefix\n").unwrap();
        let sink = FileSink::new(file);
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .without_time()
                .with_writer(sink.clone())
                .with_span_events(
                    tracing_subscriber::fmt::format::FmtSpan::NEW
                        | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
                ),
        );
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("file_span");
            let _entered = span.enter();
            tracing::info!("trace_event");
            crate::forward_log_record(
                &tracing_log::log::Record::builder()
                    .level(tracing_log::log::Level::Info)
                    .target("file_test")
                    .args(format_args!("original_log_record"))
                    .build(),
            )
            .unwrap();
        });
        sink.flush_result().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        for expected in [
            "retained-prefix",
            "file_span",
            "new",
            "close",
            "trace_event",
            "original_log_record",
        ] {
            assert!(text.contains(expected), "missing {expected}: {text}");
        }
        drop(sink);
        std::fs::remove_file(path).unwrap();
        enum Failure {
            Write,
            Zero,
            Flush,
        }
        impl Write for Failure {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                match self {
                    Self::Write => Err(io::ErrorKind::Other.into()),
                    Self::Zero => Ok(0),
                    Self::Flush => Ok(bytes.len()),
                }
            }
            fn flush(&mut self) -> io::Result<()> {
                match self {
                    Self::Flush => Err(io::ErrorKind::Other.into()),
                    _ => Ok(()),
                }
            }
        }
        for failure in [Failure::Write, Failure::Zero, Failure::Flush] {
            let failed = FileSink::new(failure);
            let _ = failed.make_writer().write_all(b"private event");
            assert!(failed.flush_result().is_err());
            assert!(failed.make_writer().write_all(b"another event").is_err());
            assert!(failed.flush_result().is_err());
        }
    }
}
