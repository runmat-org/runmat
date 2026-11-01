//! Process-scoped output is never relabeled as an individual test's stdout.
use std::fmt::Write;

use runmat_test::event::{TestEvent, TestEventPayload, WorkerProcessOutput};

#[derive(Default)]
pub(super) struct ProcessOutputReports(Vec<WorkerProcessOutput>);

impl ProcessOutputReports {
    pub(super) fn observe(&mut self, event: &TestEvent) {
        if let TestEventPayload::ProcessOutput { output } = &event.payload {
            // The raw byte owners are immutable and shared, not copied.
            self.0.push(output.clone());
        }
    }

    pub(super) fn text(&self) -> String {
        let mut text = String::new();
        for process in &self.0 {
            for (stream, captured) in [("stdout", &process.stdout), ("stderr", &process.stderr)] {
                if captured.bytes.as_slice().is_empty()
                    && captured.observed_bytes == Some(0)
                    && process.complete()
                {
                    continue;
                }
                let _ = writeln!(text, "Worker process {} {stream} (retirement={:?}, completion={:?}, observed_bytes={:?}, truncated={}):",
                    process.worker.0, process.retirement, captured.completion, captured.observed_bytes, captured.truncated);
                text.push_str(&String::from_utf8_lossy(captured.bytes.as_slice()));
                if !text.ends_with('\n') {
                    text.push('\n');
                }
            }
        }
        text
    }

    pub(super) fn tap(&self) -> String {
        // Application text must not inject TAP result lines. Treat both CR and
        // LF as line boundaries and keep every output line a TAP comment.
        self.text()
            .split(['\r', '\n'])
            .filter(|line| !line.is_empty())
            .map(|line| format!("# {line}\n"))
            .collect()
    }
}
