use runmat_core::{ExecutionStreamEntry, ExecutionStreamKind};
use std::io::{self, Write};

pub(crate) fn emit_execution_streams(streams: &[ExecutionStreamEntry]) {
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    for entry in streams {
        match entry.stream {
            ExecutionStreamKind::Stdout | ExecutionStreamKind::Stderr => {
                let bytes = entry.text.as_bytes();
                let write_result = match entry.stream {
                    ExecutionStreamKind::Stdout => stdout.write_all(bytes),
                    ExecutionStreamKind::Stderr => stderr.write_all(bytes),
                    ExecutionStreamKind::ClearScreen => unreachable!(),
                };
                if let Err(err) = write_result {
                    eprintln!("Failed to write execution stream: {err}");
                    break;
                }
            }
            ExecutionStreamKind::ClearScreen => {
                if atty::is(atty::Stream::Stdout) {
                    print!("\x1B[2J\x1B[H");
                    let _ = io::stdout().flush();
                }
            }
        }
    }
    let _ = stdout.flush();
    let _ = stderr.flush();
}
