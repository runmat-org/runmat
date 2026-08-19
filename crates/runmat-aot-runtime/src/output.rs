use std::io::Write as _;
use std::sync::Arc;

pub struct ConsoleForwarder;

pub fn install_console_forwarder() -> ConsoleForwarder {
    runmat_runtime::console::install_forwarder(Some(Arc::new(|entry| match entry.stream {
        runmat_runtime::console::ConsoleStream::Stdout => write_stdout(entry.text.as_bytes()),
        runmat_runtime::console::ConsoleStream::Stderr => write_stderr(entry.text.as_bytes()),
        runmat_runtime::console::ConsoleStream::ClearScreen => {
            write_stdout(b"\x1b[2J\x1b[H");
        }
    })));
    ConsoleForwarder
}

impl Drop for ConsoleForwarder {
    fn drop(&mut self) {
        runmat_runtime::console::install_forwarder(None);
    }
}

pub fn value(value: &runmat_value::Value) {
    println!("{value}");
}

pub fn error(message: &str) {
    eprintln!("RunMat standalone error: {message}");
}

fn write_stdout(bytes: &[u8]) {
    let mut stream = std::io::stdout().lock();
    let _ = stream.write_all(bytes);
    let _ = stream.flush();
}

fn write_stderr(bytes: &[u8]) {
    let mut stream = std::io::stderr().lock();
    let _ = stream.write_all(bytes);
    let _ = stream.flush();
}
