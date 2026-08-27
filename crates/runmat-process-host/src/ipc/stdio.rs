use tokio::io::{stdin, stdout, BufReader, BufWriter, Stdin, Stdout};

#[cfg(windows)]
use crate::ProcessHostError;
use crate::ProcessHostResult;

/// Opens the process standard streams for the framed IPC protocol.
///
/// Windows C runtimes default inherited standard streams to text mode. A frame
/// header or payload byte equal to `\n` would then be expanded to `\r\n`,
/// desynchronizing the binary length-prefixed stream. Select binary mode before
/// Tokio wraps the handles so every byte written by the codec reaches the peer
/// unchanged.
pub fn endpoint() -> ProcessHostResult<(BufReader<Stdin>, BufWriter<Stdout>)> {
    configure_binary_stdio()?;
    Ok((BufReader::new(stdin()), BufWriter::new(stdout())))
}

#[cfg(not(windows))]
fn configure_binary_stdio() -> ProcessHostResult<()> {
    Ok(())
}

#[cfg(windows)]
fn configure_binary_stdio() -> ProcessHostResult<()> {
    const STDIN_FD: std::ffi::c_int = 0;
    const STDOUT_FD: std::ffi::c_int = 1;
    const O_BINARY: std::ffi::c_int = 0x8000;

    unsafe extern "C" {
        fn _setmode(file_descriptor: std::ffi::c_int, mode: std::ffi::c_int) -> std::ffi::c_int;
    }

    // SAFETY: 0 and 1 are the C runtime's standard input/output descriptors,
    // and `_setmode` neither takes ownership nor retains either descriptor.
    let stdin_result = unsafe { _setmode(STDIN_FD, O_BINARY) };
    // SAFETY: Same invariant as above for the standard output descriptor.
    let stdout_result = unsafe { _setmode(STDOUT_FD, O_BINARY) };
    if stdin_result == -1 || stdout_result == -1 {
        return Err(ProcessHostError::Protocol(
            "failed to configure binary standard I/O for framed IPC".into(),
        ));
    }
    Ok(())
}
