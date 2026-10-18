//! Presentation of admitted supervised-training progress.

use crate::console::{record_console_line, ConsoleStream};

pub(super) fn epoch(
    completed: usize,
    total: usize,
    loss: f64,
    verbose: bool,
) -> crate::BuiltinResult<()> {
    if verbose {
        record_console_line(
            ConsoleStream::Stdout,
            format!("Epoch {completed}/{total}: training loss {loss:.6}"),
        )?;
    }
    Ok(())
}
