//! Bounded line acquisition shared by numeric delimited readers.

use std::io::{self, BufRead};

use super::cooperation::TabularCooperation;
use crate::RuntimeError;

pub(super) enum LineReadError {
    Io(io::Error),
    Cooperation(Box<RuntimeError>),
}

/// Reads through one retained invocation budget. At most 4 KiB is copied
/// between checkpoints, including when the source has no line terminator.
pub(super) async fn read_line_bytes<R: BufRead>(
    reader: &mut R,
    line: &mut Vec<u8>,
    cooperation: &mut TabularCooperation<'_>,
) -> Result<usize, LineReadError> {
    line.clear();
    let mut total = 0usize;
    loop {
        let available = reader.fill_buf().map_err(LineReadError::Io)?;
        if available.is_empty() {
            return Ok(total);
        }
        let limit = available.len().min(4096);
        let chunk = &available[..limit];
        let count = chunk
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(limit, |index| index + 1);
        let done = chunk[count - 1] == b'\n';
        line.try_reserve(count)
            .map_err(|source| LineReadError::Io(io::Error::other(source)))?;
        line.extend_from_slice(&chunk[..count]);
        reader.consume(count);
        total = total.checked_add(count).ok_or_else(|| {
            LineReadError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "tabular input line length exceeds platform limits",
            ))
        })?;
        cooperation
            .completed(count)
            .await
            .map_err(|source| LineReadError::Cooperation(Box::new(source)))?;
        if done {
            return Ok(total);
        }
    }
}
