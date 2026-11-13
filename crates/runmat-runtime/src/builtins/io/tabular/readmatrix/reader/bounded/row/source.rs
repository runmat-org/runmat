//! Fixed inline read buffer for one retained bounded source. The input
//! provider reads directly into this owner; no hidden `BufReader` heap
//! allocation sits outside the staged scan envelope.

use runmat_filesystem::FsBoundedInput;
use std::io::{self, BufRead, Read};

const READ_BUFFER_BYTES: usize = 4096;

pub(super) struct BoundedByteReader<'a> {
    source: &'a mut FsBoundedInput,
    bytes: [u8; READ_BUFFER_BYTES],
    position: usize,
    filled: usize,
}

impl<'a> BoundedByteReader<'a> {
    pub(super) fn new(source: &'a mut FsBoundedInput) -> Self {
        Self {
            source,
            bytes: [0; READ_BUFFER_BYTES],
            position: 0,
            filled: 0,
        }
    }
}

impl Read for BoundedByteReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        let available = self.fill_buf()?;
        let count = available.len().min(output.len());
        output[..count].copy_from_slice(&available[..count]);
        self.consume(count);
        Ok(count)
    }
}

impl BufRead for BoundedByteReader<'_> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.position == self.filled {
            self.position = 0;
            self.filled = self.source.read(&mut self.bytes)?;
        }
        Ok(&self.bytes[self.position..self.filled])
    }

    fn consume(&mut self, amount: usize) {
        self.position = self.position.saturating_add(amount).min(self.filled);
    }
}
