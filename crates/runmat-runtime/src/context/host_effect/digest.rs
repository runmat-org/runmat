//! Allocation-free hashing for closed typed resource facts.

use runmat_execution::Digest;
use sha2::{Digest as _, Sha256};
use std::io::{self, Write};

struct DigestWriter(Sha256);

impl Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn hash_serializable<T: serde::Serialize>(facts: &T) -> Digest {
    let mut writer = DigestWriter(Sha256::new());
    serde_json::to_writer(&mut writer, facts).expect("closed resource facts serialize");
    Digest::from_bytes(writer.0.finalize().into())
}
