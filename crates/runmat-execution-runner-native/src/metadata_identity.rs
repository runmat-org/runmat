//! Stream native metadata identity through the shared bounded JSON codec.
//! The caller's domain remains authoritative; no portable schema is invented here.
use crate::NativeExecutionResult;
use runmat_execution::Digest;
use serde::Serialize;
use sha2::{Digest as _, Sha256};

pub(crate) fn json_digest(
    domain: &[u8],
    value: &impl Serialize,
    maximum_bytes: u64,
) -> NativeExecutionResult<Digest> {
    json_identity(domain, value, maximum_bytes).map(|identity| identity.0)
}

pub(crate) fn json_identity(
    domain: &[u8],
    value: &impl Serialize,
    maximum_bytes: u64,
) -> NativeExecutionResult<(Digest, u64)> {
    struct Sink(Sha256);
    impl std::io::Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut sink = Sink(Sha256::new());
    sink.0.update(domain);
    let length = runmat_canonical_codec::write_json_bounded(value, &mut sink, maximum_bytes)?;
    Ok((Digest::from_bytes(sink.0.finalize().into()), length))
}
