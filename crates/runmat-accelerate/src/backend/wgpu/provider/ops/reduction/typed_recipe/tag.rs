//! Fixed source-tag storage: preparation queries do not allocate a String.
use super::stage::Stage;
use runmat_execution::Digest;
use std::fmt::Write as _;

// Current derived Digest Debug is Digest([u8;32]); every byte needs at most
// three digits and two separators. A future spelling change fails fallibly.
const CAPACITY: usize = "typed-reduction-".len()
    + "Single".len()
    + 1
    + "Digest([".len()
    + std::mem::size_of::<Digest>() * 5
    + "])".len();

pub(super) struct StageTag {
    bytes: [u8; CAPACITY],
    len: usize,
}
impl StageTag {
    pub(super) fn new(stage: Stage, identity: Digest) -> anyhow::Result<Self> {
        let mut tag = Self {
            bytes: [0; CAPACITY],
            len: 0,
        };
        write!(&mut tag, "typed-reduction-{stage:?}-{identity:?}")?;
        Ok(tag)
    }
    pub(super) fn as_str(&self) -> anyhow::Result<&str> {
        Ok(std::str::from_utf8(&self.bytes[..self.len])?)
    }
}
impl std::fmt::Write for StageTag {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let end = self.len.checked_add(text.len()).ok_or(std::fmt::Error)?;
        let target = self.bytes.get_mut(self.len..end).ok_or(std::fmt::Error)?;
        target.copy_from_slice(text.as_bytes());
        self.len = end;
        Ok(())
    }
}

pub(crate) struct ReductionPipelineKeys {
    pub(super) keys: [u64; 2],
    pub(super) len: usize,
}
impl AsRef<[u64]> for ReductionPipelineKeys {
    fn as_ref(&self) -> &[u64] {
        &self.keys[..self.len]
    }
}
