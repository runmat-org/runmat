//! Compare canonical encoding directly against admitted bytes.
pub(super) struct CountBytes(pub usize);
impl minicbor::encode::Write for CountBytes {
    type Error = &'static str;
    fn write_all(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or("chunk byte count overflow")?;
        Ok(())
    }
}

pub(super) struct CompareBytes<'a> {
    remaining: &'a [u8],
}
impl<'a> CompareBytes<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }
    pub(super) fn complete(&self) -> bool {
        self.remaining.is_empty()
    }
}
impl minicbor::encode::Write for CompareBytes<'_> {
    type Error = &'static str;
    fn write_all(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        if !self.remaining.starts_with(bytes) {
            return Err("chunk bytes are not canonical");
        }
        self.remaining = &self.remaining[bytes.len()..];
        Ok(())
    }
}
