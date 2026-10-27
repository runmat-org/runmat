//! Chunked private JSON record I/O. Numerical arrays are visited directly;
//! the ordinary serde codec is used only for bounded scalar/metadata items.
use super::{error, with_source, Cooperation};
use crate::BuiltinResult;
use serde::Serialize;
mod read;
pub(crate) use read::Reader;

pub(crate) struct Writer {
    pub(super) bytes: Vec<u8>,
}
impl Writer {
    pub(crate) fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    pub(crate) fn finish(self) -> Vec<u8> {
        self.bytes
    }
    pub(crate) fn token(&mut self, token: &[u8]) -> BuiltinResult<()> {
        self.bytes
            .try_reserve(token.len())
            .map_err(|source| with_source("selected capture byte allocation failed", source))?;
        self.bytes.extend_from_slice(token);
        Ok(())
    }
    pub(crate) fn key(&mut self, key: &'static str, first: bool) -> BuiltinResult<()> {
        if !first {
            self.token(b",")?;
        }
        self.token(b"\"")?;
        self.token(key.as_bytes())?;
        self.token(b"\":")
    }
    pub(crate) async fn metadata<T: Serialize + ?Sized>(
        &mut self,
        value: &T,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<()> {
        let first = self.bytes.len();
        serde_json::to_writer(&mut self.bytes, value)
            .map_err(|source| with_source("cannot encode selected capture metadata", source))?;
        c.completed((self.bytes.len() - first) as u64).await
    }
    pub(crate) async fn array<T: Serialize>(
        &mut self,
        values: impl IntoIterator<Item = T>,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<()> {
        self.token(b"[")?;
        let mut first = true;
        for value in values {
            if !first {
                self.token(b",")?;
            }
            first = false;
            self.metadata(&value, c).await?;
        }
        self.token(b"]")
    }
    pub(crate) async fn string(
        &mut self,
        value: &str,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<()> {
        self.token(b"\"")?;
        let mut start = 0;
        while start < value.len() {
            let mut end = value.len().min(start.saturating_add(1024));
            while !value.is_char_boundary(end) {
                end -= 1;
            }
            if end == start {
                return Err(error("invalid selected capture UTF-8 boundary"));
            }
            let part = serde_json::to_vec(&value[start..end])
                .map_err(|source| with_source("cannot encode selected capture text", source))?;
            self.token(&part[1..part.len() - 1])?;
            c.completed((end - start) as u64).await?;
            start = end;
        }
        self.token(b"\"")
    }
}
