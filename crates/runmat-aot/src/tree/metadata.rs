use crate::{AotError, AotResult};
use std::io::{self, Write};

struct Count {
    bytes: u64,
    limit: u64,
}
impl Write for Count {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .filter(|count| *count <= self.limit)
            .ok_or_else(|| io::Error::other("native metadata exceeds its admitted bound"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn admit_json(value: &impl serde::Serialize, limit: u64) -> AotResult<u64> {
    let mut count = Count { bytes: 0, limit };
    serde_json::to_writer(&mut count, value).map_err(resource)?;
    Ok(count.bytes)
}

pub(crate) fn canonical_json(value: &impl serde::Serialize, limit: u64) -> AotResult<Vec<u8>> {
    let length = usize::try_from(admit_json(value, limit)?).map_err(resource)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(resource)?;
    struct Output {
        bytes: Vec<u8>,
        length: usize,
    }
    impl Write for Output {
        fn write(&mut self, part: &[u8]) -> io::Result<usize> {
            if self
                .bytes
                .len()
                .checked_add(part.len())
                .is_none_or(|end| end > self.length)
            {
                return Err(io::Error::other(
                    "native metadata exceeded its preflight extent",
                ));
            }
            self.bytes.extend_from_slice(part);
            Ok(part.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = Output { bytes, length };
    serde_json::to_writer(&mut output, value).map_err(resource)?;
    if output.bytes.len() != length {
        return Err(resource("native metadata changed during encoding"));
    }
    Ok(output.bytes)
}

pub(super) fn require_canonical(value: &impl serde::Serialize, bytes: &[u8]) -> AotResult<()> {
    struct Compare<'a> {
        bytes: &'a [u8],
        offset: usize,
    }
    impl Write for Compare<'_> {
        fn write(&mut self, part: &[u8]) -> io::Result<usize> {
            let end = self
                .offset
                .checked_add(part.len())
                .ok_or_else(|| io::Error::other("metadata extent overflow"))?;
            if self.bytes.get(self.offset..end) != Some(part) {
                return Err(io::Error::other("native metadata is not canonical"));
            }
            self.offset = end;
            Ok(part.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut compare = Compare { bytes, offset: 0 };
    serde_json::to_writer(&mut compare, value)
        .map_err(|error| AotError::contract("aot.tree.codec", error.to_string()))?;
    if compare.offset != bytes.len() {
        return Err(AotError::contract(
            "aot.tree.codec",
            "trailing native metadata",
        ));
    }
    Ok(())
}

fn resource(error: impl std::fmt::Display) -> AotError {
    AotError::ResourceUnavailable {
        message: error.to_string(),
    }
}
