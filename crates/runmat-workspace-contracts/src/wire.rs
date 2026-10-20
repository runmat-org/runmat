//! One standard wire contract for native and WASM remote providers.
use serde::{de::DeserializeOwned, Serialize};
use std::io;

pub const MAX_WIRE_BYTES: usize = 2 * crate::MAX_WORKSPACE_PUBLICATION_BYTES + 8 * 1024 * 1024;

pub fn encode<T: Serialize>(value: &T) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    ciborium::into_writer(value, &mut bytes).map_err(io::Error::other)?;
    if bytes.len() > MAX_WIRE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "workspace request exceeds transport budget",
        ));
    }
    Ok(bytes)
}

pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> io::Result<T> {
    if bytes.len() > MAX_WIRE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "workspace response exceeds transport budget",
        ));
    }
    let mut reader = io::Cursor::new(bytes);
    let value = ciborium::de::from_reader_with_recursion_limit(&mut reader, 16)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if reader.position() != bytes.len() as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing workspace response data",
        ));
    }
    Ok(value)
}
