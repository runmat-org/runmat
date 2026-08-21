use std::io::{Read, Write};

use crate::{ArtifactError, ArtifactResult};

pub(super) fn write_bytes(writer: &mut impl Write, bytes: &[u8], max: u64) -> ArtifactResult<()> {
    if bytes.len() as u64 > max {
        return Err(ArtifactError::Limit("archive member is too large".into()));
    }
    writer.write_all(&(bytes.len() as u64).to_be_bytes())?;
    writer.write_all(bytes)?;
    Ok(())
}

pub(super) fn read_bytes(
    reader: &mut impl Read,
    max: u64,
    total: &mut u64,
) -> ArtifactResult<Vec<u8>> {
    let mut encoded = [0_u8; 8];
    reader.read_exact(&mut encoded)?;
    let length = u64::from_be_bytes(encoded);
    if length > max {
        return Err(ArtifactError::Limit("archive member is too large".into()));
    }
    *total = total
        .checked_add(length)
        .ok_or_else(|| ArtifactError::Limit("archive length overflow".into()))?;
    let length = usize::try_from(length)
        .map_err(|_| ArtifactError::Limit("archive member cannot fit in memory".into()))?;
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}
