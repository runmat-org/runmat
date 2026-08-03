use std::io::Read;

use crate::{ArtifactError, ArtifactResult, ExecutionBundle, LogicalObject, ObjectDescriptor};

use super::{
    canonical::read_bytes,
    manifest_codec::{decode_descriptor, decode_manifest},
    ArchiveLimits, MAGIC,
};

pub fn read_bundle(
    mut reader: impl Read,
    limits: ArchiveLimits,
) -> ArtifactResult<ExecutionBundle> {
    let mut magic = vec![0_u8; MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(ArtifactError::Invalid(
            "invalid bundle archive magic".into(),
        ));
    }
    let mut total = 0_u64;
    let manifest = read_bytes(&mut reader, limits.max_manifest_bytes, &mut total)?;
    let manifest = decode_manifest(&manifest)?;
    let mut encoded_count = [0_u8; 4];
    reader.read_exact(&mut encoded_count)?;
    let count = u32::from_be_bytes(encoded_count);
    if count > limits.max_objects {
        return Err(ArtifactError::Limit("too many archive objects".into()));
    }
    let mut objects = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let descriptor = read_bytes(&mut reader, limits.max_manifest_bytes, &mut total)?;
        let descriptor: ObjectDescriptor = decode_descriptor(&descriptor)?;
        let bytes = read_bytes(&mut reader, limits.max_object_bytes, &mut total)?;
        if total > limits.max_total_bytes {
            return Err(ArtifactError::Limit("archive is too large".into()));
        }
        objects.push(LogicalObject { descriptor, bytes });
    }
    let mut trailing = [0_u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(ArtifactError::Invalid(
            "bundle archive contains trailing bytes".into(),
        ));
    }
    let bundle = ExecutionBundle { manifest, objects };
    bundle.validate()?;
    Ok(bundle)
}
