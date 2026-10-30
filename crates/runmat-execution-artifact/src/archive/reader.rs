use std::io::Read;

use crate::{
    ArtifactError, ArtifactResult, ExecutionBundle, LogicalObject, ObjectDescriptor,
    ObjectNamespace, MAX_FOREIGN_POLICY_BYTES, MAX_PROJECT_RUNTIME_POLICY_BYTES,
};

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
        if descriptor.namespace == ObjectNamespace::ForeignPolicy {
            crate::object::validate_foreign_policy_descriptor(&descriptor)?;
        }
        if descriptor.namespace == ObjectNamespace::ProjectRuntimePolicy {
            crate::validate_project_runtime_policy_descriptor(&descriptor)?;
        }
        let max_bytes = match descriptor.namespace {
            ObjectNamespace::ForeignPolicy => limits.max_object_bytes.min(MAX_FOREIGN_POLICY_BYTES),
            ObjectNamespace::ProjectRuntimePolicy => limits
                .max_object_bytes
                .min(MAX_PROJECT_RUNTIME_POLICY_BYTES),
            _ => limits.max_object_bytes,
        };
        let bytes = read_bytes(&mut reader, max_bytes, &mut total)?;
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
