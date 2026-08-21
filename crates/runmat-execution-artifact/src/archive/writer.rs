use std::io::Write;

use crate::{ArtifactError, ArtifactResult, ExecutionBundle};

use super::{
    canonical::write_bytes,
    manifest_codec::{encode_descriptor, encode_manifest},
    ArchiveLimits, MAGIC,
};

pub fn write_bundle(
    bundle: &ExecutionBundle,
    mut writer: impl Write,
    limits: ArchiveLimits,
) -> ArtifactResult<()> {
    bundle.validate()?;
    if bundle.objects.len() > limits.max_objects as usize {
        return Err(ArtifactError::Limit("too many archive objects".into()));
    }
    writer.write_all(MAGIC)?;
    let manifest = encode_manifest(&bundle.manifest)?;
    write_bytes(&mut writer, &manifest, limits.max_manifest_bytes)?;
    writer.write_all(&(bundle.objects.len() as u32).to_be_bytes())?;
    for object in &bundle.objects {
        let descriptor = encode_descriptor(&object.descriptor)?;
        write_bytes(&mut writer, &descriptor, limits.max_manifest_bytes)?;
        write_bytes(&mut writer, &object.bytes, limits.max_object_bytes)?;
    }
    Ok(())
}
