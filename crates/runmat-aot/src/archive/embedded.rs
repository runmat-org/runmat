use crate::{AotError, AotResult};

use super::{RuntimeArchive, RuntimeArchiveManifest, RuntimePackManifest};

mod generated {
    include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
}

/// Expected compile-time pack identity shipped with this CLI, whether the
/// archive bytes are embedded or distributed separately.
pub fn embedded_runtime_manifest() -> AotResult<Option<RuntimeArchiveManifest>> {
    if generated::PAYLOAD.is_none() {
        return embedded_runtime_pack_manifest().map(|pack| pack.map(|pack| pack.runtime));
    }
    generated::MANIFEST
        .map(|manifest| {
            let manifest =
                RuntimeArchiveManifest::from_json(manifest.as_bytes()).map_err(|error| {
                    AotError::contract(
                        "aot.archive.manifest",
                        format!("embedded runtime archive manifest is invalid: {error}"),
                    )
                })?;
            manifest.validate()?;
            Ok(manifest)
        })
        .transpose()
}

/// Trusted distribution metadata in a light CLI. Embedded-archive builds keep
/// their original manifest format and do not acquire an external pack.
pub fn embedded_runtime_pack_manifest() -> AotResult<Option<RuntimePackManifest>> {
    if generated::PAYLOAD.is_some() {
        return Ok(None);
    }
    generated::MANIFEST
        .map(|manifest| {
            let manifest = RuntimePackManifest::from_json(manifest.as_bytes())?;
            manifest.validate()?;
            Ok(manifest)
        })
        .transpose()
}

pub fn embedded_runtime_archive() -> AotResult<Option<RuntimeArchive>> {
    let Some(payload) = generated::PAYLOAD else {
        return Ok(None);
    };
    let manifest = embedded_runtime_manifest()?.ok_or_else(|| {
        AotError::contract(
            "aot.archive.manifest",
            "embedded runtime archive has no manifest",
        )
    })?;
    RuntimeArchive::new(manifest, payload.to_vec()).map(Some)
}
