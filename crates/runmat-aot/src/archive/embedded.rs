use crate::{AotError, AotResult};

use super::{RuntimeArchive, RuntimeArchiveManifest};

mod generated {
    include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
}

pub fn embedded_runtime_archive() -> AotResult<Option<RuntimeArchive>> {
    let (Some(payload), Some(manifest)) = (generated::PAYLOAD, generated::MANIFEST) else {
        return Ok(None);
    };
    let manifest: RuntimeArchiveManifest = serde_json::from_str(manifest).map_err(|error| {
        AotError::contract(
            "aot.archive.manifest",
            format!("embedded runtime archive manifest is invalid: {error}"),
        )
    })?;
    RuntimeArchive::new(manifest, payload.to_vec()).map(Some)
}
