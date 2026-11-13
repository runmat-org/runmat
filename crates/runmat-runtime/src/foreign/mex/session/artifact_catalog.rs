//! Validate a complete MEX catalog before publishing any of its names.

use super::{runtime_error, InstalledMexArtifact};
use crate::RuntimeError;
use runmat_mex::MexArtifactManifest;
use std::collections::HashMap;
use std::io::Read as _;
use std::path::Path;

// The same single-object and manifest byte ceilings used for immutable
// executable archive capture. MEX bytes are retained twice while a private
// copy is verified, so admission must happen before allocating either Vec.
pub(super) const MAX_MEX_MODULE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_MEX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

pub(super) fn read_bounded(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>, RuntimeError> {
    let file = std::fs::File::open(path).map_err(|error| {
        runtime_error(
            "MEX:Artifact",
            format!("could not open {label} '{}': {error}", path.display()),
        )
    })?;
    let size = file
        .metadata()
        .map_err(|error| {
            runtime_error(
                "MEX:Artifact",
                format!("could not inspect {label} '{}': {error}", path.display()),
            )
        })?
        .len();
    if size > limit {
        return Err(runtime_error(
            "MEX:Artifact",
            format!("{label} exceeds its {limit}-byte admission limit"),
        ));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            runtime_error(
                "MEX:Artifact",
                format!("could not read {label} '{}': {error}", path.display()),
            )
        })?;
    if u64::try_from(bytes.len()).ok() != Some(size) || size > limit {
        return Err(runtime_error(
            "MEX:Artifact",
            format!("{label} changed size during bounded capture"),
        ));
    }
    Ok(bytes)
}

/// Exact host-frozen declaration, checked against the same bytes used for admission.
pub struct MexArtifactExpectation<'a> {
    pub module: &'a Path,
    pub manifest: &'a Path,
    pub module_digest: runmat_execution::Digest,
    pub manifest_digest: runmat_execution::Digest,
}

/// Prepared metadata only; binaries remain caller-owned and are not loaded.
/// Construction cannot change a live session's resolution state.
pub struct PreparedMexArtifactCatalog {
    pub(super) modules: HashMap<String, InstalledMexArtifact>,
    pub(super) byte_lease: Option<runmat_process_host::private_bytes::PrivateByteLease>,
}

impl PreparedMexArtifactCatalog {
    pub fn with_private_byte_lease(
        mut self,
        lease: runmat_process_host::private_bytes::PrivateByteLease,
    ) -> Self {
        self.byte_lease = Some(lease);
        self
    }

    pub fn prepare<'a>(
        artifacts: impl IntoIterator<Item = MexArtifactExpectation<'a>>,
    ) -> Result<Self, RuntimeError> {
        let mut prepared: HashMap<String, InstalledMexArtifact> = HashMap::new();
        for expected in artifacts {
            let (name, artifact) = prepare_artifact(expected.module, Some(&expected))?;
            if let Some(existing) = prepared.get(&name) {
                if existing.path != artifact.path || existing.identity != artifact.identity {
                    return Err(runtime_error(
                        "MEX:ArtifactConflict",
                        format!("MEX module '{name}' has conflicting project artifacts"),
                    ));
                }
            } else {
                prepared.insert(name, artifact);
            }
        }
        Ok(Self {
            modules: prepared,
            byte_lease: None,
        })
    }

    pub fn interop_manifests(&self) -> impl Iterator<Item = &runmat_types::InteropManifest> {
        self.modules.values().map(|artifact| &artifact.interop)
    }

    /// Validated selected identities for admission before this catalog is
    /// published into the live MEX session.
    pub fn artifact_identities(
        &self,
    ) -> impl Iterator<Item = &runmat_types::ForeignArtifactIdentity> {
        self.modules.values().map(|artifact| &artifact.identity)
    }
}

pub(super) fn prepare_artifact(
    module: &Path,
    expected: Option<&MexArtifactExpectation<'_>>,
) -> Result<(String, InstalledMexArtifact), RuntimeError> {
    let captured = capture_artifact(module, expected)?;
    Ok((captured.name, captured.installed))
}

pub(super) struct CapturedMexArtifact {
    pub name: String,
    pub installed: InstalledMexArtifact,
    pub module_bytes: Vec<u8>,
    pub manifest_bytes: Vec<u8>,
    pub manifest: MexArtifactManifest,
}

pub(super) fn capture_artifact(
    module: &Path,
    expected: Option<&MexArtifactExpectation<'_>>,
) -> Result<CapturedMexArtifact, RuntimeError> {
    let module = std::fs::canonicalize(module).map_err(|error| {
        runtime_error(
            "MEX:Artifact",
            format!(
                "could not resolve MEX artifact '{}': {error}",
                module.display()
            ),
        )
    })?;
    let manifest_path = runmat_mex::MexArtifactManifest::path_for_module(&module);
    if let Some(expected) = expected {
        let declared_manifest = std::fs::canonicalize(expected.manifest).map_err(|error| {
            runtime_error(
                "MEX:Artifact",
                format!("could not resolve declared MEX manifest: {error}"),
            )
        })?;
        if declared_manifest != manifest_path {
            return Err(runtime_error(
                "MEX:Artifact",
                "declared MEX manifest differs from the module sidecar",
            ));
        }
    }
    let manifest_bytes = read_bounded(&manifest_path, MAX_MEX_MANIFEST_BYTES, "MEX manifest")?;
    if let Some(expected) = expected {
        if runmat_execution::Digest::sha256(&manifest_bytes) != expected.manifest_digest {
            return Err(runtime_error(
                "MEX:Artifact",
                "MEX manifest changed after project resolution",
            ));
        }
    }
    let manifest = runmat_mex::MexArtifactManifest::from_canonical_bytes(&manifest_bytes)
        .map_err(|error| runtime_error("MEX:Artifact", error.to_string()))?;
    if manifest.module_bytes > MAX_MEX_MODULE_BYTES {
        return Err(runtime_error(
            "MEX:Artifact",
            "MEX module exceeds its byte admission limit",
        ));
    }
    let bytes = read_bounded(&module, MAX_MEX_MODULE_BYTES, "MEX module")?;
    if let Some(expected) = expected {
        if runmat_execution::Digest::sha256(&bytes) != expected.module_digest {
            return Err(runtime_error(
                "MEX:Artifact",
                "MEX module changed after project resolution",
            ));
        }
    }
    manifest.validate_current_module(&bytes).map_err(|error| {
        runtime_error(
            "MEX:Artifact",
            format!("MEX artifact '{}' is invalid: {error}", module.display()),
        )
    })?;
    if module.file_stem().and_then(|stem| stem.to_str()) != Some(&manifest.module_name) {
        return Err(runtime_error(
            "MEX:Artifact",
            format!(
                "MEX artifact '{}' does not match module name '{}'",
                module.display(),
                manifest.module_name
            ),
        ));
    }
    let installed = InstalledMexArtifact {
        path: module,
        identity: runmat_types::ForeignArtifactIdentity::new(manifest.identity.to_string())
            .map_err(|error| runtime_error("MEX:Artifact", error.to_string()))?,
        interop: manifest.interop_manifest(),
    };
    Ok(CapturedMexArtifact {
        name: manifest.module_name.clone(),
        installed,
        module_bytes: bytes,
        manifest_bytes,
        manifest,
    })
}
