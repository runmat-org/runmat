use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use runmat_execution_artifact::{LogicalObject, ObjectNamespace};
use runmat_native_ffi::{
    NativeInterfaceArtifactBundle, NativeInterfaceArtifactBundleEntry,
    NativeInterfaceArtifactManifest, NATIVE_INTERFACE_MANIFEST_MEDIA_TYPE,
    NATIVE_LIBRARY_MEDIA_TYPE,
};
use runmat_package::{ContentDigest, FrozenProject};
use runmat_types::InteropManifest;

pub(crate) struct PreparedNativeInterfaces {
    pub(crate) interop: InteropManifest,
    pub(crate) objects: Vec<LogicalObject>,
    pub(crate) bundle: NativeInterfaceArtifactBundle,
}

impl PreparedNativeInterfaces {
    pub(crate) fn empty() -> Self {
        Self {
            interop: InteropManifest::empty(),
            objects: Vec::new(),
            bundle: NativeInterfaceArtifactBundle::empty(),
        }
    }
}

pub(crate) fn prepare_native_interfaces(
    project: &FrozenProject,
) -> Result<PreparedNativeInterfaces> {
    let mut by_name = BTreeMap::new();
    let mut by_identity = BTreeMap::new();
    for declaration in &project.native_interfaces {
        let manifest_bytes = read_exact(
            &declaration.manifest_path,
            &declaration.manifest_digest,
            "manifest",
            &declaration.name,
        )?;
        let library_bytes = read_exact(
            &declaration.library_path,
            &declaration.library_digest,
            "library",
            &declaration.name,
        )?;
        let manifest = NativeInterfaceArtifactManifest::from_canonical_bytes(&manifest_bytes)
            .with_context(|| {
                format!(
                    "native interface `{}` has an invalid prepared manifest",
                    declaration.name
                )
            })?;
        if manifest.interface_name != declaration.name {
            bail!(
                "native interface declaration `{}` names prepared interface `{}`",
                declaration.name,
                manifest.interface_name
            );
        }
        manifest.validate_library(&library_bytes).with_context(|| {
            format!(
                "native interface `{}` library does not match its prepared manifest",
                declaration.name
            )
        })?;
        if let Some(owner) = by_name.insert(
            declaration.name.clone(),
            declaration.package_instance.clone(),
        ) {
            bail!(
                "native interface name `{}` is declared by package instances {} and {}",
                declaration.name,
                owner,
                declaration.package_instance
            );
        }
        let identity = manifest.identity.to_string();
        if by_identity
            .insert(identity.clone(), (manifest_bytes, library_bytes))
            .is_some()
        {
            bail!("native interface identity `{identity}` is declared more than once");
        }
    }

    let mut objects = Vec::with_capacity(by_identity.len() * 2);
    let mut bundle_entries = Vec::with_capacity(by_identity.len());
    let mut artifact_identities = Vec::with_capacity(by_identity.len());
    for (identity, (manifest, library)) in by_identity {
        bundle_entries.push(NativeInterfaceArtifactBundleEntry {
            manifest: manifest.clone(),
            library: library.clone(),
        });
        let logical_root = format!("native/{}", logical_identity(&identity));
        objects.push(LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            format!("{logical_root}/manifest.json"),
            NATIVE_INTERFACE_MANIFEST_MEDIA_TYPE,
            manifest,
        )?);
        objects.push(LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            format!("{logical_root}/library.bin"),
            NATIVE_LIBRARY_MEDIA_TYPE,
            library,
        )?);
        artifact_identities.push(identity);
    }

    let mut interop = InteropManifest::empty();
    if !artifact_identities.is_empty() {
        let first = NativeInterfaceArtifactManifest::from_canonical_bytes(&objects[0].bytes)?;
        let mut requirement = first
            .interop_manifest()
            .adapters
            .into_iter()
            .next()
            .expect("native interface manifest declares its adapter");
        requirement.artifact_identities = artifact_identities;
        interop.adapters.push(requirement);
    }
    interop
        .validate()
        .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
    let bundle = NativeInterfaceArtifactBundle::new(bundle_entries)?;
    Ok(PreparedNativeInterfaces {
        interop,
        objects,
        bundle,
    })
}

fn read_exact(
    path: &std::path::Path,
    expected: &ContentDigest,
    kind: &str,
    interface: &str,
) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).with_context(|| {
        format!(
            "read native interface `{interface}` {kind} `{}`",
            path.display()
        )
    })?;
    let actual = ContentDigest::sha256(&bytes);
    if &actual != expected {
        bail!(
            "native interface `{interface}` {kind} changed after project resolution: expected {expected}, found {actual}"
        );
    }
    Ok(bytes)
}

fn logical_identity(identity: &str) -> String {
    identity
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' {
                char::from(byte)
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_identity_is_a_portable_path_component() {
        assert_eq!(
            logical_identity("native-ffi:v1:sha256/example"),
            "native-ffi_v1_sha256_example"
        );
        assert!(logical_identity(runmat_native_ffi::NATIVE_FFI_ADAPTER_ID)
            .chars()
            .all(|value| { value.is_ascii_alphanumeric() || value == '-' || value == '_' }));
    }

    #[test]
    fn prepared_interop_identity_set_is_unique() {
        let identities = std::collections::BTreeSet::from(["a", "b"]);
        assert_eq!(identities.len(), 2);
    }
}
