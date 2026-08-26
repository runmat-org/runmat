use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use runmat_execution_artifact::LogicalObject;
use runmat_native_ffi::{
    NativeInterfaceArtifactManifest, NATIVE_INTERFACE_MANIFEST_MEDIA_TYPE,
    NATIVE_LIBRARY_MEDIA_TYPE,
};

use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Debug)]
pub(crate) struct MaterializedNativeInterface {
    pub(crate) manifest: NativeInterfaceArtifactManifest,
    pub(crate) manifest_path: PathBuf,
    pub(crate) library_path: PathBuf,
}

pub(crate) fn discover(
    objects: &[LogicalObject],
    root: &Path,
) -> NativeExecutionResult<Vec<MaterializedNativeInterface>> {
    let libraries = objects
        .iter()
        .filter(|object| object.descriptor.media_type == NATIVE_LIBRARY_MEDIA_TYPE)
        .collect::<Vec<_>>();
    let mut referenced_libraries = BTreeSet::new();
    let mut interfaces = BTreeMap::new();

    for sidecar in objects
        .iter()
        .filter(|object| object.descriptor.media_type == NATIVE_INTERFACE_MANIFEST_MEDIA_TYPE)
    {
        let manifest = NativeInterfaceArtifactManifest::from_canonical_bytes(&sidecar.bytes)
            .map_err(protocol)?;
        let matching = libraries
            .iter()
            .copied()
            .filter(|library| {
                library.descriptor.encoded_length == manifest.library_bytes
                    && manifest.validate_library(&library.bytes).is_ok()
            })
            .collect::<Vec<_>>();
        let [library] = matching.as_slice() else {
            return Err(protocol(format!(
                "native interface {} must resolve to exactly one bundled library; found {}",
                manifest.identity,
                matching.len()
            )));
        };
        manifest
            .validate_current_library(&library.bytes)
            .map_err(protocol)?;
        referenced_libraries.insert(library.descriptor.logical_name.clone());
        let identity = manifest.identity.to_string();
        let materialized = MaterializedNativeInterface {
            manifest,
            manifest_path: root.join(&sidecar.descriptor.logical_name),
            library_path: root.join(&library.descriptor.logical_name),
        };
        if interfaces.insert(identity.clone(), materialized).is_some() {
            return Err(protocol(format!(
                "native interface identity {identity} appears more than once"
            )));
        }
    }

    for library in libraries {
        if !referenced_libraries.contains(&library.descriptor.logical_name) {
            return Err(protocol(format!(
                "native library {} has no prepared interface manifest",
                library.descriptor.logical_name
            )));
        }
    }
    Ok(interfaces.into_values().collect())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
