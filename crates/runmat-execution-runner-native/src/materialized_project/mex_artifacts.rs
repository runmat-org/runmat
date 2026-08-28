use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use runmat_execution_artifact::LogicalObject;
use runmat_mex::{MexArtifactManifest, MEX_ARTIFACT_MANIFEST_MEDIA_TYPE, MEX_MODULE_MEDIA_TYPE};

use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Debug)]
pub(crate) struct MaterializedMexArtifact {
    pub(crate) manifest: MexArtifactManifest,
    pub(crate) module_path: PathBuf,
}

pub(crate) fn discover(
    objects: &[LogicalObject],
    root: &Path,
) -> NativeExecutionResult<Vec<MaterializedMexArtifact>> {
    let modules = objects
        .iter()
        .filter(|object| object.descriptor.media_type == MEX_MODULE_MEDIA_TYPE)
        .collect::<Vec<_>>();
    let mut referenced_modules = BTreeSet::new();
    let mut artifacts = BTreeMap::new();
    for sidecar in objects
        .iter()
        .filter(|object| object.descriptor.media_type == MEX_ARTIFACT_MANIFEST_MEDIA_TYPE)
    {
        let manifest =
            MexArtifactManifest::from_canonical_bytes(&sidecar.bytes).map_err(protocol)?;
        let matching = modules
            .iter()
            .copied()
            .filter(|module| manifest.validate_current_module(&module.bytes).is_ok())
            .collect::<Vec<_>>();
        let [module] = matching.as_slice() else {
            return Err(protocol(format!(
                "MEX artifact {} must resolve to exactly one bundled module; found {}",
                manifest.identity,
                matching.len()
            )));
        };
        let module_path = root.join(&module.descriptor.logical_name);
        if MexArtifactManifest::path_for_module(&module_path)
            != root.join(&sidecar.descriptor.logical_name)
        {
            return Err(protocol(format!(
                "MEX artifact {} manifest is not adjacent to its module",
                manifest.identity
            )));
        }
        referenced_modules.insert(module.descriptor.logical_name.clone());
        let identity = manifest.identity.to_string();
        if artifacts
            .insert(
                identity.clone(),
                MaterializedMexArtifact {
                    manifest,
                    module_path,
                },
            )
            .is_some()
        {
            return Err(protocol(format!(
                "MEX artifact identity {identity} appears more than once"
            )));
        }
    }
    for module in modules {
        if !referenced_modules.contains(&module.descriptor.logical_name) {
            return Err(protocol(format!(
                "MEX module {} has no prepared artifact manifest",
                module.descriptor.logical_name
            )));
        }
    }
    Ok(artifacts.into_values().collect())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
