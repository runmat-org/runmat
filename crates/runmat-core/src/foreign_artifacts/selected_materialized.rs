//! Unit-scoped private bytes and typed foreign identities for selected execution.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use runmat_execution_artifact::{ExecutionBundle, LogicalObject, ObjectNamespace};
use runmat_java::{JavaArtifactIdentity, JAVA_ARCHIVE_MEDIA_TYPE};
use runmat_mex::{
    MexArtifactManifest, MexSourceLanguage, MEX_ARTIFACT_MANIFEST_MEDIA_TYPE, MEX_MODULE_MEDIA_TYPE,
};
use runmat_native_ffi::{
    NativeInterfaceArtifactManifest, NATIVE_INTERFACE_MANIFEST_MEDIA_TYPE,
    NATIVE_LIBRARY_MEDIA_TYPE,
};
use runmat_python::{PythonArtifactBundle, PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE};
use runmat_types::ForeignArtifactIdentity;

use super::{MaterializedForeignView, PrivateArtifactPath, PrivateArtifactRoot};

/// Exact selected foreign objects, with no project path, session, or runtime port.
/// Retain this owner until the selected host has finished delayed callbacks and
/// shutdown. The object order is the authenticated bundle order (Java classpath).
pub struct SelectedForeignMaterialization {
    _root: PrivateArtifactRoot,
    parent_owned: bool,
    native: BTreeMap<ForeignArtifactIdentity, (PrivateArtifactPath, PrivateArtifactPath)>,
    mex: BTreeMap<ForeignArtifactIdentity, (PrivateArtifactPath, PrivateArtifactPath, bool)>,
    java: Vec<(JavaArtifactIdentity, PrivateArtifactPath)>,
    python: Option<PythonArtifactBundle>,
    policy: Option<runmat_config::runtime::ForeignConfig>,
}

impl SelectedForeignMaterialization {
    pub(crate) fn private_byte_lease(&self) -> runmat_execution_artifact::PrivateArtifactLease {
        self._root.lease()
    }
    pub(crate) fn snapshot_parent(
        &self,
    ) -> Result<runmat_process_host::shared_memory::SharedSnapshotParent, String> {
        if !self.parent_owned {
            return Err("joined native host requires parent-owned selected bytes".into());
        }
        runmat_process_host::shared_memory::SharedSnapshotParent::admit(self._root.path())
            .map_err(|error| error.to_string())
    }

    pub(crate) fn persist_unverified(&self, reason: &str) -> Result<(), String> {
        self._root.lease().persist_unverified(None, reason)
    }

    pub fn from_bundle(bundle: &ExecutionBundle) -> Result<Self, String> {
        Self::from_bundle_with_root(
            bundle,
            PrivateArtifactRoot::new("runmat-selected-foreign-")?,
            false,
        )
    }

    /// Child-side selected bytes under the parent's one-shot attempt owner.
    /// Dropping this value cannot erase paths before the parent reaps a child
    /// with retained MEX callbacks.
    pub fn from_bundle_in_parent_scratch(
        bundle: &ExecutionBundle,
        scratch: &Path,
        unit_index: usize,
    ) -> Result<Self, String> {
        Self::from_bundle_with_root(
            bundle,
            PrivateArtifactRoot::in_parent_scratch(scratch, unit_index)?,
            true,
        )
    }

    fn from_bundle_with_root(
        bundle: &ExecutionBundle,
        mut root: PrivateArtifactRoot,
        parent_owned: bool,
    ) -> Result<Self, String> {
        bundle.validate().map_err(|error| error.to_string())?;
        let [recipe] = bundle.manifest.recipes.as_slice() else {
            return Err("selected foreign materialization requires one exact recipe".into());
        };
        let policy = match recipe.foreign_policy.as_ref() {
            Some(descriptor) => {
                let object = bundle
                    .objects
                    .iter()
                    .find(|object| &object.descriptor == descriptor)
                    .ok_or("selected foreign policy object is absent")?;
                Some(super::policy::decode_admitted_policy(object)?)
            }
            None if recipe.interop == runmat_types::InteropManifest::empty() => None,
            None => return Err("selected foreign recipe has no policy".into()),
        };
        let foreign = bundle
            .objects
            .iter()
            .filter(|object| object.descriptor.namespace == ObjectNamespace::ForeignArtifact)
            .collect::<Vec<_>>();
        let mut paths = BTreeMap::new();
        for object in &foreign {
            let path = root.write_object(object)?;
            if paths
                .insert(object.descriptor.logical_name.as_str(), path)
                .is_some()
            {
                return Err("selected foreign bundle repeats a logical object path".into());
            }
        }
        let native = discover_native(&foreign, &paths)?;
        let mex = discover_mex(&foreign, &paths)?;
        let java = discover_java(&foreign, &paths)?;
        let python = discover_python(&foreign)?;
        Ok(Self {
            _root: root,
            parent_owned,
            native,
            mex,
            java,
            python,
            policy,
        })
    }
}

impl MaterializedForeignView for SelectedForeignMaterialization {
    fn private_byte_lease(&self) -> runmat_execution_artifact::PrivateArtifactLease {
        self._root.lease()
    }

    fn foreign_policy(&self) -> Option<&runmat_config::runtime::ForeignConfig> {
        self.policy.as_ref()
    }
    fn native_interface(
        &self,
        identity: &ForeignArtifactIdentity,
    ) -> Option<(&PrivateArtifactPath, &PrivateArtifactPath)> {
        self.native
            .get(identity)
            .map(|(library, manifest)| (library, manifest))
    }

    fn mex_module(
        &self,
        identity: &ForeignArtifactIdentity,
    ) -> Option<(&PrivateArtifactPath, &PrivateArtifactPath, bool)> {
        self.mex
            .get(identity)
            .map(|(module, manifest, cuda)| (module, manifest, *cuda))
    }

    fn java_archives(&self) -> Vec<(JavaArtifactIdentity, &PrivateArtifactPath)> {
        self.java
            .iter()
            .map(|(identity, path)| (identity.clone(), path))
            .collect()
    }

    fn python_bundle(&self) -> Option<&PythonArtifactBundle> {
        self.python.as_ref()
    }
}

fn discover_native(
    objects: &[&LogicalObject],
    paths: &BTreeMap<&str, PrivateArtifactPath>,
) -> Result<BTreeMap<ForeignArtifactIdentity, (PrivateArtifactPath, PrivateArtifactPath)>, String> {
    let libraries = objects
        .iter()
        .copied()
        .filter(|object| object.descriptor.media_type == NATIVE_LIBRARY_MEDIA_TYPE)
        .collect::<Vec<_>>();
    let mut referenced = BTreeSet::new();
    let mut selected = BTreeMap::new();
    for sidecar in objects
        .iter()
        .copied()
        .filter(|object| object.descriptor.media_type == NATIVE_INTERFACE_MANIFEST_MEDIA_TYPE)
    {
        let manifest = NativeInterfaceArtifactManifest::from_canonical_bytes(&sidecar.bytes)
            .map_err(|error| error.to_string())?;
        let matches = libraries
            .iter()
            .copied()
            .filter(|library| {
                library.descriptor.encoded_length == manifest.library_bytes
                    && manifest.validate_library(&library.bytes).is_ok()
            })
            .collect::<Vec<_>>();
        let [library] = matches.as_slice() else {
            return Err(format!(
                "native interface {} must resolve to one selected library",
                manifest.identity
            ));
        };
        manifest
            .validate_current_library(&library.bytes)
            .map_err(|error| error.to_string())?;
        referenced.insert(library.descriptor.logical_name.as_str());
        let pair = (
            paths[library.descriptor.logical_name.as_str()].clone(),
            paths[sidecar.descriptor.logical_name.as_str()].clone(),
        );
        let identity = ForeignArtifactIdentity::new(manifest.identity.as_str())
            .map_err(|error| error.to_string())?;
        if selected.insert(identity, pair).is_some() {
            return Err("selected native interface identity is duplicated".into());
        }
    }
    if libraries
        .iter()
        .any(|library| !referenced.contains(library.descriptor.logical_name.as_str()))
    {
        return Err("selected native library has no interface manifest".into());
    }
    Ok(selected)
}

fn discover_mex(
    objects: &[&LogicalObject],
    paths: &BTreeMap<&str, PrivateArtifactPath>,
) -> Result<
    BTreeMap<ForeignArtifactIdentity, (PrivateArtifactPath, PrivateArtifactPath, bool)>,
    String,
> {
    let modules = objects
        .iter()
        .copied()
        .filter(|object| object.descriptor.media_type == MEX_MODULE_MEDIA_TYPE)
        .collect::<Vec<_>>();
    let mut referenced = BTreeSet::new();
    let mut selected = BTreeMap::new();
    for sidecar in objects
        .iter()
        .copied()
        .filter(|object| object.descriptor.media_type == MEX_ARTIFACT_MANIFEST_MEDIA_TYPE)
    {
        let manifest = MexArtifactManifest::from_canonical_bytes(&sidecar.bytes)
            .map_err(|error| error.to_string())?;
        let matches = modules
            .iter()
            .copied()
            .filter(|module| manifest.validate_current_module(&module.bytes).is_ok())
            .collect::<Vec<_>>();
        let [module] = matches.as_slice() else {
            return Err(format!(
                "MEX artifact {} must resolve to one selected module",
                manifest.identity
            ));
        };
        let module_path = paths[module.descriptor.logical_name.as_str()].as_path();
        let sidecar_path = paths[sidecar.descriptor.logical_name.as_str()].as_path();
        if MexArtifactManifest::path_for_module(module_path) != sidecar_path {
            return Err("selected MEX manifest is not adjacent to its module".into());
        }
        referenced.insert(module.descriptor.logical_name.as_str());
        let identity = ForeignArtifactIdentity::new(manifest.identity.as_str())
            .map_err(|error| error.to_string())?;
        if selected
            .insert(
                identity,
                (
                    paths[module.descriptor.logical_name.as_str()].clone(),
                    paths[sidecar.descriptor.logical_name.as_str()].clone(),
                    manifest.source_language == MexSourceLanguage::Cuda,
                ),
            )
            .is_some()
        {
            return Err("selected MEX artifact identity is duplicated".into());
        }
    }
    if modules
        .iter()
        .any(|module| !referenced.contains(module.descriptor.logical_name.as_str()))
    {
        return Err("selected MEX module has no artifact manifest".into());
    }
    Ok(selected)
}

fn discover_java(
    objects: &[&LogicalObject],
    paths: &BTreeMap<&str, PrivateArtifactPath>,
) -> Result<Vec<(JavaArtifactIdentity, PrivateArtifactPath)>, String> {
    let mut identities = BTreeSet::new();
    let mut selected = Vec::new();
    for object in objects
        .iter()
        .copied()
        .filter(|object| object.descriptor.media_type == JAVA_ARCHIVE_MEDIA_TYPE)
    {
        let identity = JavaArtifactIdentity::for_bytes(&object.bytes);
        if !identities.insert(identity.clone()) {
            return Err("selected Java artifact identity is duplicated".into());
        }
        selected.push((
            identity,
            paths[object.descriptor.logical_name.as_str()].clone(),
        ));
    }
    Ok(selected)
}

fn discover_python(objects: &[&LogicalObject]) -> Result<Option<PythonArtifactBundle>, String> {
    let mut bundles = objects
        .iter()
        .copied()
        .filter(|object| object.descriptor.media_type == PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE);
    let Some(object) = bundles.next() else {
        return Ok(None);
    };
    if bundles.next().is_some() {
        return Err("selected unit has more than one Python artifact bundle".into());
    }
    PythonArtifactBundle::from_canonical_bytes(&object.bytes)
        .map(Some)
        .map_err(|error| error.to_string())
}
