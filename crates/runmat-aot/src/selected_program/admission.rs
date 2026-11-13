//! Operator-issued selected-target expectations. This file is a caller grant,
//! not part of the target pack and never populated from its own declarations.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::archive::RuntimeArchiveCapabilities;
use crate::installation::{ExternalNativeInstallationAdmission, NativeInstallationAdmissionRecord};
use crate::link::{
    ExternalNativeLinkAdmission, HostedNativeBuildTrees, HostedNativeTree, NativeHostedDependency,
    NativeLinkAdapter, NativeLinkExecutionHost, NativeStorageGrant,
};
use anyhow::{anyhow, bail, Context, Result};
use runmat_execution::host::{
    NativeAbiBinding, NativeDeploymentProfile, NativeDeploymentServiceKind, NativeNumericalPolicy,
    SelectedNativeTarget, TargetPackId,
};
use runmat_execution::Digest;
use runmat_execution_artifact::target_pack::{
    ComponentId, ExternalNumericalAdmission, TargetPackManifest, TargetPackServiceKind,
    MAX_COMPONENT_BYTES, MAX_PACK_BYTES,
};
use runmat_package::NormalizedRelativePath;
use runmat_package_cache::ArchiveLimits;
use runmat_types::CapabilitySet;

const MAX_ADMISSION_BYTES: u64 = 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;

pub struct SelectedPaths {
    pub manifest: PathBuf,
    pub components: PathBuf,
    pub host_admission: PathBuf,
}

impl SelectedPaths {
    pub fn from_options(
        manifest: Option<PathBuf>,
        components: Option<PathBuf>,
        host_admission: Option<PathBuf>,
    ) -> Result<Option<Self>> {
        match (manifest, components, host_admission) {
            (None, None, None) => Ok(None),
            (Some(manifest), Some(components), Some(host_admission)) => Ok(Some(Self {
                manifest,
                components,
                host_admission,
            })),
            _ => bail!("selected compilation needs pack manifest, component root and independent host admission together"),
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedHostAdmission {
    pub schema_version: u16,
    pub expected_target: SelectedNativeTarget,
    pub expected_abi: NativeAbiBinding,
    pub numerical: NativeNumericalPolicy,
    pub profile: NativeDeploymentProfile,
    pub platform_service_abi: Digest,
    pub service_abis: Vec<ServiceAbi>,
    pub authorized_runtime_capabilities: CapabilitySet,
    pub additional_components: Vec<ComponentId>,
    pub numerical_qualification: Option<NumericalQualification>,
    pub runtime_capabilities: RuntimeArchiveCapabilities,
    /// Invocation inputs supplied by the caller, independently of pack metadata.
    pub clock_inputs: Option<runmat_execution::host::NativeClockInputs>,
    pub local_service_inputs: Option<runmat_execution::host::NativeLocalServiceInputs>,
    pub link: QualifiedLink,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceAbi {
    pub kind: TargetPackServiceKind,
    pub abi: Digest,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalQualification {
    pub pack: TargetPackId,
    pub target: Digest,
    pub backend: Digest,
    pub policy: Digest,
    pub evidence: Digest,
}

#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum LinkAdapterInput {
    ClangDriver { linker: NormalizedRelativePath },
    MsvcLinker,
}

#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum HostedDependencyInput {
    Library {
        name: String,
        path: NormalizedRelativePath,
    },
    Framework {
        name: String,
        path: NormalizedRelativePath,
    },
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StorageInput {
    archive: ArchiveLimits,
    metadata_bytes: u64,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BlobInput {
    component: ComponentId,
    path: PathBuf,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct HostedTreeInput {
    metadata: PathBuf,
    blobs: Vec<BlobInput>,
}

impl HostedTreeInput {
    fn resolve_against(&mut self, base: &Path) {
        if self.metadata.is_relative() {
            self.metadata = base.join(&self.metadata);
        }
        for blob in &mut self.blobs {
            if blob.path.is_relative() {
                blob.path = base.join(&blob.path);
            }
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedLink {
    adapter: LinkAdapterInput,
    entrypoint: NormalizedRelativePath,
    toolchain: Digest,
    sysroot: Digest,
    qualification: Digest,
    storage: StorageInput,
    hosted_dependencies: Vec<HostedDependencyInput>,
    hosted_toolchain: Option<HostedTreeInput>,
    hosted_sysroot: Option<HostedTreeInput>,
}

impl QualifiedLink {
    pub fn storage_grant(&self) -> NativeStorageGrant {
        NativeStorageGrant {
            archive: self.storage.archive,
            metadata_bytes: self.storage.metadata_bytes,
        }
    }
}

pub struct LoadedHostedTree {
    metadata: Vec<u8>,
    blobs: BTreeMap<ComponentId, Vec<u8>>,
}

impl LoadedHostedTree {
    fn borrow(&self) -> HostedNativeTree<'_> {
        HostedNativeTree {
            metadata: &self.metadata,
            blobs: &self.blobs,
        }
    }
}

pub struct LoadedLink {
    pub admission: ExternalNativeLinkAdmission,
    toolchain: Option<LoadedHostedTree>,
    sysroot: Option<LoadedHostedTree>,
}

impl LoadedLink {
    pub fn hosted(&self) -> HostedNativeBuildTrees<'_> {
        HostedNativeBuildTrees {
            toolchain: self.toolchain.as_ref().map(LoadedHostedTree::borrow),
            sysroot: self.sysroot.as_ref().map(LoadedHostedTree::borrow),
        }
    }
}

impl SelectedHostAdmission {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = read_bounded(path, MAX_ADMISSION_BYTES)?;
        let mut admission: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("decode independent host admission `{}`", path.display()))?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        if let Some(tree) = admission.link.hosted_toolchain.as_mut() {
            tree.resolve_against(base);
        }
        if let Some(tree) = admission.link.hosted_sysroot.as_mut() {
            tree.resolve_against(base);
        }
        if admission.schema_version != 1 {
            bail!("unsupported selected host admission schema");
        }
        admission.expected_target.validate()?;
        admission.expected_abi.validate()?;
        admission.numerical.validate()?;
        if let Some(inputs) = &admission.clock_inputs {
            inputs.validate()?;
        }
        if let Some(inputs) = &admission.local_service_inputs {
            inputs.fingerprint()?;
        }
        admission
            .link
            .storage_grant()
            .archive
            .validate()
            .map_err(anyhow::Error::msg)?;
        if admission.link.storage.metadata_bytes == 0 {
            bail!("selected host metadata grant must be nonzero");
        }
        if admission
            .additional_components
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || admission
                .service_abis
                .windows(2)
                .any(|pair| pair[0].kind >= pair[1].kind)
        {
            bail!("selected host components and service ABIs must be sorted and unique");
        }
        Ok(admission)
    }

    pub fn service_abis(&self) -> BTreeMap<NativeDeploymentServiceKind, Digest> {
        self.service_abis
            .iter()
            .map(|service| (service.kind, service.abi))
            .collect()
    }

    pub fn numerical_qualification(&self) -> Option<ExternalNumericalAdmission> {
        self.numerical_qualification
            .as_ref()
            .map(|value| ExternalNumericalAdmission {
                pack: value.pack,
                target: value.target,
                backend: value.backend,
                policy: value.policy,
                evidence: value.evidence,
            })
    }

    pub fn load_link(&self, product: Digest) -> Result<LoadedLink> {
        let storage = self.link.storage_grant();
        storage.archive.validate().map_err(anyhow::Error::msg)?;
        let mut hosted_dependencies = BTreeMap::new();
        for dependency in &self.link.hosted_dependencies {
            let (key, path) = match dependency {
                HostedDependencyInput::Library { name, path } => {
                    (NativeHostedDependency::Library(name.clone()), path.clone())
                }
                HostedDependencyInput::Framework { name, path } => (
                    NativeHostedDependency::Framework(name.clone()),
                    path.clone(),
                ),
            };
            if hosted_dependencies.insert(key, path).is_some() {
                bail!("duplicate qualified hosted dependency locator");
            }
        }
        let admission = ExternalNativeLinkAdmission {
            product,
            execution_host: NativeLinkExecutionHost::current()?,
            adapter: match &self.link.adapter {
                LinkAdapterInput::ClangDriver { linker } => NativeLinkAdapter::ClangDriver {
                    linker: linker.clone(),
                },
                LinkAdapterInput::MsvcLinker => NativeLinkAdapter::MsvcLinker,
            },
            entrypoint: self.link.entrypoint.clone(),
            toolchain: self.link.toolchain,
            sysroot: self.link.sysroot,
            qualification: self.link.qualification,
            storage,
            hosted_dependencies,
        };
        let toolchain = self
            .link
            .hosted_toolchain
            .as_ref()
            .map(|tree| load_hosted_tree(tree, &storage))
            .transpose()?;
        let sysroot = self
            .link
            .hosted_sysroot
            .as_ref()
            .map(|tree| load_hosted_tree(tree, &storage))
            .transpose()?;
        Ok(LoadedLink {
            admission,
            toolchain,
            sysroot,
        })
    }
}

fn load_hosted_tree(
    tree: &HostedTreeInput,
    grant: &NativeStorageGrant,
) -> Result<LoadedHostedTree> {
    let metadata = read_bounded(
        &tree.metadata,
        grant.metadata_bytes.min(MAX_COMPONENT_BYTES),
    )?;
    let mut blobs = BTreeMap::new();
    let mut total = metadata.len() as u64;
    for blob in &tree.blobs {
        let bytes = read_bounded(&blob.path, grant.archive.max_file_bytes)?;
        total = total
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| anyhow!("hosted tree byte extent overflow"))?;
        if total > grant.archive.max_expanded_bytes {
            bail!("hosted tree exceeds independent caller byte grant");
        }
        if blobs.insert(blob.component.clone(), bytes).is_some() {
            bail!("duplicate hosted tree blob identity");
        }
    }
    Ok(LoadedHostedTree { metadata, blobs })
}

pub fn load_manifest(path: &Path) -> Result<TargetPackManifest> {
    let bytes = read_bounded(path, MAX_MANIFEST_BYTES)?;
    TargetPackManifest::from_canonical_bytes(&bytes).map_err(anyhow::Error::msg)
}

/// Decode an independently issued deployment admission. This trusted operator
/// input is never inferred from a persisted link receipt or target pack.
pub(super) fn load_installation_admission(
    path: &Path,
) -> Result<ExternalNativeInstallationAdmission> {
    let bytes = read_bounded(path, MAX_ADMISSION_BYTES)?;
    NativeInstallationAdmissionRecord::from_canonical_bytes(&bytes)
        .with_context(|| format!("decode installation admission `{}`", path.display()))
        .map(NativeInstallationAdmissionRecord::into_external)
}

pub fn read_component(
    root: &Path,
    manifest: &TargetPackManifest,
    id: &ComponentId,
    grant: &NativeStorageGrant,
) -> Result<Vec<u8>> {
    let descriptor = manifest
        .components
        .iter()
        .find(|entry| &entry.id == id)
        .ok_or_else(|| anyhow!("selected component `{}` has no descriptor", id.as_str()))?;
    if descriptor.object.encoded_length > MAX_COMPONENT_BYTES {
        bail!("selected component exceeds canonical byte bound");
    }
    let canonical_root = std::fs::canonicalize(root)
        .with_context(|| format!("resolve component root `{}`", root.display()))?;
    let path = canonical_root.join(id.as_str());
    let canonical_path = std::fs::canonicalize(&path)
        .with_context(|| format!("resolve component `{}`", id.as_str()))?;
    if !canonical_path.starts_with(&canonical_root) || !canonical_path.is_file() {
        bail!("selected component path escapes its explicit root or is not a file");
    }
    let bytes = read_bounded(
        &canonical_path,
        descriptor
            .object
            .encoded_length
            .min(grant.archive.max_file_bytes),
    )?;
    if bytes.len() as u64 != descriptor.object.encoded_length
        || Digest::sha256(&bytes) != descriptor.object.digest
    {
        bail!(
            "selected component `{}` differs from its exact descriptor",
            id.as_str()
        );
    }
    Ok(bytes)
}

pub fn read_selected_components(
    root: &Path,
    manifest: &TargetPackManifest,
    ids: &[ComponentId],
    grant: &NativeStorageGrant,
) -> Result<BTreeMap<ComponentId, Vec<u8>>> {
    let mut result = BTreeMap::new();
    let mut total = 0u64;
    for id in ids {
        let bytes = read_component(root, manifest, id, grant)?;
        total = total
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| anyhow!("selected pack byte extent overflow"))?;
        if total > MAX_PACK_BYTES || total > grant.archive.max_expanded_bytes {
            bail!("selected pack exceeds canonical or independently granted byte bound");
        }
        result.insert(id.clone(), bytes);
    }
    Ok(result)
}

pub fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path).with_context(|| format!("open `{}`", path.display()))?;
    if file.metadata()?.len() > maximum {
        bail!("`{}` exceeds its admitted byte bound", path.display());
    }
    let mut bytes = Vec::new();
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("read `{}`", path.display()))?;
    if bytes.len() as u64 > maximum {
        bail!(
            "`{}` changed beyond its admitted byte bound",
            path.display()
        );
    }
    Ok(bytes)
}
