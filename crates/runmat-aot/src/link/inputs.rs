//! One private lease for exact selected build inputs and ordered native bytes.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use runmat_execution_artifact::target_pack::ComponentId;
use runmat_native_codegen::aot::{RelocatableNativeObject, SelectedNativeProduct};

use super::toolchain::ExternalNativeLinkAdmission;
use crate::tree::{physical::write_new, resolve_build_tree, HostedNativeTree, ResolvedNativeTree};
use crate::{archive::RuntimeArchive, AotError, AotResult};

/// Explicit external bytes for a non-hermetic HostedDeclared input. Neither
/// an ambient path nor a matching pack declaration synthesizes these bytes.
#[derive(Clone, Copy, Default)]
pub struct HostedNativeBuildTrees<'a> {
    pub toolchain: Option<HostedNativeTree<'a>>,
    pub sysroot: Option<HostedNativeTree<'a>>,
}

pub(super) struct SelectedLinkInputs<'a> {
    pub lease: tempfile::TempDir,
    pub object: PathBuf,
    pub runtime: PathBuf,
    pub output: PathBuf,
    pub components: BTreeMap<ComponentId, PathBuf>,
    pub toolchain_root: PathBuf,
    pub sysroot_root: PathBuf,
    toolchain: ResolvedNativeTree<'a>,
    sysroot: ResolvedNativeTree<'a>,
}

impl<'a> SelectedLinkInputs<'a> {
    pub fn stage(
        selected: &'a SelectedNativeProduct<'_>,
        object: &RelocatableNativeObject,
        runtime: &RuntimeArchive,
        admission: &ExternalNativeLinkAdmission,
        hosted: HostedNativeBuildTrees<'a>,
        parent: &Path,
    ) -> AotResult<Self> {
        super::plan::validate_pair(object, runtime)?;
        if &object.manifest.product != selected.product() {
            return Err(invalid(
                "object/runtime do not belong to this verified selection",
            ));
        }
        let product = selected.product().selected().expect("selected constructor");
        let contents = selected.contents();
        let manifest = contents.manifest();
        let toolchain = resolve_build_tree(
            contents,
            &manifest.build_inputs.linker,
            hosted.toolchain,
            admission.toolchain,
            &admission.storage,
        )?;
        let sysroot = resolve_build_tree(
            contents,
            &manifest.build_inputs.sysroot,
            hosted.sysroot,
            admission.sysroot,
            &admission.storage,
        )?;
        admission.validate(product.fingerprint().map_err(invalid)?, &toolchain)?;
        admission.storage.file(object.bytes.len() as u64)?;
        let mut bytes = toolchain
            .manifest
            .total_bytes
            .checked_add(sysroot.manifest.total_bytes)
            .and_then(|value| value.checked_add(object.bytes.len() as u64))
            .ok_or_else(|| invalid("selected input extent overflow"))?;
        for id in contents.selection().runtime_components() {
            admission.storage.file(
                contents
                    .component_bytes(id)
                    .ok_or_else(|| invalid("selected bytes are missing"))?
                    .len() as u64,
            )?;
            bytes = bytes
                .checked_add(
                    contents
                        .component_bytes(id)
                        .ok_or_else(|| invalid("selected bytes are missing"))?
                        .len() as u64,
                )
                .ok_or_else(|| invalid("selected input extent overflow"))?;
        }
        let additional_files = contents.selection().runtime_components().len() as u64 + 1;
        let files = toolchain
            .manifest
            .file_count
            .checked_add(sysroot.manifest.file_count)
            .and_then(|count| count.checked_add(additional_files))
            .ok_or_else(|| invalid("selected input count overflow"))?;
        let entries = admission
            .storage
            .admit(&toolchain.manifest)?
            .checked_add(admission.storage.admit(&sysroot.manifest)?)
            .and_then(|count| count.checked_add(additional_files))
            // Private lease, toolchain root, sysroot root, component root.
            .and_then(|count| count.checked_add(4))
            .ok_or_else(|| invalid("selected input count overflow"))?;
        admission.storage.admit_total(entries, files, bytes)?;
        let decoded_runtime = runtime.decode()?;
        if contents.component_bytes(&manifest.runtime_component) != Some(decoded_runtime.as_slice())
        {
            return Err(invalid(
                "runtime payload differs from exact retained component bytes",
            ));
        }
        if !parent.is_dir() {
            return Err(invalid("selected link output directory does not exist"));
        }
        let parent = std::fs::canonicalize(parent).map_err(|error| {
            AotError::io("resolve selected link output directory", parent, error)
        })?;
        let lease = tempfile::Builder::new()
            .prefix(".runmat-selected-link-")
            .tempdir_in(&parent)
            .map_err(|error| AotError::io("lease selected link inputs", &parent, error))?;
        let object_path = lease.path().join("program.o");
        super::native_input::validate(
            &object.bytes,
            runmat_execution_artifact::target_pack::TargetPackComponentKind::NativeObject,
            &manifest.target.target,
        )?;
        write_new(&object_path, &object.bytes)?;
        let toolchain_root = lease.path().join("toolchain");
        let sysroot_root = lease.path().join("sysroot");
        toolchain.stage(&toolchain_root, &admission.storage)?;
        sysroot.stage(&sysroot_root, &admission.storage)?;
        let component_root = lease.path().join("components");
        std::fs::create_dir(&component_root).map_err(|error| {
            AotError::io("stage selected component directory", &component_root, error)
        })?;
        let mut components = BTreeMap::new();
        for (ordinal, id) in contents.selection().runtime_components().iter().enumerate() {
            let bytes = contents
                .component_bytes(id)
                .ok_or_else(|| invalid("selected runtime member bytes are missing"))?;
            let descriptor = manifest
                .components
                .iter()
                .find(|entry| &entry.id == id)
                .ok_or_else(|| invalid("selected runtime member descriptor is missing"))?;
            if matches!(descriptor.kind, runmat_execution_artifact::target_pack::TargetPackComponentKind::NativeObject
                | runmat_execution_artifact::target_pack::TargetPackComponentKind::StaticArchive) {
                super::native_input::validate(bytes, descriptor.kind, &manifest.target.target)?;
            }
            // Filenames are private ordinal locators, never semantic labels or
            // command-line options. Repeated link occurrences reuse the lease.
            let path = component_root.join(format!("member-{ordinal}"));
            write_new(&path, bytes)?;
            components.insert(id.clone(), path);
        }
        let runtime_path = components
            .get(&manifest.runtime_component)
            .cloned()
            .ok_or_else(|| invalid("selected runtime archive was not retained"))?;
        let output = lease.path().join(
            if manifest.target.target.operating_system
                == runmat_execution::host::NativeOperatingSystem::Windows
            {
                "program.exe"
            } else {
                "program"
            },
        );
        let staged = Self {
            lease,
            object: object_path,
            runtime: runtime_path,
            output,
            components,
            toolchain_root,
            sysroot_root,
            toolchain,
            sysroot,
        };
        staged.verify(selected, object)?;
        Ok(staged)
    }

    pub fn verify(
        &self,
        selected: &SelectedNativeProduct<'_>,
        object: &RelocatableNativeObject,
    ) -> AotResult<()> {
        self.toolchain.verify_staged(&self.toolchain_root)?;
        self.sysroot.verify_staged(&self.sysroot_root)?;
        verify_file(&self.object, &object.bytes)?;
        for (id, path) in &self.components {
            verify_file(
                path,
                selected
                    .contents()
                    .component_bytes(id)
                    .ok_or_else(|| invalid("retained member bytes disappeared"))?,
            )?;
        }
        Ok(())
    }

    pub fn sysroot_file(&self, path: &runmat_package::NormalizedRelativePath) -> AotResult<&[u8]> {
        self.sysroot
            .file(path.as_str())
            .ok_or_else(|| invalid("hosted dependency is not an exact regular sysroot member"))
    }
}

fn verify_file(path: &Path, expected: &[u8]) -> AotResult<()> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| AotError::io("inspect leased native input", path, error))?;
    if !metadata.is_file()
        || std::fs::read(path)
            .map_err(|error| AotError::io("verify leased native input", path, error))?
            != expected
    {
        return Err(invalid(
            "leased native input differs from exact authenticated bytes",
        ));
    }
    Ok(())
}

fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.link.inputs", error.to_string())
}
