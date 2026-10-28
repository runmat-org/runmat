//! Selected native local-effect identity, separate from generic host ports.
//!
//! The AOT owner must install this for every selected product, even when its
//! local effect set is empty. An empty selected scope denies ambient provider
//! fallback rather than behaving like an ordinary unselected RuntimeContext.

use super::RuntimeServicePorts;
use runmat_execution::host::{
    NativeDeploymentProfile, NativeDeploymentServiceKind, NativeFilesystemEntry,
    NativeLocalServiceInputs, NativeSelectedProductBinding, VerifiedNativeServiceIssuer,
};
use runmat_execution::Digest;
use runmat_filesystem::{
    FsProvider, InvocationFilesystem, InvocationFilesystemSelection, MemoryFsProvider,
    SelectedFsProvider,
};
use runmat_types::EffectKind;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

/// A policy refusal contains no worker-local runtime resources. Both the
/// language diagnostic and filesystem/environment I/O adapters retain this
/// same typed cause, including on WebAssembly where language errors may own
/// thread-local resources that cannot be put inside `std::io::Error`.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("selected native invocation lacks local effect {effect:?}")]
pub(crate) struct SelectedLocalEffectDenied {
    effect: EffectKind,
}

#[derive(Clone)]
pub struct RuntimeSelectedLocalServices {
    product: Digest,
    invocation: Digest,
    issuers: BTreeMap<NativeDeploymentServiceKind, VerifiedNativeServiceIssuer>,
    filesystem: Option<InvocationFilesystem>,
    directory: Option<PathBuf>,
    temporary_directory: Option<PathBuf>,
}

impl RuntimeSelectedLocalServices {
    /// Materialize one exact selected local service set. Deterministic input
    /// bytes are data, never issuer authority; hosted providers are created
    /// only for effects selected by the verified product.
    pub fn materialize(
        product: &NativeSelectedProductBinding,
        invocation: Digest,
        issuers: impl IntoIterator<Item = VerifiedNativeServiceIssuer>,
        inputs: Option<&NativeLocalServiceInputs>,
    ) -> Result<Self, crate::RuntimeError> {
        runmat_execution::host::validate_native_local_inputs(Some(product), inputs)
            .map_err(|error| mismatch(&error.to_string()))?;
        let has = |effect| {
            product
                .services
                .iter()
                .any(|service| service.kind == NativeDeploymentServiceKind::Effect(effect))
        };
        let environment_read = has(EffectKind::EnvironmentRead);
        let environment_write = has(EffectKind::EnvironmentWrite);
        let filesystem_read = has(EffectKind::FilesystemRead);
        let filesystem_write = has(EffectKind::FilesystemWrite);
        let deterministic = product.profile == NativeDeploymentProfile::DeterministicDeployed;
        let directory = if deterministic {
            let declared = inputs
                .and_then(|inputs| inputs.environment.as_ref())
                .map(|environment| PathBuf::from(&environment.current_directory));
            declared.or_else(|| (filesystem_read || filesystem_write).then(|| PathBuf::from("/")))
        } else if environment_read || filesystem_read || filesystem_write {
            // A hosted filesystem resolves relative paths against its own
            // captured native directory. This is not permission to expose
            // `pwd`: InvocationFilesystem gates that separate environment read.
            // A synthetic "/" would change relative-path semantics and is
            // not an absolute native path on Windows.
            Some(std::env::current_dir().map_err(|error| mismatch(&error.to_string()))?)
        } else {
            None
        };
        let filesystem = if filesystem_read || filesystem_write {
            let provider: Arc<dyn FsProvider> = if deterministic {
                let inputs = inputs
                    .and_then(|inputs| inputs.filesystem.as_ref())
                    .ok_or_else(|| mismatch("selected virtual filesystem inputs are absent"))?;
                let memory = MemoryFsProvider::new();
                for entry in &inputs.entries {
                    match entry {
                        NativeFilesystemEntry::Directory { path } => memory
                            .create_dir_project_path(Path::new(path.as_str()), true)
                            .map_err(|error| mismatch(&error.to_string()))?,
                        NativeFilesystemEntry::File {
                            path,
                            bytes,
                            readonly,
                        } => {
                            memory
                                .write_project_path(Path::new(path.as_str()), bytes)
                                .map_err(|error| mismatch(&error.to_string()))?;
                            memory
                                .set_readonly_project_path(Path::new(path.as_str()), *readonly)
                                .map_err(|error| mismatch(&error.to_string()))?;
                        }
                    }
                }
                if let Some(directory) = &directory {
                    let metadata = memory
                        .metadata_project_path(directory)
                        .map_err(|error| mismatch(&error.to_string()))?;
                    if !metadata.is_dir() {
                        return Err(mismatch("selected working directory is not a directory"));
                    }
                }
                Arc::new(memory)
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    Arc::new(runmat_filesystem::NativeFsProvider)
                }
                #[cfg(target_arch = "wasm32")]
                {
                    return Err(mismatch(
                        "HostedNative filesystem is unavailable in the browser",
                    ));
                }
            };
            let attenuated = Arc::new(
                SelectedFsProvider::new(provider, filesystem_read, filesystem_write)
                    .map_err(|error| mismatch(&error.to_string()))?,
            ) as Arc<dyn FsProvider>;
            Some(
                InvocationFilesystem::new(
                    attenuated,
                    directory.clone().unwrap_or_else(|| PathBuf::from("/")),
                    environment_read,
                    environment_write,
                )
                .map_err(|error| mismatch(&error.to_string()))?,
            )
        } else {
            None
        };
        Self::new(product, invocation, issuers, filesystem, directory)
    }

    pub fn new(
        product: &NativeSelectedProductBinding,
        invocation: Digest,
        issuers: impl IntoIterator<Item = VerifiedNativeServiceIssuer>,
        filesystem: Option<InvocationFilesystem>,
        directory: Option<PathBuf>,
    ) -> Result<Self, crate::RuntimeError> {
        let product_identity = product
            .fingerprint()
            .map_err(|error| mismatch(&error.to_string()))?;
        if invocation.bytes().iter().all(|byte| *byte == 0) {
            return Err(mismatch("missing selected invocation identity"));
        }
        let mut selected = BTreeMap::new();
        for issuer in issuers {
            let kind = issuer.binding().kind;
            if !kind.is_local_effect_service()
                || issuer.product() != product_identity
                || product.services.iter().find(|service| service.kind == kind)
                    != Some(issuer.binding())
            {
                return Err(mismatch("local issuer differs from selected product/kind"));
            }
            if selected.insert(kind, issuer).is_some() {
                return Err(mismatch("duplicate selected local issuer"));
            }
        }
        if product
            .services
            .iter()
            .filter(|service| service.kind.is_local_effect_service())
            .any(|service| !selected.contains_key(&service.kind))
        {
            return Err(mismatch("selected local issuer set is incomplete"));
        }
        let filesystem_selected = selected.keys().any(|kind| {
            matches!(
                kind,
                NativeDeploymentServiceKind::Effect(
                    EffectKind::FilesystemRead | EffectKind::FilesystemWrite
                )
            )
        });
        if filesystem.is_some() != filesystem_selected {
            return Err(mismatch("selected filesystem provider/issuer mismatch"));
        }
        if filesystem.is_some() && directory.is_none() {
            return Err(mismatch(
                "selected filesystem has no invocation working directory",
            ));
        }
        let temporary_directory = selected
            .contains_key(&NativeDeploymentServiceKind::Effect(
                EffectKind::EnvironmentRead,
            ))
            .then(|| match product.profile {
                NativeDeploymentProfile::HostedNative => std::env::temp_dir(),
                NativeDeploymentProfile::DeterministicDeployed => PathBuf::from("/tmp"),
            });
        Ok(Self {
            product: product_identity,
            invocation,
            issuers: selected,
            filesystem,
            directory,
            temporary_directory,
        })
    }

    pub fn require(
        &self,
        issuer: &VerifiedNativeServiceIssuer,
        invocation: Digest,
    ) -> Result<(), crate::RuntimeError> {
        if self.invocation != invocation
            || issuer.product() != self.product
            || self.issuers.get(&issuer.binding().kind) != Some(issuer)
        {
            return Err(mismatch(
                "live local service differs from selected issuer/invocation",
            ));
        }
        Ok(())
    }

    pub(crate) fn filesystem_selection(&self) -> InvocationFilesystemSelection {
        if let Some(owner) = &self.filesystem {
            return InvocationFilesystemSelection::Selected(owner.clone());
        }
        InvocationFilesystemSelection::Denied {
            directory: self.directory.clone(),
            environment_read: self.has(EffectKind::EnvironmentRead),
        }
    }

    pub fn has(&self, effect: EffectKind) -> bool {
        self.issuers
            .contains_key(&NativeDeploymentServiceKind::Effect(effect))
    }

    pub(crate) fn matches_product_invocation(&self, product: Digest, invocation: Digest) -> bool {
        self.product == product && self.invocation == invocation
    }

    pub(crate) fn temporary_directory(&self) -> Option<&Path> {
        self.temporary_directory.as_deref()
    }
}

impl RuntimeServicePorts {
    pub(crate) fn has_selected_local_services(&self) -> bool {
        self.shared.selected_local.is_some()
    }

    pub(crate) fn selected_local_matches(&self, product: Digest, invocation: Digest) -> bool {
        self.shared
            .selected_local
            .as_ref()
            .is_some_and(|selected| selected.matches_product_invocation(product, invocation))
    }

    pub(crate) fn selected_temporary_directory(&self) -> Option<PathBuf> {
        self.shared
            .selected_local
            .as_ref()
            .and_then(|selected| selected.temporary_directory())
            .map(Path::to_path_buf)
    }

    pub fn with_selected_local_services(mut self, selected: RuntimeSelectedLocalServices) -> Self {
        Rc::make_mut(&mut self.shared).selected_local = Some(Rc::new(selected));
        self
    }

    pub fn require_selected_local_service(
        &self,
        issuer: &VerifiedNativeServiceIssuer,
        invocation: Digest,
    ) -> Result<(), crate::RuntimeError> {
        self.shared
            .selected_local
            .as_ref()
            .ok_or_else(|| mismatch("selected local services were not installed"))?
            .require(issuer, invocation)
    }

    /// Effect admission for ordinary Runtime helpers reached from an already
    /// selected native invocation. Absence is meaningful only in that scope;
    /// ordinary host sessions retain their existing compatibility behavior.
    pub fn require_selected_local_effect(
        &self,
        effect: EffectKind,
    ) -> Result<(), crate::RuntimeError> {
        self.check_selected_local_effect(effect).map_err(|source| {
            let mut error = mismatch("selected native invocation lacks this local effect");
            error.source = Some(Box::new(source));
            error
        })
    }

    pub(crate) fn check_selected_local_effect(
        &self,
        effect: EffectKind,
    ) -> Result<(), SelectedLocalEffectDenied> {
        if let Some(selected) = &self.shared.selected_local {
            if !selected.has(effect) {
                return Err(SelectedLocalEffectDenied { effect });
            }
        }
        Ok(())
    }

    pub(crate) fn selected_filesystem_for_active_context(
        &self,
    ) -> Option<InvocationFilesystemSelection> {
        self.shared
            .selected_local
            .as_ref()
            .map(|selected| selected.filesystem_selection())
    }
}

fn mismatch(message: &str) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:Native:SelectedLocalServiceMismatch")
        .build()
}
