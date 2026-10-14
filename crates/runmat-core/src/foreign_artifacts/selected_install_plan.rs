//! Side-effect-free selected input and host-policy admission.

use runmat_config::runtime::ForeignConfig;
use runmat_types::{ForeignArtifactIdentity, InteropManifest};

use crate::ForeignUnitHost;

use super::install::{MaterializedForeignView, SelectedForeignInstallError};
use super::policy_admission::SelectedPolicyAdmission;
use super::PrivateArtifactPath;

pub(crate) struct SelectedInteropInstallPlan<'a> {
    pub(super) private_byte_lease: Option<runmat_execution_artifact::PrivateArtifactLease>,
    pub(super) manifest: &'a InteropManifest,
    pub(super) policy: Option<&'a ForeignConfig>,
    pub(super) admission: Option<SelectedPolicyAdmission>,
    pub(super) native: Vec<(&'a PrivateArtifactPath, &'a PrivateArtifactPath)>,
    pub(super) mex_catalog: runmat_runtime::foreign::PreparedMexArtifactCatalog,
    pub(super) python: Option<&'a runmat_python::PythonArtifactBundle>,
    pub(super) java: Vec<(runmat_java::JavaArtifactIdentity, &'a PrivateArtifactPath)>,
    pub(super) requires_cuda: bool,
}

pub(crate) fn preflight_selected_interop<'a>(
    host: &ForeignUnitHost,
    manifest: &'a InteropManifest,
    materialized: &'a dyn MaterializedForeignView,
) -> Result<SelectedInteropInstallPlan<'a>, SelectedForeignInstallError> {
    let private_byte_lease = materialized.private_byte_lease();
    manifest
        .validate()
        .map_err(|error| SelectedForeignInstallError::Validation(error.to_string()))?;
    let ids = host.registered_adapter_ids();
    if let Some(unknown) = manifest.adapters.iter().find(|requirement| {
        requirement.adapter != ids.native_ffi
            && requirement.adapter != ids.mex
            && requirement.adapter != ids.python
            && requirement.adapter != ids.java
    }) {
        return Err(SelectedForeignInstallError::Validation(format!(
            "selected host has no registered adapter for {}",
            unknown.adapter
        )));
    }
    if *manifest == InteropManifest::empty() {
        return Ok(SelectedInteropInstallPlan {
            private_byte_lease: Some(private_byte_lease.clone()),
            manifest,
            policy: None,
            admission: None,
            native: Vec::new(),
            mex_catalog: runmat_runtime::foreign::PreparedMexArtifactCatalog::prepare(
                std::iter::empty::<runmat_runtime::foreign::MexArtifactExpectation<'a>>(),
            )
            .map_err(SelectedForeignInstallError::from)?
            .with_private_byte_lease(private_byte_lease),
            python: None,
            java: Vec::new(),
            requires_cuda: false,
        });
    }
    let policy = materialized.foreign_policy().ok_or_else(|| {
        SelectedForeignInstallError::Validation(
            "selected foreign recipe has no authenticated policy".into(),
        )
    })?;
    let admission = super::policy_admission::admit(policy, manifest, materialized, &ids)?;
    let mut native = Vec::new();
    let mut mex = Vec::new();
    let mut python = None;
    let mut java = Vec::new();
    let mut requires_cuda = false;
    for requirement in &manifest.adapters {
        if requirement.adapter == ids.native_ffi {
            for identity in &requirement.artifact_identities {
                let pair = materialized.native_interface(identity).ok_or_else(|| {
                    SelectedForeignInstallError::Validation(format!(
                        "selected host has no materialized native interface for required artifact {identity}"
                    ))
                })?;
                require_private_root(pair.0, &private_byte_lease)?;
                require_private_root(pair.1, &private_byte_lease)?;
                native.push(pair);
            }
        } else if requirement.adapter == ids.mex {
            for identity in &requirement.artifact_identities {
                let module = materialized.mex_module(identity).ok_or_else(|| {
                    SelectedForeignInstallError::Validation(format!(
                        "selected host has no materialized MEX module for required artifact {identity}"
                    ))
                })?;
                require_private_root(module.0, &private_byte_lease)?;
                require_private_root(module.1, &private_byte_lease)?;
                requires_cuda |= module.2;
                mex.push(module);
            }
        } else if requirement.adapter == ids.python {
            let bundle = materialized.python_bundle().ok_or_else(|| {
                SelectedForeignInstallError::Validation(
                    "selected host has no materialized Python artifact bundle".into(),
                )
            })?;
            let available = bundle.artifact_identities();
            if let Some(missing) = requirement
                .artifact_identities
                .iter()
                .find(|identity| !available.contains(identity.as_str()))
            {
                return Err(SelectedForeignInstallError::Validation(format!(
                    "selected host has no materialized Python environment or wheel for required identity {missing}"
                )));
            }
            python = Some(bundle);
        } else if requirement.adapter == ids.java {
            let available = materialized.java_archives();
            for required in &requirement.artifact_identities {
                if !available.iter().any(|(identity, _)| {
                    ForeignArtifactIdentity::new(identity.as_str())
                        .is_ok_and(|candidate| candidate == *required)
                }) {
                    return Err(SelectedForeignInstallError::Validation(format!(
                        "selected host has no materialized Java artifact for required identity {required}"
                    )));
                }
            }
            java = available
                .into_iter()
                .filter(|(identity, _)| {
                    ForeignArtifactIdentity::new(identity.as_str())
                        .is_ok_and(|candidate| requirement.artifact_identities.contains(&candidate))
                })
                .collect();
            for (_, path) in &java {
                require_private_root(path, &private_byte_lease)?;
            }
        }
    }
    let mex_catalog =
        runmat_runtime::foreign::PreparedMexArtifactCatalog::prepare(mex.into_iter().map(
            |(module, sidecar, _)| runmat_runtime::foreign::MexArtifactExpectation {
                module: module.as_path(),
                manifest: sidecar.as_path(),
                module_digest: runmat_execution::Digest::from_bytes(
                    *module.content_digest().bytes(),
                ),
                manifest_digest: runmat_execution::Digest::from_bytes(
                    *sidecar.content_digest().bytes(),
                ),
            },
        ))
        .map_err(SelectedForeignInstallError::from)?
        .with_private_byte_lease(private_byte_lease.clone());
    Ok(SelectedInteropInstallPlan {
        private_byte_lease: Some(private_byte_lease),
        manifest,
        policy: Some(policy),
        admission: Some(admission),
        native,
        mex_catalog,
        python,
        java,
        requires_cuda,
    })
}

fn require_private_root(
    path: &PrivateArtifactPath,
    lease: &runmat_execution_artifact::PrivateArtifactLease,
) -> Result<(), SelectedForeignInstallError> {
    if !path.as_path().starts_with(lease.path()) {
        return Err(SelectedForeignInstallError::Validation(
            "selected artifact path differs from its retained private byte owner".into(),
        ));
    }
    Ok(())
}
