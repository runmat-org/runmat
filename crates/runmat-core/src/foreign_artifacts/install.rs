//! Exact selected-unit foreign installation shared by interactive and worker hosts.
//!
//! The view provides paths under a caller-owned private byte root that has
//! already been authenticated against the selected archive. This module
//! selects only typed artifact identities from the admitted unit manifest;
//! it never discovers a project path or name on its own.

use runmat_runtime::RuntimeError;
use runmat_types::{ForeignArtifactIdentity, InteropManifest};

use crate::ForeignUnitHost;

use super::PrivateArtifactPath;

/// Pure selected-materialization failures and live adapter failures retain
/// separate identities until the caller's runtime or worker-wire boundary.
#[derive(Debug)]
pub enum SelectedForeignInstallError {
    Validation(String),
    Runtime(Box<RuntimeError>),
}

impl SelectedForeignInstallError {
    pub fn into_runtime_error(self) -> RuntimeError {
        match self {
            Self::Validation(message) => runmat_runtime::build_runtime_error(message)
                .with_identifier("RunMat:SelectedForeignInstall")
                .build(),
            Self::Runtime(error) => *error,
        }
    }
}

impl std::fmt::Display for SelectedForeignInstallError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(message) => formatter.write_str(message),
            Self::Runtime(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

impl From<RuntimeError> for SelectedForeignInstallError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(Box::new(error))
    }
}

/// Borrowed, authenticated private materialization for one selected unit.
/// The implementor keeps its byte root alive through host shutdown.
pub trait MaterializedForeignView {
    fn private_byte_lease(&self) -> runmat_execution_artifact::PrivateArtifactLease;
    fn foreign_policy(&self) -> Option<&runmat_config::runtime::ForeignConfig>;
    fn native_interface(
        &self,
        identity: &ForeignArtifactIdentity,
    ) -> Option<(&PrivateArtifactPath, &PrivateArtifactPath)>;
    fn mex_module(
        &self,
        identity: &ForeignArtifactIdentity,
    ) -> Option<(&PrivateArtifactPath, &PrivateArtifactPath, bool)>;
    /// Preserve the authenticated bundle's classpath order.
    fn java_archives(&self) -> Vec<(runmat_java::JavaArtifactIdentity, &PrivateArtifactPath)>;
    fn python_bundle(&self) -> Option<&runmat_python::PythonArtifactBundle>;
}

pub async fn install_selected_interop(
    host: &ForeignUnitHost,
    manifest: &InteropManifest,
    materialized: &dyn MaterializedForeignView,
    runtime: Option<&runmat_runtime::context::RuntimeContext>,
) -> Result<(), SelectedForeignInstallError> {
    let plan =
        super::selected_install_plan::preflight_selected_interop(host, manifest, materialized)?;
    if host.selected_owners().mex.has_unretired_work() {
        return Err(SelectedForeignInstallError::Validation(
            "selected host has loaded MEX work requiring its scoped lifecycle owner".into(),
        ));
    }
    prepare_selected_cuda(&plan)?;
    let staged = super::selected_replacement::PreparedSelectedHostReplacement::prepare(host, plan)?;
    let retirement = staged.retire_project_hosts(runtime).await?;
    let publication_fence = staged.admit_commit_after_retirement(retirement)?.commit();
    drop(publication_fence);
    Ok(())
}

/// CUDA provider registration is host preparation, not a pure policy check.
pub(crate) fn prepare_selected_cuda(
    plan: &super::selected_install_plan::SelectedInteropInstallPlan<'_>,
) -> Result<(), SelectedForeignInstallError> {
    if plan.requires_cuda {
        ensure_cuda_mex_provider().map_err(SelectedForeignInstallError::Validation)?;
    }
    Ok(())
}

fn ensure_cuda_mex_provider() -> Result<(), String> {
    if runmat_accelerate_api::provider_for_native_device(
        runmat_accelerate_api::NativeDeviceApi::Cuda,
    )
    .is_some()
    {
        return Ok(());
    }
    match runmat_accelerate::backend::cuda::register_cuda_provider() {
        Ok(Some(_)) => Ok(()),
        Ok(None) => Err(
            "selected host requires an available CUDA driver and device for this MEX artifact"
                .into(),
        ),
        Err(error) => Err(format!(
            "selected host could not initialize its CUDA MEX provider: {error}"
        )),
    }
}
