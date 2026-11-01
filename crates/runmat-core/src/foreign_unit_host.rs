//! Lightweight foreign adapter owner for one admitted executable unit.
//!
//! The interactive session and selected worker units use this same factory.
//! Each instance has its own handle registry, adapter catalogs, MEX module
//! table, Python session, and Java process/classpath. It owns no workspace,
//! source evaluator, or execution service port.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use runmat_process_host::ProcessGroupPlacement;
use runmat_runtime::context::{RuntimeContext, RuntimeForeignService};
use runmat_runtime::foreign::{
    ForeignAdapter, ForeignRuntime, InteropAdmissionPlan, JavaAdapter, MexRuntimeSession,
    NativeFfiAdapter, NativeFfiIsolationPolicy, PythonAdapter,
};
use runmat_runtime::RuntimeError;
use runmat_types::ForeignAdapterId;

mod retirement;

pub(crate) struct RegisteredForeignAdapterIds {
    pub native_ffi: ForeignAdapterId,
    pub mex: ForeignAdapterId,
    pub python: ForeignAdapterId,
    pub java: ForeignAdapterId,
}

pub(crate) struct SelectedForeignOwners<'a> {
    pub runtime: &'a ForeignRuntime,
    pub native: &'a NativeFfiAdapter,
    pub java: &'a JavaAdapter,
    pub python: &'a PythonAdapter,
    pub mex: &'a MexRuntimeSession,
}

/// A selected host remains live when a MEX callback rejects retirement.
/// The caller must retain the host and its private materialization for retry
/// or through an isolated worker's final process retirement.
pub enum ForeignUnitRetirement {
    Retained(RuntimeError),
    Retired(Result<(), RuntimeError>),
}

pub struct ForeignUnitHost {
    process_group_placement: ProcessGroupPlacement,
    runtime: Rc<ForeignRuntime>,
    native_ffi: Rc<NativeFfiAdapter>,
    java: Rc<JavaAdapter>,
    python: Rc<PythonAdapter>,
    mex: Rc<MexRuntimeSession>,
}

impl ForeignUnitHost {
    /// Borrow the Session's already registered adapter owners for the selected
    /// Direct install. This wrapper never owns Session state or a workspace.
    pub(crate) fn from_session_adapters(
        runtime: Rc<ForeignRuntime>,
        native_ffi: Rc<NativeFfiAdapter>,
        java: Rc<JavaAdapter>,
        python: Rc<PythonAdapter>,
        mex: Rc<MexRuntimeSession>,
    ) -> Self {
        Self {
            process_group_placement: ProcessGroupPlacement::OwnGroup,
            runtime,
            native_ffi,
            java,
            python,
            mex,
        }
    }

    pub fn new() -> Result<Self, RuntimeError> {
        Self::new_with_process_group(ProcessGroupPlacement::OwnGroup, None)
    }

    /// The one-shot Graph worker owns the enclosing process group. Isolated
    /// native, MEX, and Python helpers must remain in it until the parent proves tree
    /// retirement and releases its private artifact lease.
    pub fn new_in_graph_containment(
        materialization: &crate::foreign_artifacts::SelectedForeignMaterialization,
    ) -> Result<Self, RuntimeError> {
        let snapshot_parent = materialization.snapshot_parent().map_err(|error| {
            runmat_runtime::build_runtime_error(error)
                .with_builtin("native_ffi")
                .build()
        })?;
        let host = Self::new_with_process_group(
            ProcessGroupPlacement::JoinAncestorGroup,
            Some(snapshot_parent),
        )?;
        host.attach_private_byte_lease(materialization)?;
        Ok(host)
    }

    fn new_with_process_group(
        placement: ProcessGroupPlacement,
        snapshot_parent: Option<runmat_process_host::shared_memory::SharedSnapshotParent>,
    ) -> Result<Self, RuntimeError> {
        let runtime = Rc::new(ForeignRuntime::new(
            runmat_runtime::foreign::ForeignPlatform::Native,
        ));
        let native_ffi = NativeFfiAdapter::new_in_process_group(
            runtime.handles().clone(),
            placement,
            snapshot_parent.clone(),
        )?;
        runtime.register_adapter(native_ffi.clone())?;
        let java = JavaAdapter::new(runtime.handles().clone())?;
        runtime.register_adapter(java.clone())?;
        let python = PythonAdapter::new_in_process_group_with_parent(
            runtime.handles().clone(),
            placement,
            snapshot_parent.clone(),
        )?;
        runtime.register_adapter(python.clone())?;
        let mex = Rc::new(MexRuntimeSession::new_in_process_group(placement));
        runtime.register_adapter(mex.clone())?;
        Ok(Self {
            process_group_placement: placement,
            runtime,
            native_ffi,
            java,
            python,
            mex,
        })
    }

    pub(crate) fn attach_private_byte_lease(
        &self,
        materialization: &crate::foreign_artifacts::SelectedForeignMaterialization,
    ) -> Result<(), RuntimeError> {
        let lease = materialization.private_byte_lease();
        self.native_ffi.attach_private_byte_lease(lease.clone())?;
        self.java.attach_selected_byte_lease(lease.clone())?;
        self.python.attach_selected_byte_lease(lease)?;
        Ok(())
    }

    pub fn foreign_service(&self) -> Rc<dyn RuntimeForeignService> {
        self.runtime.clone()
    }

    pub(crate) fn selected_mex_runtime(&self) -> Rc<MexRuntimeSession> {
        Rc::clone(&self.mex)
    }

    pub(crate) fn registered_adapter_ids(&self) -> RegisteredForeignAdapterIds {
        RegisteredForeignAdapterIds {
            native_ffi: self.native_ffi.descriptor().adapter,
            mex: self.mex.descriptor().adapter,
            python: self.python.descriptor().adapter,
            java: self.java.descriptor().adapter,
        }
    }

    pub(crate) fn selected_owners(&self) -> SelectedForeignOwners<'_> {
        SelectedForeignOwners {
            runtime: &self.runtime,
            native: &self.native_ffi,
            java: &self.java,
            python: &self.python,
            mex: &self.mex,
        }
    }

    pub fn set_native_ffi_isolation_policy(
        &self,
        policy: NativeFfiIsolationPolicy,
    ) -> Result<(), RuntimeError> {
        self.native_ffi.set_isolation_policy(policy)
    }

    pub fn install_native_interface_artifact(
        &self,
        library_path: &Path,
        manifest_path: &Path,
    ) -> Result<String, RuntimeError> {
        self.native_ffi
            .install_prepared_artifact(library_path, manifest_path)
    }

    pub fn install_java_project_artifacts(
        &self,
        artifacts: &[(runmat_java::JavaArtifactIdentity, PathBuf)],
    ) -> Result<(), RuntimeError> {
        self.java.install_project_artifacts(artifacts)
    }

    pub fn install_portable_python_artifact_bundle(
        &self,
        bundle: &runmat_python::PythonArtifactBundle,
    ) -> Result<(), RuntimeError> {
        self.python.install_portable_artifact_bundle(bundle)
    }

    pub fn install_mex_artifact(&self, module_path: &Path) -> Result<String, RuntimeError> {
        self.mex.install_artifact(module_path)
    }

    pub fn admit_interop_manifest(
        &self,
        manifest: &runmat_types::InteropManifest,
    ) -> Result<InteropAdmissionPlan, RuntimeError> {
        self.runtime.admit(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EmptySelectedView(crate::foreign_artifacts::PrivateArtifactRoot);

    impl crate::foreign_artifacts::MaterializedForeignView for EmptySelectedView {
        fn private_byte_lease(&self) -> runmat_execution_artifact::PrivateArtifactLease {
            self.0.lease()
        }

        fn foreign_policy(&self) -> Option<&runmat_config::runtime::ForeignConfig> {
            None
        }

        fn native_interface(
            &self,
            _: &runmat_types::ForeignArtifactIdentity,
        ) -> Option<(
            &crate::foreign_artifacts::PrivateArtifactPath,
            &crate::foreign_artifacts::PrivateArtifactPath,
        )> {
            None
        }

        fn mex_module(
            &self,
            _: &runmat_types::ForeignArtifactIdentity,
        ) -> Option<(
            &crate::foreign_artifacts::PrivateArtifactPath,
            &crate::foreign_artifacts::PrivateArtifactPath,
            bool,
        )> {
            None
        }

        fn java_archives(
            &self,
        ) -> Vec<(
            runmat_java::JavaArtifactIdentity,
            &crate::foreign_artifacts::PrivateArtifactPath,
        )> {
            Vec::new()
        }

        fn python_bundle(&self) -> Option<&runmat_python::PythonArtifactBundle> {
            None
        }
    }

    #[test]
    fn selected_units_do_not_share_foreign_catalog_or_handle_registry() {
        let old = ForeignUnitHost::new().unwrap();
        let new = ForeignUnitHost::new().unwrap();
        assert!(!Rc::ptr_eq(&old.runtime, &new.runtime));
        assert!(!Rc::ptr_eq(&old.native_ffi, &new.native_ffi));
        assert!(!Rc::ptr_eq(&old.java, &new.java));
        assert!(!Rc::ptr_eq(&old.python, &new.python));
        assert!(!Rc::ptr_eq(&old.mex, &new.mex));
    }

    #[tokio::test]
    async fn disjoint_selected_closure_clears_prior_project_python_owner_after_rejected_selection()
    {
        let host = ForeignUnitHost::new().unwrap();
        host.install_portable_python_artifact_bundle(&runmat_python::PythonArtifactBundle::empty())
            .unwrap();
        assert!(host.python.has_project_artifacts().unwrap());
        let view = EmptySelectedView(
            crate::foreign_artifacts::PrivateArtifactRoot::new("runmat-disjoint-selected-")
                .unwrap(),
        );
        let mut rejected = runmat_types::InteropManifest::empty();
        rejected.schema_version = 0;
        assert!(
            crate::foreign_artifacts::install_selected_interop(&host, &rejected, &view, None)
                .await
                .is_err()
        );
        assert!(host.python.has_project_artifacts().unwrap());
        crate::foreign_artifacts::install_selected_interop(
            &host,
            &runmat_types::InteropManifest::empty(),
            &view,
            None,
        )
        .await
        .unwrap();
        assert!(!host.python.has_project_artifacts().unwrap());
    }
}
