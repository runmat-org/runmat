use anyhow::Result;
use runmat_execution_artifact::{ForeignArtifactClosure, LogicalObject};
use runmat_types::InteropManifest;
use std::sync::Arc;

mod install;
mod java_artifacts;
mod mex_artifacts;
mod native_interfaces;
mod policy;
mod policy_admission;
mod python_artifacts;
mod selected_bundle_input;
mod selected_issuance;
mod selected_install_plan;
mod selected_materialized;
mod selected_receipt;
mod selected_replacement;
mod selected_service;

pub(crate) use install::prepare_selected_cuda;
pub use install::{install_selected_interop, MaterializedForeignView, SelectedForeignInstallError};
pub use policy::{capture_foreign_policy_object, decode_admitted_policy};
pub(crate) use policy::{java_runtime, mex_policy, native_policy};
pub use runmat_execution_artifact::{
    PrivateArtifactPath, PrivateArtifactRoot, SELECTED_GRAPH_ATTEMPT_SCRATCH_PREFIX,
};
pub use selected_bundle_input::PreparedSelectedForeignInput;
pub(crate) use selected_install_plan::preflight_selected_interop;
pub use selected_materialized::SelectedForeignMaterialization;
pub(crate) use selected_receipt::selected_unit_interop_receipt;
pub(crate) use selected_replacement::PreparedSelectedHostReplacement;
pub use selected_service::selected_foreign_service;

pub(crate) use selected_issuance::{ForeignIssuanceSource, SelectedForeignPreparation};
pub struct PreparedForeignArtifacts {
    pub interop: InteropManifest,
    pub accelerators: Vec<runmat_execution::resource::AcceleratorRequest>,
    pub objects: Vec<LogicalObject>,
    pub closures: Vec<ForeignArtifactClosure>,
    pub native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle,
    pub mex_artifacts: runmat_mex::MexArtifactBundle,
    pub java_artifacts: runmat_java::JavaArtifactBundle,
    pub python_artifacts: runmat_python::PythonArtifactBundle,
}

/// Exact configured foreign inputs captured after selected code compilation
/// and before its invocation. No code or adapter is selected from a later
/// project handoff, configuration, or host path at graph-spawn time.
pub struct PreparedProjectForeignArtifacts {
    handoff: Arc<runmat_package::FrozenProjectHandoff>,
    runtime_config: Arc<runmat_config::runtime::RunMatRuntimeConfig>,
    python_required: bool,
    selected_inferred: Option<InteropManifest>,
    artifacts: Arc<PreparedForeignArtifacts>,
    policy_object: LogicalObject,
}

impl PreparedProjectForeignArtifacts {
    /// A comparison-only basis for same-definition reuse. Object descriptors
    /// bind the prepared bytes by digest; the installed closure and captured
    /// host policy are included without copying the foreign payloads.
    pub fn selected_reuse_basis(&self) -> Result<runmat_execution::Digest> {
        selected_artifact_basis(&self.artifacts, &self.runtime_config, self.python_required)
    }

    pub fn prepare(
        handoff: Arc<runmat_package::FrozenProjectHandoff>,
        runtime_config: Arc<runmat_config::runtime::RunMatRuntimeConfig>,
        python_required: bool,
    ) -> Result<Self> {
        handoff.validate()?;
        let artifacts =
            prepare_foreign_artifacts(&handoff.project, &runtime_config, python_required)?;
        Self::from_artifacts(handoff, runtime_config, python_required, None, artifacts)
    }

    /// Preserve the complete frozen project authority for escaped callables,
    /// adding only host-runtime grants proven by this exact compiled unit.
    pub(crate) fn prepare_for_selection(
        handoff: Arc<runmat_package::FrozenProjectHandoff>,
        runtime_config: Arc<runmat_config::runtime::RunMatRuntimeConfig>,
        inferred: &InteropManifest,
    ) -> Result<Self> {
        handoff.validate()?;
        inferred.validate()?;
        let python = runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID)?;
        let python_required = inferred.adapters.iter().any(|item| item.adapter == python);
        let artifacts =
            prepare_foreign_artifacts(&handoff.project, &runtime_config, python_required)?;
        let artifacts = complete_runtime_grants(artifacts, inferred)?;
        Self::from_artifacts(
            handoff,
            runtime_config,
            python_required,
            Some(inferred.clone()),
            artifacts,
        )
    }

    /// Observe only the adapter families established by the compiler's
    /// function-only manifest. The frozen project remains the exact source
    /// authority. Existing preparers still read all declarations within each
    /// selected adapter family before Artifact selects exact child identities.
    pub(crate) fn prepare_selected(
        handoff: Arc<runmat_package::FrozenProjectHandoff>,
        runtime_config: Arc<runmat_config::runtime::RunMatRuntimeConfig>,
        inferred: &InteropManifest,
    ) -> Result<Self> {
        handoff.validate()?;
        let artifacts =
            prepare_foreign_artifacts_selected(&handoff.project, &runtime_config, inferred)?;
        let python = runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID)?;
        let python_required = inferred.adapters.iter().any(|item| item.adapter == python);
        Self::from_artifacts(
            handoff,
            runtime_config,
            python_required,
            Some(inferred.clone()),
            artifacts,
        )
    }

    fn from_artifacts(
        handoff: Arc<runmat_package::FrozenProjectHandoff>,
        runtime_config: Arc<runmat_config::runtime::RunMatRuntimeConfig>,
        python_required: bool,
        selected_inferred: Option<InteropManifest>,
        artifacts: PreparedForeignArtifacts,
    ) -> Result<Self> {
        let policy_object = policy::capture_foreign_policy_object(&runtime_config.foreign)
            .map_err(anyhow::Error::msg)?;
        Ok(Self {
            handoff,
            runtime_config,
            python_required,
            selected_inferred,
            artifacts: Arc::new(artifacts),
            policy_object,
        })
    }

    pub fn matches_handoff(&self, handoff: &runmat_package::FrozenProjectHandoff) -> bool {
        self.handoff.as_ref() == handoff
    }

    pub fn project_revision(&self) -> runmat_package::ProjectRevision {
        self.handoff.revision()
    }

    pub fn runtime_config(&self) -> &runmat_config::runtime::RunMatRuntimeConfig {
        &self.runtime_config
    }

    pub fn python_required(&self) -> bool {
        self.python_required
    }

    pub(crate) fn matches_selection(
        &self,
        handoff: &runmat_package::FrozenProjectHandoff,
        inferred: &InteropManifest,
    ) -> bool {
        self.matches_handoff(handoff) && self.selected_inferred.as_ref() == Some(inferred)
    }

    pub fn artifacts(&self) -> &PreparedForeignArtifacts {
        &self.artifacts
    }

    pub(crate) fn policy_object(&self) -> &LogicalObject {
        &self.policy_object
    }
}

fn selected_artifact_basis(
    artifacts: &PreparedForeignArtifacts,
    runtime_config: &runmat_config::runtime::RunMatRuntimeConfig,
    python_required: bool,
) -> Result<runmat_execution::Digest> {
    let mut objects = artifacts
        .objects
        .iter()
        .map(|object| {
            object.validate()?;
            Ok(&object.descriptor)
        })
        .collect::<runmat_execution_artifact::ArtifactResult<Vec<_>>>()?;
    objects.sort();
    let mut closures = artifacts.closures.iter().collect::<Vec<_>>();
    closures.sort();
    let encoded = serde_json::to_vec(&(
        "runmat-selected-foreign-basis-v1",
        &artifacts.interop,
        objects,
        closures,
        &runtime_config.foreign,
        python_required,
    ))?;
    Ok(runmat_execution::Digest::sha256(&encoded))
}

impl PreparedForeignArtifacts {
    pub fn empty() -> Self {
        Self {
            interop: InteropManifest::empty(),
            accelerators: Vec::new(),
            objects: Vec::new(),
            closures: Vec::new(),
            native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle::empty(),
            mex_artifacts: runmat_mex::MexArtifactBundle::empty(),
            java_artifacts: runmat_java::JavaArtifactBundle::empty(),
            python_artifacts: runmat_python::PythonArtifactBundle::empty(),
        }
    }
}

pub fn prepare_foreign_artifacts(
    project: &runmat_package::FrozenProject,
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    python_required: bool,
) -> Result<PreparedForeignArtifacts> {
    let native = native_interfaces::prepare_native_interfaces(project)?;
    let mex = mex_artifacts::prepare(project)?;
    let java = java_artifacts::prepare(project, false)?;
    let python = python_artifacts::prepare(Some(project), &config.foreign.python, python_required)?;
    assemble_prepared_foreign_artifacts(native, mex, java, python)
}

fn prepare_foreign_artifacts_selected(
    project: &runmat_package::FrozenProject,
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    inferred: &InteropManifest,
) -> Result<PreparedForeignArtifacts> {
    inferred.validate()?;
    let selected = inferred
        .adapters
        .iter()
        .map(|item| item.adapter.clone())
        .collect();
    let artifacts = prepare_foreign_artifacts_for_selection(project, config, &selected, None)?;
    complete_runtime_grants(artifacts, inferred)
}

fn complete_runtime_grants(
    mut artifacts: PreparedForeignArtifacts,
    inferred: &InteropManifest,
) -> Result<PreparedForeignArtifacts> {
    let java = runmat_types::ForeignAdapterId::new(runmat_java::JAVA_ADAPTER_ID)?;
    if inferred.adapters.iter().any(|item| item.adapter == java)
        && !artifacts
            .interop
            .adapters
            .iter()
            .any(|requirement| requirement.adapter == java)
    {
        let runtime = java_artifacts::prepare_runtime()?;
        artifacts.interop = InteropManifest::merge([artifacts.interop, runtime.interop])
            .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
    }
    if let Some(grant) = runtime_native_grant(inferred)? {
        let native = &grant.adapters[0].adapter;
        if !artifacts
            .interop
            .adapters
            .iter()
            .any(|requirement| &requirement.adapter == native)
        {
            artifacts.interop = InteropManifest::merge([artifacts.interop, grant])
                .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
        }
    }
    Ok(artifacts)
}

/// A compiler-reachable raw `loadlibrary` uses a host-local path at call
/// time. Its adapter grant carries no immutable native interface identity.
fn runtime_native_grant(inferred: &InteropManifest) -> Result<Option<InteropManifest>> {
    let native = runmat_types::ForeignAdapterId::new(runmat_native_ffi::NATIVE_FFI_ADAPTER_ID)?;
    inferred
        .adapters
        .iter()
        .find(|requirement| requirement.adapter == native)
        .map(|requirement| {
            anyhow::ensure!(
                requirement.artifact_identities.is_empty(),
                "runtime native adapter cannot inherit ambient artifact identities"
            );
            let mut grant = InteropManifest::empty();
            grant.adapters.push(requirement.clone());
            grant
                .validate()
                .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
            Ok(grant)
        })
        .transpose()
}

/// Issue only selected adapter families and, when known, exact named MEX
/// modules. Every emitted artifact identity is an explicit authenticated
/// grant. `None` MEX names grants all frozen declarations for an unknown
/// dynamic call (or Core's parent-selection projection).
pub fn prepare_foreign_artifacts_for_selection(
    project: &runmat_package::FrozenProject,
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    selected: &std::collections::BTreeSet<runmat_types::ForeignAdapterId>,
    selected_mex_names: Option<&std::collections::BTreeSet<String>>,
) -> Result<PreparedForeignArtifacts> {
    let native_id = runmat_types::ForeignAdapterId::new(runmat_native_ffi::NATIVE_FFI_ADAPTER_ID)?;
    let mex_id = runmat_types::ForeignAdapterId::new(runmat_mex::MEX_ADAPTER_ID)?;
    let java_id = runmat_types::ForeignAdapterId::new(runmat_java::JAVA_ADAPTER_ID)?;
    let python_id = runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID)?;
    for adapter in selected {
        anyhow::ensure!(
            adapter == &native_id
                || adapter == &mex_id
                || adapter == &java_id
                || adapter == &python_id,
            "selected foreign adapter has no prepared project issuer"
        );
    }
    if let Some(names) = selected_mex_names {
        anyhow::ensure!(
            names.is_empty() || selected.contains(&mex_id),
            "selected MEX names require the MEX adapter"
        );
        for name in names {
            anyhow::ensure!(
                project.mex_artifacts.iter().any(|item| &item.name == name),
                "selected MEX module `{name}` is absent from the frozen project"
            );
        }
    }
    let includes = |id: &runmat_types::ForeignAdapterId| selected.contains(id);
    let native = if includes(&native_id) {
        native_interfaces::prepare_native_interfaces(project)?
    } else {
        native_interfaces::PreparedNativeInterfaces {
            interop: InteropManifest::empty(),
            objects: Vec::new(),
            closures: Vec::new(),
            bundle: runmat_native_ffi::NativeInterfaceArtifactBundle::empty(),
        }
    };
    let mex = if includes(&mex_id) {
        mex_artifacts::prepare_selected(project, selected_mex_names)?
    } else {
        mex_artifacts::PreparedMexArtifacts {
            interop: InteropManifest::empty(),
            accelerators: Vec::new(),
            objects: Vec::new(),
            closures: Vec::new(),
            bundle: runmat_mex::MexArtifactBundle::empty(),
        }
    };
    let java = if includes(&java_id) {
        java_artifacts::prepare(project, true)?
    } else {
        java_artifacts::PreparedJavaArtifacts {
            interop: InteropManifest::empty(),
            objects: Vec::new(),
            closures: Vec::new(),
            bundle: runmat_java::JavaArtifactBundle::empty(),
        }
    };
    let python = python_artifacts::prepare(
        includes(&python_id).then_some(project),
        &config.foreign.python,
        includes(&python_id),
    )?;
    assemble_prepared_foreign_artifacts(native, mex, java, python)
}

fn assemble_prepared_foreign_artifacts(
    native: native_interfaces::PreparedNativeInterfaces,
    mex: mex_artifacts::PreparedMexArtifacts,
    java: java_artifacts::PreparedJavaArtifacts,
    python: python_artifacts::PreparedPythonArtifacts,
) -> Result<PreparedForeignArtifacts> {
    let interop =
        InteropManifest::merge([native.interop, mex.interop, java.interop, python.interop])
            .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
    let mut objects = native.objects;
    objects.extend(mex.objects);
    objects.extend(java.objects);
    objects.extend(python.objects);
    let mut closures = native.closures;
    closures.extend(mex.closures);
    closures.extend(java.closures);
    closures.extend(python.closures);
    Ok(PreparedForeignArtifacts {
        interop,
        accelerators: mex.accelerators,
        objects,
        closures,
        native_interfaces: native.bundle,
        mex_artifacts: mex.bundle,
        java_artifacts: java.bundle,
        python_artifacts: python.bundle,
    })
}

/// Issue only host-runtime grants proven reachable from a projectless unit.
/// Raw native loads retain host-local path semantics; no local path becomes an
/// immutable native-interface identity or ambient selected grant.
pub(crate) fn prepare_runtime_only_for_selection(
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    inferred: &InteropManifest,
) -> Result<PreparedForeignArtifacts> {
    inferred
        .validate()
        .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
    let java = runmat_types::ForeignAdapterId::new(runmat_java::JAVA_ADAPTER_ID)?;
    let python = runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID)?;
    let native = runmat_types::ForeignAdapterId::new(runmat_native_ffi::NATIVE_FFI_ADAPTER_ID)?;
    for requirement in &inferred.adapters {
        anyhow::ensure!(
            requirement.adapter == java
                || requirement.adapter == python
                || requirement.adapter == native,
            "runtime-only selected adapter {} requires immutable project artifacts",
            requirement.adapter
        );
        anyhow::ensure!(
            requirement.artifact_identities.is_empty(),
            "runtime-only selected adapter {} cannot inherit ambient artifact identities",
            requirement.adapter
        );
    }
    let python_required = inferred.adapters.iter().any(|item| item.adapter == python);
    let java_required = inferred.adapters.iter().any(|item| item.adapter == java);
    let mut artifacts = if python_required {
        prepare_python_runtime(config)?
    } else {
        PreparedForeignArtifacts::empty()
    };
    if java_required {
        let java = java_artifacts::prepare_runtime()?;
        artifacts.interop = InteropManifest::merge([artifacts.interop, java.interop])
            .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
        artifacts.java_artifacts = java.bundle;
    }
    if let Some(native) = runtime_native_grant(inferred)? {
        artifacts.interop = InteropManifest::merge([artifacts.interop, native])
            .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
    }
    Ok(artifacts)
}

pub fn prepare_python_runtime(
    config: &runmat_config::runtime::RunMatRuntimeConfig,
) -> Result<PreparedForeignArtifacts> {
    let python = python_artifacts::prepare(None, &config.foreign.python, true)?;
    Ok(PreparedForeignArtifacts {
        interop: python.interop,
        accelerators: Vec::new(),
        objects: python.objects,
        closures: python.closures,
        native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle::empty(),
        mex_artifacts: runmat_mex::MexArtifactBundle::empty(),
        java_artifacts: runmat_java::JavaArtifactBundle::empty(),
        python_artifacts: python.bundle,
    })
}
