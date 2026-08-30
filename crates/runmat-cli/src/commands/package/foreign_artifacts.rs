use anyhow::Result;
use runmat_execution_artifact::{ForeignArtifactClosure, LogicalObject};
use runmat_types::InteropManifest;

use super::{java_artifacts, mex_artifacts, native_interfaces, python_artifacts};

pub(crate) struct PreparedForeignArtifacts {
    pub(crate) interop: InteropManifest,
    pub(crate) accelerators: Vec<runmat_execution::resource::AcceleratorRequest>,
    pub(crate) objects: Vec<LogicalObject>,
    pub(crate) closures: Vec<ForeignArtifactClosure>,
    pub(crate) native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle,
    pub(crate) mex_artifacts: runmat_mex::MexArtifactBundle,
    pub(crate) java_artifacts: runmat_java::JavaArtifactBundle,
    pub(crate) python_artifacts: runmat_python::PythonArtifactBundle,
}

impl PreparedForeignArtifacts {
    pub(crate) fn empty() -> Self {
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

pub(crate) fn prepare_foreign_artifacts(
    project: &runmat_package::FrozenProject,
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    python_required: bool,
) -> Result<PreparedForeignArtifacts> {
    let native = native_interfaces::prepare_native_interfaces(project)?;
    let mex = mex_artifacts::prepare(project)?;
    let java = java_artifacts::prepare(project)?;
    let python = python_artifacts::prepare(Some(project), &config.foreign.python, python_required)?;
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

pub(crate) fn prepare_python_runtime(
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
