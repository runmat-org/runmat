use anyhow::Result;
use runmat_execution_artifact::LogicalObject;
use runmat_types::InteropManifest;

use super::{java_artifacts, mex_artifacts, native_interfaces};

pub(crate) struct PreparedForeignArtifacts {
    pub(crate) interop: InteropManifest,
    pub(crate) objects: Vec<LogicalObject>,
    pub(crate) native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle,
    pub(crate) mex_artifacts: runmat_mex::MexArtifactBundle,
    pub(crate) java_artifacts: runmat_java::JavaArtifactBundle,
}

impl PreparedForeignArtifacts {
    pub(crate) fn empty() -> Self {
        Self {
            interop: InteropManifest::empty(),
            objects: Vec::new(),
            native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle::empty(),
            mex_artifacts: runmat_mex::MexArtifactBundle::empty(),
            java_artifacts: runmat_java::JavaArtifactBundle::empty(),
        }
    }
}

pub(crate) fn prepare_foreign_artifacts(
    project: &runmat_package::FrozenProject,
) -> Result<PreparedForeignArtifacts> {
    let native = native_interfaces::prepare_native_interfaces(project)?;
    let mex = mex_artifacts::prepare(project)?;
    let java = java_artifacts::prepare(project)?;
    let interop = InteropManifest::merge([native.interop, mex.interop, java.interop])
        .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
    let mut objects = native.objects;
    objects.extend(mex.objects);
    objects.extend(java.objects);
    Ok(PreparedForeignArtifacts {
        interop,
        objects,
        native_interfaces: native.bundle,
        mex_artifacts: mex.bundle,
        java_artifacts: java.bundle,
    })
}
