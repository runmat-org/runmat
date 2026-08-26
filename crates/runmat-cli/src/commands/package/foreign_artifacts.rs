use anyhow::Result;
use runmat_execution_artifact::LogicalObject;
use runmat_types::InteropManifest;

use super::{java_artifacts, native_interfaces};

pub(crate) struct PreparedForeignArtifacts {
    pub(crate) interop: InteropManifest,
    pub(crate) objects: Vec<LogicalObject>,
    pub(crate) native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle,
    pub(crate) java_artifacts: runmat_java::JavaArtifactBundle,
}

impl PreparedForeignArtifacts {
    pub(crate) fn empty() -> Self {
        Self {
            interop: InteropManifest::empty(),
            objects: Vec::new(),
            native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle::empty(),
            java_artifacts: runmat_java::JavaArtifactBundle::empty(),
        }
    }
}

pub(crate) fn prepare_foreign_artifacts(
    project: &runmat_package::FrozenProject,
) -> Result<PreparedForeignArtifacts> {
    let native = native_interfaces::prepare_native_interfaces(project)?;
    let java = java_artifacts::prepare(project)?;
    let mut interop = InteropManifest::empty();
    interop.foreign_types.extend(native.interop.foreign_types);
    interop.foreign_types.extend(java.interop.foreign_types);
    interop.adapters.extend(native.interop.adapters);
    interop.adapters.extend(java.interop.adapters);
    interop
        .foreign_types
        .sort_by(|left, right| left.type_identity.cmp(&right.type_identity));
    interop
        .adapters
        .sort_by(|left, right| left.adapter.cmp(&right.adapter));
    interop
        .validate()
        .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
    let mut objects = native.objects;
    objects.extend(java.objects);
    Ok(PreparedForeignArtifacts {
        interop,
        objects,
        native_interfaces: native.bundle,
        java_artifacts: java.bundle,
    })
}
