use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use runmat_execution_artifact::LogicalObject;
use runmat_java::{JavaArtifactIdentity, JAVA_ARCHIVE_MEDIA_TYPE};

use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Debug)]
pub(crate) struct MaterializedJavaArtifact {
    pub(crate) identity: JavaArtifactIdentity,
    pub(crate) path: PathBuf,
}

pub(crate) fn discover(
    objects: &[LogicalObject],
    root: &Path,
) -> NativeExecutionResult<Vec<MaterializedJavaArtifact>> {
    let mut identities = BTreeSet::new();
    let mut artifacts = Vec::new();
    for object in objects
        .iter()
        .filter(|object| object.descriptor.media_type == JAVA_ARCHIVE_MEDIA_TYPE)
    {
        let identity = JavaArtifactIdentity::for_bytes(&object.bytes);
        if !identities.insert(identity.clone()) {
            return Err(protocol(format!(
                "Java artifact identity {identity} appears more than once"
            )));
        }
        artifacts.push(MaterializedJavaArtifact {
            identity,
            path: root.join(&object.descriptor.logical_name),
        });
    }
    Ok(artifacts)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
