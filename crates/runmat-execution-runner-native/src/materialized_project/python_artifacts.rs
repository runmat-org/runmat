use std::path::{Path, PathBuf};

use runmat_execution_artifact::LogicalObject;
use runmat_python::{PythonArtifactBundle, PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE};

use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Debug)]
pub(super) struct MaterializedPythonArtifact {
    pub(super) logical_name: String,
    pub(super) module: String,
    pub(super) path: PathBuf,
}

pub(crate) struct MaterializedPythonBundle {
    pub(crate) bundle: Option<PythonArtifactBundle>,
    pub(crate) artifacts: Vec<MaterializedPythonArtifact>,
}

pub(crate) fn discover(
    objects: &[LogicalObject],
    root: &Path,
) -> NativeExecutionResult<MaterializedPythonBundle> {
    let mut matches = objects
        .iter()
        .filter(|object| object.descriptor.media_type == PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE);
    let Some(object) = matches.next() else {
        return Ok(MaterializedPythonBundle {
            bundle: None,
            artifacts: Vec::new(),
        });
    };
    if matches.next().is_some() {
        return Err(protocol(
            "execution bundle contains more than one Python artifact bundle",
        ));
    }
    let bundle = PythonArtifactBundle::from_canonical_bytes(&object.bytes).map_err(protocol)?;
    let mut artifacts = Vec::with_capacity(bundle.artifacts.len());
    for artifact in &bundle.artifacts {
        let path = root
            .join("python-wheels")
            .join(&artifact.logical_name)
            .join(&artifact.filename);
        let parent = path
            .parent()
            .ok_or_else(|| protocol("Python wheel has no materialization parent"))?;
        std::fs::create_dir_all(parent).map_err(protocol)?;
        super::make_private(parent)?;
        super::write_exact(&path, &artifact.bytes)?;
        artifacts.push(MaterializedPythonArtifact {
            logical_name: artifact.logical_name.clone(),
            module: artifact.module.clone(),
            path,
        });
    }
    Ok(MaterializedPythonBundle {
        bundle: Some(bundle),
        artifacts,
    })
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
