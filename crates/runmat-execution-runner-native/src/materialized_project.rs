use std::path::Path;

use runmat_execution_artifact::{ExecutionBundle, ObjectNamespace};
use runmat_package::FrozenProjectHandoff;

use crate::{NativeExecutionError, NativeExecutionResult};

/// One private, exact, credential-free materialization of a portable bundle.
///
/// The temporary root owns only verified bundle bytes. Keeping this guard alive
/// keeps every path installed in the frozen-project handoff valid.
pub(crate) struct MaterializedProject {
    _root: tempfile::TempDir,
    handoff: FrozenProjectHandoff,
}

impl MaterializedProject {
    pub(crate) fn from_bundle(bundle: &ExecutionBundle) -> NativeExecutionResult<Self> {
        bundle.validate().map_err(protocol)?;
        let root = tempfile::Builder::new()
            .prefix("runmat-execution-")
            .tempdir()
            .map_err(protocol)?;
        make_private(root.path())?;
        for object in &bundle.objects {
            if object.descriptor.namespace != ObjectNamespace::ProgramSource {
                continue;
            }
            let target = root.path().join(&object.descriptor.logical_name);
            let parent = target
                .parent()
                .ok_or_else(|| protocol("bundle source has no materialization parent"))?;
            std::fs::create_dir_all(parent).map_err(protocol)?;
            make_private(parent)?;
            write_exact(&target, &object.bytes)?;
        }
        let handoff = bundle.project_handoff_at(root.path()).map_err(protocol)?;
        verify_materialized_sources(&handoff)?;
        Ok(Self {
            _root: root,
            handoff,
        })
    }

    pub(crate) fn handoff(&self) -> &FrozenProjectHandoff {
        &self.handoff
    }
}

fn write_exact(path: &Path, bytes: &[u8]) -> NativeExecutionResult<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(protocol)?;
    std::io::Write::write_all(&mut file, bytes).map_err(protocol)?;
    file.sync_all().map_err(protocol)?;
    let mut permissions = file.metadata().map_err(protocol)?.permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(path, permissions).map_err(protocol)?;
    Ok(())
}

fn verify_materialized_sources(handoff: &FrozenProjectHandoff) -> NativeExecutionResult<()> {
    for (source, path) in handoff.project.all_sources() {
        let bytes = std::fs::read(path).map_err(protocol)?;
        if runmat_package::ContentDigest::sha256(&bytes) != source.id.content_digest {
            return Err(protocol(format!(
                "materialized source {} differs from its frozen digest",
                source.id.relative_path
            )));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn make_private(path: &Path) -> NativeExecutionResult<()> {
    use std::os::unix::fs::PermissionsExt as _;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).map_err(protocol)?;
    Ok(())
}

#[cfg(not(unix))]
fn make_private(_path: &Path) -> NativeExecutionResult<()> {
    Ok(())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};
    use runmat_execution_artifact::{ExecutableForm, ExecutionBundleBuilder, ProgramBuildRecipe};

    use super::MaterializedProject;

    #[test]
    fn exact_sources_are_rebased_into_a_private_read_only_root() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(temp.path().join("src")).unwrap();
        std::fs::write(
            temp.path().join("runmat.toml"),
            "[package]\nname = \"materialized\"\n[sources]\nroots = [\"src\"]\n",
        )
        .unwrap();
        std::fs::write(
            temp.path().join("src/helper.m"),
            "function y = helper(); y = 42; end\n",
        )
        .unwrap();
        let project =
            runmat_package::build_frozen_project(&temp.path().join("runmat.toml"), BTreeSet::new())
                .unwrap();
        let revision = ProgramRevision::new(
            Digest::from_bytes(*project.graph_digest().bytes()),
            Digest::from_bytes(*project.source_revision().bytes()),
            ProgramEnvironment::new(
                1,
                1,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                "matlab",
            )
            .unwrap(),
        )
        .unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision.clone(),
            entrypoint: "helper".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("portable"),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let bundle = ExecutionBundleBuilder::native(&project, revision)
            .unwrap()
            .with_materialized_program(
                recipe,
                ExecutableForm::InterpreterBytecodeV1,
                serde_json::to_vec(&runmat_vm::FunctionRegistry::default()).unwrap(),
            )
            .build()
            .unwrap();
        let materialized = MaterializedProject::from_bundle(&bundle).unwrap();
        let source = materialized
            .handoff()
            .project
            .access_paths
            .values()
            .next()
            .unwrap();
        assert!(source.is_absolute());
        assert_eq!(
            std::fs::read_to_string(source).unwrap(),
            "function y = helper(); y = 42; end\n"
        );
        assert!(std::fs::metadata(source).unwrap().permissions().readonly());
    }
}
