use std::path::Path;

use runmat_execution_artifact::{ExecutionBundle, ObjectNamespace};
use runmat_package::FrozenProjectHandoff;

use crate::{NativeExecutionError, NativeExecutionResult};

mod java_artifacts;
mod native_interfaces;
pub(crate) use java_artifacts::MaterializedJavaArtifact;
pub(crate) use native_interfaces::MaterializedNativeInterface;

/// One private, exact, credential-free materialization of a portable bundle.
///
/// The temporary root owns only verified bundle bytes. Keeping this guard alive
/// keeps every path installed in the frozen-project handoff valid.
pub(crate) struct MaterializedProject {
    _root: tempfile::TempDir,
    handoff: Option<FrozenProjectHandoff>,
    native_interfaces: Vec<MaterializedNativeInterface>,
    java_artifacts: Vec<MaterializedJavaArtifact>,
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
            if !matches!(
                object.descriptor.namespace,
                ObjectNamespace::ProgramSource | ObjectNamespace::ForeignArtifact
            ) {
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
        let handoff = bundle
            .requires_source_project()
            .then(|| bundle.project_handoff_at(root.path()))
            .transpose()
            .map_err(protocol)?;
        if let Some(handoff) = &handoff {
            verify_materialized_sources(handoff)?;
        }
        let foreign_objects = bundle
            .objects
            .iter()
            .filter(|object| object.descriptor.namespace == ObjectNamespace::ForeignArtifact)
            .cloned()
            .collect::<Vec<_>>();
        let native_interfaces = native_interfaces::discover(&foreign_objects, root.path())?;
        let java_artifacts = java_artifacts::discover(&foreign_objects, root.path())?;
        Ok(Self {
            _root: root,
            handoff,
            native_interfaces,
            java_artifacts,
        })
    }

    pub(crate) fn handoff(&self) -> Option<&FrozenProjectHandoff> {
        self.handoff.as_ref()
    }

    pub(crate) fn native_interface(&self, identity: &str) -> Option<&MaterializedNativeInterface> {
        self.native_interfaces
            .iter()
            .find(|interface| interface.manifest.identity.as_str() == identity)
    }

    pub(crate) fn java_artifact(&self, identity: &str) -> Option<&MaterializedJavaArtifact> {
        self.java_artifacts
            .iter()
            .find(|artifact| artifact.identity.as_str() == identity)
    }

    pub(crate) fn java_artifacts(&self) -> &[MaterializedJavaArtifact] {
        &self.java_artifacts
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
    use runmat_execution_artifact::{
        ExecutableForm, ExecutionBundleBuilder, LogicalObject, ObjectNamespace, ProgramBuildRecipe,
    };
    use runmat_native_ffi::{
        NativeInterfaceArtifactManifest, NativeLibrary, NativeLibraryMetadata,
        NATIVE_FFI_METADATA_SCHEMA_VERSION, NATIVE_INTERFACE_MANIFEST_MEDIA_TYPE,
        NATIVE_LIBRARY_MEDIA_TYPE,
    };

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
            .unwrap()
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

    #[test]
    fn compiled_foreign_artifacts_are_exactly_materialized_without_a_source_handoff() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("src")).unwrap();
        std::fs::write(
            temp.path().join("runmat.toml"),
            "[package]\nname = \"materialized-foreign\"\n[sources]\nroots = [\"src\"]\n",
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
        let library_bytes = b"synthetic native library".to_vec();
        let manifest = NativeInterfaceArtifactManifest::from_library(
            "fixture",
            NativeLibraryMetadata {
                schema_version: NATIVE_FFI_METADATA_SCHEMA_VERSION,
                target_triple: target_lexicon::HOST.to_string(),
                source_digest: "01".repeat(32),
                libraries: vec![NativeLibrary {
                    name: "fixture".into(),
                    path: "libfixture.native".into(),
                    dependencies: Vec::new(),
                    symbols: Vec::new(),
                }],
                structures: Vec::new(),
                enumerations: Vec::new(),
                aliases: Vec::new(),
            },
            &library_bytes,
        )
        .unwrap();
        let identity = manifest.identity.to_string();
        let sidecar = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            "foreign/fixture/interface.runmat.json",
            NATIVE_INTERFACE_MANIFEST_MEDIA_TYPE,
            manifest.canonical_bytes().unwrap(),
        )
        .unwrap();
        let library = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            "foreign/fixture/libfixture.native",
            NATIVE_LIBRARY_MEDIA_TYPE,
            library_bytes.clone(),
        )
        .unwrap();
        let java_bytes = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0".to_vec();
        let java_identity = runmat_java::JavaArtifactIdentity::for_bytes(&java_bytes);
        let java = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            "java/fixture.jar",
            runmat_java::JAVA_ARCHIVE_MEDIA_TYPE,
            java_bytes.clone(),
        )
        .unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision.clone(),
            entrypoint: "fixture".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "compiled".into(),
            target: runmat_execution_artifact::ProgramTarget::native(
                "host",
                runmat_execution_artifact::NativeTargetIdentity {
                    architecture: std::env::consts::ARCH.into(),
                    operating_system: std::env::consts::OS.into(),
                    pointer_width: usize::BITS as u16,
                    abi: std::env::consts::FAMILY.into(),
                    object_format: if cfg!(target_os = "macos") {
                        "mach-o"
                    } else if cfg!(windows) {
                        "coff"
                    } else {
                        "elf"
                    }
                    .into(),
                },
            ),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let bundle = ExecutionBundleBuilder::native(&project, revision)
            .unwrap()
            .with_compiled_package_closure()
            .with_foreign_artifact(sidecar)
            .unwrap()
            .with_foreign_artifact(library)
            .unwrap()
            .with_foreign_artifact(java)
            .unwrap()
            .with_materialized_program(
                recipe,
                ExecutableForm::NativeObjectV1,
                runmat_execution_artifact::NativeObjectPayload::new(
                    if cfg!(target_os = "macos") {
                        "mach-o"
                    } else if cfg!(windows) {
                        "coff"
                    } else {
                        "elf"
                    },
                    br#"{"schema_version":1}"#.to_vec(),
                    b"native object".to_vec(),
                )
                .unwrap()
                .to_canonical_bytes()
                .unwrap(),
            )
            .build()
            .unwrap();

        let materialized = MaterializedProject::from_bundle(&bundle).unwrap();
        assert!(materialized.handoff().is_none());
        let interface = materialized.native_interface(&identity).unwrap();
        assert_eq!(
            std::fs::read(&interface.library_path).unwrap(),
            library_bytes
        );
        assert_eq!(
            NativeInterfaceArtifactManifest::read(&interface.manifest_path).unwrap(),
            manifest
        );
        assert!(std::fs::metadata(&interface.library_path)
            .unwrap()
            .permissions()
            .readonly());
        let java = materialized.java_artifact(java_identity.as_str()).unwrap();
        assert_eq!(std::fs::read(&java.path).unwrap(), java_bytes);
        assert!(std::fs::metadata(&java.path)
            .unwrap()
            .permissions()
            .readonly());
    }
}
