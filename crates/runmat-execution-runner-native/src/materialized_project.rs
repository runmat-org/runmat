use std::path::Path;

use runmat_execution_artifact::{ExecutionBundle, ObjectNamespace};
use runmat_package::FrozenProjectHandoff;

use crate::{NativeExecutionError, NativeExecutionResult};

mod java_artifacts;
mod mex_artifacts;
mod native_interfaces;
mod python_artifacts;
pub(crate) use java_artifacts::MaterializedJavaArtifact;
pub(crate) use mex_artifacts::MaterializedMexArtifact;
pub(crate) use native_interfaces::MaterializedNativeInterface;
use python_artifacts::MaterializedPythonArtifact;

/// One private, exact, credential-free materialization of a portable bundle.
///
/// The temporary root owns only verified bundle bytes. Keeping this guard alive
/// keeps every path installed in the frozen-project handoff valid.
pub(crate) struct MaterializedProject {
    _root: tempfile::TempDir,
    handoff: Option<FrozenProjectHandoff>,
    native_interfaces: Vec<MaterializedNativeInterface>,
    mex_artifacts: Vec<MaterializedMexArtifact>,
    java_artifacts: Vec<MaterializedJavaArtifact>,
    python_bundle: Option<runmat_python::PythonArtifactBundle>,
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
        let mut handoff = bundle
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
        let mex_artifacts = mex_artifacts::discover(&foreign_objects, root.path())?;
        let java_artifacts = java_artifacts::discover(&foreign_objects, root.path())?;
        let python = python_artifacts::discover(&foreign_objects, root.path())?;
        if let Some(handoff) = handoff.as_mut() {
            rebase_foreign_artifacts(
                handoff,
                &native_interfaces,
                &mex_artifacts,
                &java_artifacts,
                &python.artifacts,
            )?;
        }
        Ok(Self {
            _root: root,
            handoff,
            native_interfaces,
            mex_artifacts,
            java_artifacts,
            python_bundle: python.bundle,
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

    pub(crate) fn mex_artifact(&self, identity: &str) -> Option<&MaterializedMexArtifact> {
        self.mex_artifacts
            .iter()
            .find(|artifact| artifact.manifest.identity.as_str() == identity)
    }

    pub(crate) fn requires_cuda_mex(&self) -> bool {
        self.mex_artifacts.iter().any(|artifact| {
            artifact.manifest.source_language == runmat_mex::MexSourceLanguage::Cuda
        })
    }

    pub(crate) fn java_artifacts(&self) -> &[MaterializedJavaArtifact] {
        &self.java_artifacts
    }

    pub(crate) fn python_bundle(&self) -> Option<&runmat_python::PythonArtifactBundle> {
        self.python_bundle.as_ref()
    }
}

fn rebase_foreign_artifacts(
    handoff: &mut FrozenProjectHandoff,
    native_interfaces: &[MaterializedNativeInterface],
    mex_artifacts: &[MaterializedMexArtifact],
    java_artifacts: &[MaterializedJavaArtifact],
    python_artifacts: &[MaterializedPythonArtifact],
) -> NativeExecutionResult<()> {
    for interface in &mut handoff.project.native_interfaces {
        let materialized = native_interfaces
            .iter()
            .find(|materialized| {
                materialized.manifest.interface_name == interface.name
                    && digest_path(&materialized.manifest_path).ok().as_ref()
                        == Some(&interface.manifest_digest)
                    && digest_path(&materialized.library_path).ok().as_ref()
                        == Some(&interface.library_digest)
            })
            .ok_or_else(|| {
                protocol(format!(
                    "frozen native interface {} has no exact materialized artifact",
                    interface.name
                ))
            })?;
        interface.manifest_path = materialized.manifest_path.clone();
        interface.library_path = materialized.library_path.clone();
    }
    for artifact in &mut handoff.project.mex_artifacts {
        let materialized = mex_artifacts
            .iter()
            .find(|materialized| {
                materialized.manifest.module_name == artifact.name
                    && digest_path(&runmat_mex::MexArtifactManifest::path_for_module(
                        &materialized.module_path,
                    ))
                    .ok()
                    .as_ref()
                        == Some(&artifact.manifest_digest)
                    && digest_path(&materialized.module_path).ok().as_ref()
                        == Some(&artifact.module_digest)
            })
            .ok_or_else(|| {
                protocol(format!(
                    "frozen MEX artifact {} has no exact materialized module",
                    artifact.name
                ))
            })?;
        artifact.module_path = materialized.module_path.clone();
        artifact.manifest_path =
            runmat_mex::MexArtifactManifest::path_for_module(&materialized.module_path);
    }
    for artifact in &mut handoff.project.java_artifacts {
        let materialized = java_artifacts
            .iter()
            .find(|materialized| {
                digest_path(&materialized.path).ok().as_ref() == Some(&artifact.digest)
            })
            .ok_or_else(|| {
                protocol(format!(
                    "frozen Java artifact {} has no exact materialized archive",
                    artifact.name
                ))
            })?;
        artifact.path = materialized.path.clone();
    }
    for artifact in &mut handoff.project.python_artifacts {
        let materialized = python_artifacts
            .iter()
            .find(|materialized| {
                materialized.logical_name == artifact.name
                    && materialized.module == artifact.module
                    && digest_path(&materialized.path).ok().as_ref() == Some(&artifact.digest)
            })
            .ok_or_else(|| {
                protocol(format!(
                    "frozen Python artifact {} has no exact materialized wheel",
                    artifact.name
                ))
            })?;
        artifact.path = materialized.path.clone();
    }
    handoff.validate().map_err(protocol)
}

fn digest_path(path: &Path) -> NativeExecutionResult<runmat_package::ContentDigest> {
    std::fs::read(path)
        .map(|bytes| runmat_package::ContentDigest::sha256(&bytes))
        .map_err(protocol)
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
        ExecutableForm, ExecutionBundleBuilder, ForeignArtifactClosure, LogicalObject,
        ObjectNamespace, ProgramBuildRecipe,
    };
    use runmat_mex::{
        MexArtifactManifest, MexBuild, MEX_ARTIFACT_MANIFEST_MEDIA_TYPE, MEX_MODULE_MEDIA_TYPE,
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
                runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
                runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
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
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let bundle = ExecutionBundleBuilder::native(&project, revision)
            .unwrap()
            .with_materialized_program(
                recipe,
                ExecutableForm::InterpreterBytecodeV2,
                runmat_vm::encode_interpreter_program_v2(&runmat_vm::FunctionRegistry::default())
                    .unwrap(),
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
                runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
                runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
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
        let python_bundle = runmat_python::PythonArtifactBundle::new(
            runmat_python::PythonEnvironmentIdentity {
                implementation: "cpython".into(),
                version: runmat_python::PythonVersion {
                    major: 3,
                    minor: 12,
                    patch: 0,
                },
                abi_tag: "cp312".into(),
                platform_tag: "fixture".into(),
                execution_mode: runmat_python::PythonExecutionMode::OutOfProcess,
            },
            Vec::new(),
        )
        .unwrap();
        let python = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            "python/artifacts.json",
            runmat_python::PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE,
            python_bundle.canonical_bytes().unwrap(),
        )
        .unwrap();
        let mex_source = temp.path().join("materialized_mex.c");
        std::fs::write(
            &mex_source,
            r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs > 0) plhs[0] = mxCreateDoubleScalar(7.0);
}
"#,
        )
        .unwrap();
        let mex_output = MexBuild::new(&mex_source, temp.path()).compile().unwrap();
        let mex_manifest_bytes = std::fs::read(&mex_output.manifest).unwrap();
        let mex_module_bytes = std::fs::read(&mex_output.module).unwrap();
        let mex_manifest = MexArtifactManifest::from_canonical_bytes(&mex_manifest_bytes).unwrap();
        let mex_identity = mex_manifest.identity.to_string();
        let native_target = runmat_native_codegen::NativeTarget::current()
            .execution_identity()
            .unwrap();
        let native_object_format = native_target.object_format;
        let mex_root = format!("mex/{mex_identity}");
        let mex_filename = mex_output.module.file_name().unwrap().to_string_lossy();
        let mex_sidecar = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            format!("{mex_root}/{mex_filename}.runmat.json"),
            MEX_ARTIFACT_MANIFEST_MEDIA_TYPE,
            mex_manifest_bytes,
        )
        .unwrap();
        let mex_module = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            format!("{mex_root}/{mex_filename}"),
            MEX_MODULE_MEDIA_TYPE,
            mex_module_bytes.clone(),
        )
        .unwrap();
        let java_adapter =
            runmat_types::ForeignAdapterId::new(runmat_java::JAVA_ADAPTER_ID).unwrap();
        let java_artifact =
            runmat_types::ForeignArtifactIdentity::new(java_identity.to_string()).unwrap();
        let java_interop = runmat_types::InteropManifest {
            schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: vec![runmat_types::ForeignAdapterRequirement {
                adapter: java_adapter.clone(),
                minimum_version: runmat_java::JAVA_ADAPTER_VERSION,
                capabilities: runmat_types::CapabilitySet(BTreeSet::from([
                    runmat_types::CapabilityRequirement::ForeignRuntime,
                ])),
                execution_stack: runmat_types::ExecutionStackRequirement::Process,
                artifact_identities: vec![java_artifact.clone()],
            }],
            adapter_contracts: Vec::new(),
        };
        let python_adapter =
            runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID).unwrap();
        let python_artifacts = python_bundle
            .artifact_identities()
            .into_iter()
            .map(runmat_types::ForeignArtifactIdentity::new)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let python_interop = runmat_types::InteropManifest {
            schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: vec![runmat_types::ForeignAdapterRequirement {
                adapter: python_adapter.clone(),
                minimum_version: runmat_python::PYTHON_ADAPTER_VERSION,
                capabilities: runmat_types::CapabilitySet(BTreeSet::from([
                    runmat_types::CapabilityRequirement::ForeignRuntime,
                ])),
                execution_stack: runmat_types::ExecutionStackRequirement::Process,
                artifact_identities: python_artifacts.clone(),
            }],
            adapter_contracts: Vec::new(),
        };
        let interop = runmat_types::InteropManifest::merge([
            manifest.interop_manifest(),
            mex_manifest.interop_manifest(),
            java_interop,
            python_interop,
        ])
        .unwrap();
        let mut closures = vec![
            ForeignArtifactClosure::new(
                runmat_types::ForeignAdapterId::new(runmat_native_ffi::NATIVE_FFI_ADAPTER_ID)
                    .unwrap(),
                runmat_types::ForeignArtifactIdentity::new(identity.as_str()).unwrap(),
                vec![sidecar.descriptor.digest, library.descriptor.digest],
            )
            .unwrap(),
            ForeignArtifactClosure::new(java_adapter, java_artifact, vec![java.descriptor.digest])
                .unwrap(),
            ForeignArtifactClosure::new(
                runmat_types::ForeignAdapterId::new(runmat_mex::MEX_ADAPTER_ID).unwrap(),
                runmat_types::ForeignArtifactIdentity::new(mex_identity.as_str()).unwrap(),
                vec![mex_sidecar.descriptor.digest, mex_module.descriptor.digest],
            )
            .unwrap(),
        ];
        closures.extend(python_artifacts.into_iter().map(|artifact| {
            ForeignArtifactClosure::new(
                python_adapter.clone(),
                artifact,
                vec![python.descriptor.digest],
            )
            .unwrap()
        }));
        let recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision.clone(),
            entrypoint: "fixture".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "compiled".into(),
            target: runmat_execution_artifact::ProgramTarget::native("host", native_target),
            interop,
            accelerators: Vec::new(),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let mut builder = ExecutionBundleBuilder::native(&project, revision)
            .unwrap()
            .with_compiled_package_closure();
        for object in [sidecar, library, java, python, mex_sidecar, mex_module] {
            builder = builder.with_foreign_object(object).unwrap();
        }
        for closure in closures {
            builder = builder.with_foreign_artifact_closure(closure).unwrap();
        }
        let bundle = builder
            .with_materialized_program(
                recipe,
                ExecutableForm::NativeObjectV1,
                runmat_execution_artifact::NativeObjectPayload::new(
                    native_object_format,
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
        assert_eq!(materialized.python_bundle(), Some(&python_bundle));
        assert!(std::fs::metadata(&java.path)
            .unwrap()
            .permissions()
            .readonly());
        let mex = materialized.mex_artifact(&mex_identity).unwrap();
        assert_eq!(std::fs::read(&mex.module_path).unwrap(), mex_module_bytes);
        assert!(std::fs::metadata(&mex.module_path)
            .unwrap()
            .permissions()
            .readonly());
        let mex_sidecar = MexArtifactManifest::path_for_module(&mex.module_path);
        assert_eq!(
            MexArtifactManifest::from_canonical_bytes(&std::fs::read(mex_sidecar).unwrap())
                .unwrap(),
            mex_manifest
        );
    }
}
