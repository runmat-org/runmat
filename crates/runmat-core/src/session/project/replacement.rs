//! Prepare exact foreign project inventories without changing live adapters.

use super::project_artifact_error;
use runmat_package::{FrozenProjectHandoff, FrozenProjectHandoffError};
use runmat_runtime::foreign::{
    ForeignRuntime, JavaAdapter, MexArtifactExpectation, MexRuntimeSession, NativeFfiAdapter,
    NativeProjectArtifactExpectation, PreparedHostPolicyChanges, PreparedJavaProjectReplacement,
    PreparedMexArtifactCatalog, PreparedNativeProjectReplacement, PreparedPythonProjectReplacement,
    PythonAdapter, ReadyJavaProjectReplacement, ReadyMexArtifactCatalog,
    ReadyNativeProjectReplacement, ReadyPythonProjectReplacement, VerifiedJavaProjectRetirement,
    VerifiedNativeProjectRetirement, VerifiedPythonProjectRetirement,
};
use runmat_types::{
    CapabilityRequirement, CapabilitySet, ExecutionStackRequirement, ForeignAdapterId,
    ForeignAdapterRequirement, InteropManifest, INTEROP_MANIFEST_SCHEMA_VERSION,
};
use std::collections::BTreeSet;

pub(super) struct PreparedProjectForeignReplacement<'a> {
    pub(super) native: Option<PreparedNativeProjectReplacement<'a>>,
    pub(super) mex: PreparedMexArtifactCatalog,
    pub(super) java: Option<PreparedJavaProjectReplacement<'a>>,
    pub(super) python: Option<PreparedPythonProjectReplacement<'a>>,
    pub(super) interop: InteropManifest,
}

pub(super) struct VerifiedProjectHostRetirement<'a> {
    native: Option<VerifiedNativeProjectRetirement<'a>>,
    java: Option<VerifiedJavaProjectRetirement<'a>>,
    python: Option<VerifiedPythonProjectRetirement<'a>>,
}

impl<'a> PreparedProjectForeignReplacement<'a> {
    pub(super) async fn retire_project_hosts(
        &self,
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
    ) -> Result<VerifiedProjectHostRetirement<'a>, FrozenProjectHandoffError> {
        let native = if let Some(staged) = self.native.as_ref() {
            Some(
                staged
                    .retire_owner(runtime)
                    .await
                    .map_err(project_artifact_error)?,
            )
        } else {
            None
        };
        let java = self
            .java
            .as_ref()
            .map(PreparedJavaProjectReplacement::retire_owner)
            .transpose()
            .map_err(project_artifact_error)?;
        let python = if let Some(staged) = self.python.as_ref() {
            Some(
                staged
                    .retire_owner(runtime)
                    .await
                    .map_err(project_artifact_error)?,
            )
        } else {
            None
        };
        Ok(VerifiedProjectHostRetirement {
            native,
            java,
            python,
        })
    }

    /// Hold local inventory guards before the shared registry guard. The
    /// resulting owner performs one synchronous, infallible publication.
    pub(super) fn admit_commit(
        self,
        runtime: &'a ForeignRuntime,
        mex: &'a MexRuntimeSession,
        retirement: VerifiedProjectHostRetirement<'a>,
    ) -> Result<ReadyProjectForeignReplacement<'a>, FrozenProjectHandoffError> {
        let mut host_changes = Vec::with_capacity(3);
        if let Some(native) = self.native.as_ref() {
            if let Some(change) = native.host_policy_change() {
                host_changes.push(change);
            }
        }
        if let Some(java) = self.java.as_ref() {
            host_changes.push(java.host_restart_change());
        }
        if let Some(python) = self.python.as_ref() {
            host_changes.push(python.host_policy_change());
        }
        let native = match (self.native, retirement.native) {
            (Some(staged), Some(ticket)) => Some(
                staged
                    .admit_commit_after_retirement(ticket)
                    .map_err(project_artifact_error)?,
            ),
            (None, None) => None,
            _ => unreachable!("Native stage and retirement ticket are paired"),
        };
        let java = match (self.java, retirement.java) {
            (Some(staged), Some(ticket)) => Some(
                staged
                    .admit_commit_after_retirement(ticket)
                    .map_err(project_artifact_error)?,
            ),
            (None, None) => None,
            _ => unreachable!("Java stage and retirement ticket are paired"),
        };
        let mex = mex
            .admit_artifact_catalog(self.mex)
            .map_err(project_artifact_error)?;
        let python = match (self.python, retirement.python) {
            (Some(staged), Some(ticket)) => Some(
                staged
                    .admit_local_after_retirement(ticket)
                    .map_err(project_artifact_error)?,
            ),
            (None, None) => None,
            _ => unreachable!("Python stage and retirement ticket are paired"),
        };
        let host = (!host_changes.is_empty())
            .then(|| runtime.handles().prepare_host_policy_changes(&host_changes))
            .transpose()
            .map_err(project_artifact_error)?;
        let (python, other_host) = match (python, host) {
            (Some(python), Some(host)) => (
                Some(
                    python
                        .admit_with_shared_host(host)
                        .map_err(project_artifact_error)?,
                ),
                None,
            ),
            (None, host) => (None, host),
            (Some(_), None) => unreachable!("Python stage always requests host restart"),
        };
        Ok(ReadyProjectForeignReplacement {
            native,
            java,
            python,
            other_host,
            mex,
        })
    }

    pub(super) fn prepare(
        handoff: &FrozenProjectHandoff,
        private_lease: Option<runmat_execution_artifact::PrivateArtifactLease>,
        native_adapter: &'a NativeFfiAdapter,
        java_adapter: &'a JavaAdapter,
        python_adapter: &'a PythonAdapter,
    ) -> Result<Self, FrozenProjectHandoffError> {
        let mex = PreparedMexArtifactCatalog::prepare(handoff.project.mex_artifacts.iter().map(
            |artifact| MexArtifactExpectation {
                module: &artifact.module_path,
                manifest: &artifact.manifest_path,
                module_digest: runmat_execution::Digest::from_bytes(
                    *artifact.module_digest.bytes(),
                ),
                manifest_digest: runmat_execution::Digest::from_bytes(
                    *artifact.manifest_digest.bytes(),
                ),
            },
        ))
        .map_err(project_artifact_error)?;
        let mex = match (
            handoff.project.mex_artifacts.is_empty(),
            private_lease.as_ref(),
        ) {
            (false, Some(lease)) => mex.with_private_byte_lease(lease.clone()),
            _ => mex,
        };
        let native_next = !handoff.project.native_interfaces.is_empty();
        let native = (native_next
            || native_adapter
                .has_project_artifacts()
                .map_err(project_artifact_error)?)
        .then(|| {
            native_adapter.prepare_project_replacement(
                handoff.project.native_interfaces.iter().map(|interface| {
                    NativeProjectArtifactExpectation {
                        library: &interface.library_path,
                        manifest: &interface.manifest_path,
                        library_digest: runmat_execution::Digest::from_bytes(
                            *interface.library_digest.bytes(),
                        ),
                        manifest_digest: runmat_execution::Digest::from_bytes(
                            *interface.manifest_digest.bytes(),
                        ),
                    }
                }),
            )
        })
        .transpose()
        .map_err(project_artifact_error)?;
        let native = native
            .map(|staged| match (native_next, private_lease.as_ref()) {
                (true, Some(lease)) => staged.with_private_byte_lease(lease.clone()),
                _ => Ok(staged.clear_private_byte_lease()),
            })
            .transpose()
            .map_err(project_artifact_error)?;
        let mut manifests = native.as_ref().map_or_else(Vec::new, |staged| {
            staged.interop_manifests().cloned().collect::<Vec<_>>()
        });
        manifests.extend(mex.interop_manifests().cloned());

        let mut jars = handoff
            .project
            .java_artifacts
            .iter()
            .map(|artifact| {
                let bytes = std::fs::read(&artifact.path).map_err(project_artifact_error)?;
                let digest = runmat_package::ContentDigest::sha256(&bytes);
                if digest != artifact.digest {
                    return Err(project_artifact_error(format!(
                        "Java artifact `{}` changed after project resolution: expected {}, found {}",
                        artifact.name, artifact.digest, digest
                    )));
                }
                Ok((
                    runmat_java::JavaArtifactIdentity::for_bytes(&bytes),
                    artifact.path.clone(),
                ))
            })
            .collect::<Result<Vec<_>, FrozenProjectHandoffError>>()?;
        jars.sort_by(|left, right| left.0.cmp(&right.0));
        let java_next = !jars.is_empty();
        let java = (java_next
            || java_adapter
                .has_project_artifacts()
                .map_err(project_artifact_error)?)
        .then(|| java_adapter.prepare_project_replacement(&jars))
        .transpose()
        .map_err(project_artifact_error)?;
        let java = java.map(|staged| match (java_next, private_lease.as_ref()) {
            (true, Some(lease)) => staged.with_private_byte_lease(lease.clone()),
            _ => staged.clear_private_byte_lease(),
        });
        if !jars.is_empty() {
            let staged = java.as_ref().ok_or_else(|| {
                project_artifact_error("new Java artifacts require a staged owner")
            })?;
            manifests.push(adapter_manifest(
                runmat_java::JAVA_ADAPTER_ID,
                runmat_java::JAVA_ADAPTER_VERSION,
                ExecutionStackRequirement::Process,
                staged
                    .artifact_identities()
                    .map(|identity| {
                        runmat_types::ForeignArtifactIdentity::new(identity.to_string())
                            .map_err(project_artifact_error)
                    })
                    .collect::<Result<_, _>>()?,
            )?);
        }

        let python_bundle = python_bundle(handoff)?;
        let python_next = python_bundle.is_some();
        let python = (python_next
            || python_adapter
                .has_project_artifacts()
                .map_err(project_artifact_error)?)
        .then(|| python_adapter.prepare_project_replacement(python_bundle.as_ref()))
        .transpose()
        .map_err(project_artifact_error)?;
        let python = python.map(|staged| match (python_next, private_lease) {
            (true, Some(lease)) => staged.with_private_byte_lease(lease),
            _ => staged.clear_private_byte_lease(),
        });
        if python_bundle.is_some() {
            let staged = python.as_ref().ok_or_else(|| {
                project_artifact_error("new Python artifacts require a staged owner")
            })?;
            manifests.push(adapter_manifest(
                runmat_python::PYTHON_ADAPTER_ID,
                runmat_python::PYTHON_ADAPTER_VERSION,
                ExecutionStackRequirement::Process,
                staged.artifact_identities().cloned().collect(),
            )?);
        }
        let interop = InteropManifest::merge(manifests).map_err(project_artifact_error)?;
        Ok(Self {
            native,
            mex,
            java,
            python,
            interop,
        })
    }
}

pub(super) struct ReadyProjectForeignReplacement<'a> {
    native: Option<ReadyNativeProjectReplacement<'a>>,
    java: Option<ReadyJavaProjectReplacement<'a>>,
    python: Option<ReadyPythonProjectReplacement<'a>>,
    other_host: Option<PreparedHostPolicyChanges<'a>>,
    mex: ReadyMexArtifactCatalog<'a>,
}

impl<'a> ReadyProjectForeignReplacement<'a> {
    pub(super) fn commit(self) -> VerifiedProjectHostRetirement<'a> {
        let native = self.native.and_then(ReadyNativeProjectReplacement::commit);
        let java = self.java.and_then(ReadyJavaProjectReplacement::commit);
        let python = self.python.and_then(ReadyPythonProjectReplacement::commit);
        if let Some(host) = self.other_host {
            host.commit();
        }
        self.mex.commit();
        VerifiedProjectHostRetirement {
            native,
            java,
            python,
        }
    }
}

fn python_bundle(
    handoff: &FrozenProjectHandoff,
) -> Result<Option<runmat_python::PythonArtifactBundle>, FrozenProjectHandoffError> {
    if handoff.project.python_artifacts.is_empty() {
        return Ok(None);
    }
    let mut artifacts = Vec::with_capacity(handoff.project.python_artifacts.len());
    for artifact in &handoff.project.python_artifacts {
        let bytes = std::fs::read(&artifact.path).map_err(project_artifact_error)?;
        let digest = runmat_package::ContentDigest::sha256(&bytes);
        if digest != artifact.digest {
            return Err(project_artifact_error(format!(
                "Python artifact `{}` changed after project resolution: expected {}, found {}",
                artifact.name, artifact.digest, digest
            )));
        }
        let filename = artifact
            .path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                project_artifact_error(format!(
                    "Python artifact `{}` has a non-Unicode filename",
                    artifact.name
                ))
            })?;
        artifacts.push(runmat_python::PythonArtifactBundleEntry::wheel(
            artifact.name.clone(),
            artifact.module.clone(),
            filename.to_owned(),
            bytes,
        ));
    }
    runmat_python::PythonArtifactBundle::artifacts_only(artifacts)
        .map(Some)
        .map_err(project_artifact_error)
}

fn adapter_manifest(
    id: &str,
    minimum_version: u32,
    execution_stack: ExecutionStackRequirement,
    artifact_identities: Vec<runmat_types::ForeignArtifactIdentity>,
) -> Result<InteropManifest, FrozenProjectHandoffError> {
    Ok(InteropManifest {
        schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
        foreign_types: Vec::new(),
        adapters: vec![ForeignAdapterRequirement {
            adapter: ForeignAdapterId::new(id).map_err(project_artifact_error)?,
            minimum_version,
            capabilities: CapabilitySet(BTreeSet::from([CapabilityRequirement::ForeignRuntime])),
            execution_stack,
            artifact_identities,
        }],
        adapter_contracts: Vec::new(),
    })
}
