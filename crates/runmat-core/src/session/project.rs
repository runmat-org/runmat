use super::*;

impl RunMatSession {
    /// Install a validated, host-frozen project snapshot for subsequent execution.
    ///
    /// The session uses this exact graph and source catalog for the primary source,
    /// companion sources, static analysis, and dynamically loaded functions. This
    /// is the browser/server handoff boundary; it avoids rediscovery against a
    /// host filesystem that may not exist in the execution environment.
    pub fn install_project_handoff(
        &mut self,
        handoff: runmat_package::FrozenProjectHandoff,
    ) -> std::result::Result<
        runmat_package::ProjectRevision,
        runmat_package::FrozenProjectHandoffError,
    > {
        handoff.validate()?;
        #[cfg(not(target_arch = "wasm32"))]
        self.mex_runtime
            .clear(
                &runmat_runtime::user_functions::DynamicFunctionClearRequest::NativeExtensions,
                self.runtime_context.clone(),
            )
            .map_err(|error| {
                runmat_package::FrozenProjectHandoffError::Revision(format!(
                    "could not clear loaded MEX modules before installing project: {error}"
                ))
            })?;
        #[cfg(not(target_arch = "wasm32"))]
        self.install_project_foreign_artifacts(&handoff)?;
        let revision = handoff.revision();
        let program_revision = runmat_execution::ProgramRevision::new(
            runmat_execution::Digest::from_bytes(*revision.graph_digest.bytes()),
            runmat_execution::Digest::from_bytes(*revision.source_revision.bytes()),
            crate::program_environment(self.compat_mode),
        )
        .expect("validated project revision and Core environment are valid");
        self.runtime_context = self
            .runtime_context
            .clone()
            .with_program_revision(Some(program_revision));
        #[cfg(not(target_arch = "wasm32"))]
        self.generic_native_cache
            .publish_project_revision(Some(&revision))
            .map_err(|error| {
                runmat_package::FrozenProjectHandoffError::Revision(error.to_string())
            })?;
        self.project_handoff = Some(handoff);
        self.pending_companion_source_discovery = None;
        self.dynamic_function_cache
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clear();
        Ok(revision)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn install_project_foreign_artifacts(
        &self,
        handoff: &runmat_package::FrozenProjectHandoff,
    ) -> std::result::Result<(), runmat_package::FrozenProjectHandoffError> {
        self.mex_runtime.clear_installed_artifacts();
        self.python_adapter
            .clear_artifact_bundle()
            .map_err(project_artifact_error)?;
        let mut manifests = Vec::new();
        for interface in &handoff.project.native_interfaces {
            self.native_ffi_adapter
                .install_prepared_artifact(&interface.library_path, &interface.manifest_path)
                .map_err(project_artifact_error)?;
            let manifest =
                runmat_native_ffi::NativeInterfaceArtifactManifest::read(&interface.manifest_path)
                    .map_err(project_artifact_error)?;
            manifests.push(manifest.interop_manifest());
        }
        for artifact in &handoff.project.mex_artifacts {
            self.mex_runtime
                .install_artifact(&artifact.module_path)
                .map_err(project_artifact_error)?;
            let bytes = std::fs::read(&artifact.manifest_path).map_err(project_artifact_error)?;
            let manifest = runmat_mex::MexArtifactManifest::from_canonical_bytes(&bytes)
                .map_err(project_artifact_error)?;
            manifests.push(manifest.interop_manifest());
        }
        let mut java = handoff
            .project
            .java_artifacts
            .iter()
            .map(|artifact| {
                let bytes = std::fs::read(&artifact.path).map_err(project_artifact_error)?;
                Ok((
                    runmat_java::JavaArtifactIdentity::for_bytes(&bytes),
                    artifact.path.clone(),
                ))
            })
            .collect::<std::result::Result<Vec<_>, runmat_package::FrozenProjectHandoffError>>()?;
        java.sort_by(|left, right| left.0.cmp(&right.0));
        self.java_adapter
            .install_project_artifacts(&java)
            .map_err(project_artifact_error)?;
        if !java.is_empty() {
            manifests.push(runmat_types::InteropManifest {
                schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
                foreign_types: Vec::new(),
                adapters: vec![runmat_types::ForeignAdapterRequirement {
                    adapter: runmat_java::JAVA_ADAPTER_ID.into(),
                    minimum_version: runmat_java::JAVA_ADAPTER_VERSION,
                    capabilities: runmat_types::CapabilitySet(std::collections::BTreeSet::from([
                        runmat_types::CapabilityRequirement::ForeignRuntime,
                    ])),
                    artifact_identities: java
                        .iter()
                        .map(|(identity, _)| identity.to_string())
                        .collect(),
                }],
            });
        }
        if !handoff.project.python_artifacts.is_empty() {
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
                    })?
                    .to_owned();
                artifacts.push(runmat_python::PythonArtifactBundleEntry::wheel(
                    artifact.name.clone(),
                    artifact.module.clone(),
                    filename,
                    bytes,
                ));
            }
            let bundle = runmat_python::PythonArtifactBundle::artifacts_only(artifacts)
                .map_err(project_artifact_error)?;
            self.python_adapter
                .install_artifact_bundle(&bundle)
                .map_err(project_artifact_error)?;
            manifests.push(runmat_types::InteropManifest {
                schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
                foreign_types: Vec::new(),
                adapters: vec![runmat_types::ForeignAdapterRequirement {
                    adapter: runmat_python::PYTHON_ADAPTER_ID.into(),
                    minimum_version: runmat_python::PYTHON_ADAPTER_VERSION,
                    capabilities: runmat_types::CapabilitySet(std::collections::BTreeSet::from([
                        runmat_types::CapabilityRequirement::ForeignRuntime,
                    ])),
                    artifact_identities: bundle.artifact_identities().into_iter().collect(),
                }],
            });
        }
        let interop =
            runmat_types::InteropManifest::merge(manifests).map_err(project_artifact_error)?;
        self.foreign_runtime
            .admit(&interop)
            .map_err(project_artifact_error)?;
        Ok(())
    }

    /// Remove the host-frozen snapshot and restore normal project discovery.
    pub fn clear_project_handoff(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.generic_native_cache
            .publish_project_revision(None)
            .expect("native project dependency generation is valid");
        self.project_handoff = None;
        self.runtime_context = self.runtime_context.clone().with_program_revision(None);
        self.pending_companion_source_discovery = None;
        self.dynamic_function_cache
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clear();
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(error) = self.mex_runtime.clear(
            &runmat_runtime::user_functions::DynamicFunctionClearRequest::NativeExtensions,
            self.runtime_context.clone(),
        ) {
            tracing::warn!(%error, "could not clear loaded MEX modules while clearing project");
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.mex_runtime.clear_installed_artifacts();
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(error) = self.python_adapter.clear_artifact_bundle() {
            tracing::warn!(%error, "could not clear Python artifacts while clearing project");
        }
    }

    /// Return the revision currently installed at the session boundary.
    pub fn project_revision(&self) -> Option<runmat_package::ProjectRevision> {
        self.project_handoff
            .as_ref()
            .map(runmat_package::FrozenProjectHandoff::revision)
    }

    /// Borrow the validated snapshot installed for this session, if any.
    pub fn project_handoff(&self) -> Option<&runmat_package::FrozenProjectHandoff> {
        self.project_handoff.as_ref()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn project_artifact_error(
    error: impl std::fmt::Display,
) -> runmat_package::FrozenProjectHandoffError {
    runmat_package::FrozenProjectHandoffError::Revision(format!(
        "could not install frozen project artifact: {error}"
    ))
}
