use super::*;

#[cfg(not(target_arch = "wasm32"))]
mod bytes;
#[cfg(not(target_arch = "wasm32"))]
mod replacement;

impl RunMatSession {
    /// Keep the exact validated project bytes bound to every foreign owner
    /// before installing a handoff that may load or invoke those bytes.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn attach_foreign_private_root(
        &mut self,
        root: &crate::foreign_artifacts::PrivateArtifactRoot,
    ) -> std::result::Result<(), RuntimeError> {
        self.native_ffi_adapter
            .attach_private_byte_lease(root.lease())?;
        self.python_adapter
            .attach_selected_byte_lease(root.lease())?;
        self.java_adapter.attach_selected_byte_lease(root.lease())
    }

    /// Capture the configured foreign policy for future selected products.
    /// No foreign object is read until an exact compiled unit establishes
    /// whether the Python runtime is required.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_selected_foreign_config(
        &mut self,
        config: std::sync::Arc<runmat_config::runtime::RunMatRuntimeConfig>,
    ) {
        self.selected_foreign_config = Some(config);
        self.prepared_project_foreign = None;
        self.prepared_selected_foreign_input = None;
    }

    /// Perform the one configured foreign preparation for this exact project
    /// and selected unit, after code analysis and before its V3 publication.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn prepare_selected_project_foreign(
        &mut self,
        inferred: &runmat_types::InteropManifest,
    ) -> std::result::Result<Option<crate::foreign_artifacts::SelectedForeignPreparation>, String>
    {
        inferred.validate().map_err(|error| error.to_string())?;
        let python = runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID)
            .map_err(|error| error.to_string())?;
        let python_required = inferred.adapters.iter().any(|item| item.adapter == python);
        // An admitted bundle retains its own policy, not the worker's ambient configuration.
        if self.prepared_selected_foreign_input.is_none() {
            self.refresh_selected_python_config();
        }
        let Some(handoff) = self.project_handoff.as_ref() else {
            if *inferred == runmat_types::InteropManifest::empty() {
                return Ok(None);
            }
            let config = self
                .selected_foreign_config
                .as_ref()
                .ok_or("selected runtime-only foreign execution requires captured configuration")?;
            let artifacts =
                crate::foreign_artifacts::prepare_runtime_only_for_selection(config, inferred)
                    .map_err(|error| error.to_string())?;
            return Ok(Some(
                crate::foreign_artifacts::SelectedForeignPreparation::runtime_only(
                    config.clone(),
                    std::sync::Arc::new(artifacts),
                )
                .map_err(|error| error.to_string())?,
            ));
        };
        if let Some(prepared) = self.prepared_selected_foreign_input.as_ref() {
            if !prepared.matches_handoff(handoff) {
                return Err("selected Direct bundle differs from installed project handoff".into());
            }
            if python_required && prepared.python_execution_mode().is_none() {
                return Err(
                    "selected Direct bundle has no authenticated Python execution mode".into(),
                );
            }
            return Ok(Some(
                crate::foreign_artifacts::SelectedForeignPreparation::Admitted(prepared.clone()),
            ));
        }
        if *inferred == runmat_types::InteropManifest::empty()
            && handoff.project.native_interfaces.is_empty()
            && handoff.project.mex_artifacts.is_empty()
            && handoff.project.java_artifacts.is_empty()
            && handoff.project.python_artifacts.is_empty()
        {
            return Ok(None);
        }
        if let Some(prepared) = self.prepared_project_foreign.as_ref() {
            if prepared.matches_selection(handoff, inferred) {
                return Ok(Some(
                    crate::foreign_artifacts::SelectedForeignPreparation::Configured(
                        prepared.clone(),
                    ),
                ));
            }
        }
        let config = self
            .selected_foreign_config
            .as_ref()
            .ok_or("selected project foreign artifacts require captured runtime configuration")?;
        let prepared = std::sync::Arc::new(
            crate::foreign_artifacts::PreparedProjectForeignArtifacts::prepare_for_selection(
                std::sync::Arc::new(handoff.clone()),
                config.clone(),
                inferred,
            )
            .map_err(|error| error.to_string())?,
        );
        self.prepared_project_foreign = Some(prepared.clone());
        Ok(Some(
            crate::foreign_artifacts::SelectedForeignPreparation::Configured(prepared),
        ))
    }

    /// Bind one already-validated Direct bundle's exact recipe and rebased
    /// handoff before the worker compiles its selected test/source product.
    /// The input owns authenticated object bytes; no mutable path is read here.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn attach_prepared_selected_foreign_input(
        &mut self,
        prepared: std::sync::Arc<crate::foreign_artifacts::PreparedSelectedForeignInput>,
    ) -> std::result::Result<(), runmat_package::FrozenProjectHandoffError> {
        let handoff = self.project_handoff.as_ref().ok_or_else(|| {
            runmat_package::FrozenProjectHandoffError::Revision(
                "selected Direct input requires an installed project handoff".into(),
            )
        })?;
        if !prepared.matches_handoff(handoff) {
            return Err(runmat_package::FrozenProjectHandoffError::Revision(
                "selected Direct input differs from installed project handoff".into(),
            ));
        }
        self.prepared_project_foreign = None;
        self.prepared_selected_foreign_input = Some(prepared);
        Ok(())
    }

    /// Attach foreign bytes/configuration prepared for the currently installed
    /// frozen handoff. Hosts call this after selected compilation determines
    /// Python demand and before publishing that selected product. It performs
    /// no filesystem access or foreign installation.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn attach_prepared_project_foreign_artifacts(
        &mut self,
        prepared: std::sync::Arc<crate::foreign_artifacts::PreparedProjectForeignArtifacts>,
    ) -> std::result::Result<(), runmat_package::FrozenProjectHandoffError> {
        let handoff = self.project_handoff.as_ref().ok_or_else(|| {
            runmat_package::FrozenProjectHandoffError::Revision(
                "prepared foreign artifacts require an installed project handoff".into(),
            )
        })?;
        if !prepared.matches_handoff(handoff) {
            return Err(runmat_package::FrozenProjectHandoffError::Revision(
                "prepared foreign artifacts differ from the installed project handoff".into(),
            ));
        }
        self.prepared_project_foreign = Some(prepared);
        self.prepared_selected_foreign_input = None;
        Ok(())
    }

    /// Install a validated, host-frozen project snapshot for subsequent execution.
    ///
    /// The session uses this exact graph and source catalog for the primary source,
    /// companion sources, static analysis, and dynamically loaded functions. This
    /// is the browser/server handoff boundary; it avoids rediscovery against a
    /// host filesystem that may not exist in the execution environment.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn install_project_handoff(
        &mut self,
        mut handoff: runmat_package::FrozenProjectHandoff,
    ) -> std::result::Result<
        runmat_package::ProjectRevision,
        runmat_package::FrozenProjectHandoffError,
    > {
        handoff.validate()?;
        let revision = handoff.revision();
        let program_revision = self.program_revision_for_handoff(&revision)?;
        let cache = self
            .generic_native_cache
            .prepare_project_revision(Some(&revision))
            .map_err(project_artifact_error)?;
        let private_bytes = bytes::ProjectPrivateBytes::materialize(&mut handoff)?;
        let host = crate::ForeignUnitHost::from_session_adapters(
            std::rc::Rc::clone(&self.foreign_runtime),
            std::rc::Rc::clone(&self.native_ffi_adapter),
            std::rc::Rc::clone(&self.java_adapter),
            std::rc::Rc::clone(&self.python_adapter),
            std::rc::Rc::clone(&self.mex_runtime),
        );
        let owners = host.selected_owners();
        let replacement = replacement::PreparedProjectForeignReplacement::prepare(
            &handoff,
            private_bytes
                .as_ref()
                .map(bytes::ProjectPrivateBytes::lease),
            owners.native,
            owners.java,
            owners.python,
        )?;
        owners
            .runtime
            .admit_with_prepared_selected_replacements(
                &replacement.interop,
                replacement.native.as_ref(),
                &replacement.mex,
                replacement.java.as_ref(),
                replacement.python.as_ref(),
            )
            .map_err(project_artifact_error)?;
        self.retire_project_native_extensions()
            .await
            .map_err(|error| {
                runmat_package::FrozenProjectHandoffError::Revision(format!(
                    "could not clear loaded MEX modules before installing project: {error}"
                ))
            })?;
        let retirement = replacement
            .retire_project_hosts(Some(&self.runtime_context))
            .await?;
        let cache = cache
            .admit_commit(&mut self.generic_native_cache)
            .map_err(project_artifact_error)?;
        let publication_fence = replacement
            .admit_commit(owners.runtime, owners.mex, retirement)?
            .commit();
        cache.commit();
        self.commit_project_source_handoff(handoff, program_revision);
        self.private_project_root = private_bytes.map(bytes::ProjectPrivateBytes::into_root);
        drop(publication_fence);
        Ok(revision)
    }

    /// Install only one admitted Direct recipe's foreign closure into this
    /// Session's existing adapter owners, then publish its project sources.
    /// The caller retains the authenticated materialization through shutdown.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn install_selected_project_handoff(
        &mut self,
        handoff: runmat_package::FrozenProjectHandoff,
        prepared: std::sync::Arc<crate::foreign_artifacts::PreparedSelectedForeignInput>,
        manifest: &runmat_types::InteropManifest,
        materialized: &dyn crate::foreign_artifacts::MaterializedForeignView,
    ) -> std::result::Result<
        runmat_package::ProjectRevision,
        runmat_package::FrozenProjectHandoffError,
    > {
        handoff.validate()?;
        manifest.validate().map_err(project_artifact_error)?;
        if !prepared.matches_handoff(&handoff) || prepared.interop() != manifest {
            return Err(project_artifact_error(
                "selected Direct input differs from its exact handoff or recipe interop",
            ));
        }
        match (prepared.policy_object(), materialized.foreign_policy()) {
            (Some(object), Some(policy)) => {
                crate::foreign_artifacts::decode_admitted_policy(object)
                    .map_err(project_artifact_error)?;
                let supplied = crate::foreign_artifacts::capture_foreign_policy_object(policy)
                    .map_err(project_artifact_error)?;
                if supplied.descriptor != object.descriptor
                    || supplied.bytes.as_slice() != object.bytes.as_slice()
                {
                    return Err(project_artifact_error(
                        "selected Direct host policy differs from authenticated recipe policy",
                    ));
                }
            }
            (None, None) if *manifest == runmat_types::InteropManifest::empty() => {}
            _ => {
                return Err(project_artifact_error(
                    "selected Direct recipe lacks matching authenticated host policy",
                ))
            }
        }

        let host = crate::ForeignUnitHost::from_session_adapters(
            std::rc::Rc::clone(&self.foreign_runtime),
            std::rc::Rc::clone(&self.native_ffi_adapter),
            std::rc::Rc::clone(&self.java_adapter),
            std::rc::Rc::clone(&self.python_adapter),
            std::rc::Rc::clone(&self.mex_runtime),
        );
        let plan =
            crate::foreign_artifacts::preflight_selected_interop(&host, manifest, materialized)
                .map_err(project_artifact_error)?;
        let revision = handoff.revision();
        let program_revision = self.program_revision_for_handoff(&revision)?;
        let cache = self
            .generic_native_cache
            .prepare_project_revision(Some(&revision))
            .map_err(project_artifact_error)?;
        crate::foreign_artifacts::prepare_selected_cuda(&plan).map_err(project_artifact_error)?;
        let replacement =
            crate::foreign_artifacts::PreparedSelectedHostReplacement::prepare(&host, plan)
                .map_err(project_artifact_error)?;
        self.retire_project_native_extensions()
            .await
            .map_err(|error| {
                runmat_package::FrozenProjectHandoffError::Revision(format!(
                    "could not clear loaded MEX modules before installing selected project: {error}"
                ))
            })?;
        let retirement = replacement
            .retire_project_hosts(Some(&self.runtime_context))
            .await
            .map_err(project_artifact_error)?;
        let cache = cache
            .admit_commit(&mut self.generic_native_cache)
            .map_err(project_artifact_error)?;
        let replacement = replacement
            .admit_commit_after_retirement(retirement)
            .map_err(project_artifact_error)?;
        let publication_fence = replacement.commit();
        cache.commit();
        self.commit_project_source_handoff(handoff, program_revision);
        self.prepared_selected_foreign_input = Some(prepared);
        drop(publication_fence);
        Ok(revision)
    }

    /// Admit an artifact-only selected closure into this Session's existing
    /// adapters. The caller retains its authenticated materialization and
    /// controls the worker lifetime. No project source is published; the
    /// staged selected MEX catalog replaces ambient name visibility exactly.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn install_selected_interop_without_handoff(
        &self,
        manifest: &runmat_types::InteropManifest,
        materialized: &dyn crate::foreign_artifacts::MaterializedForeignView,
    ) -> std::result::Result<(), crate::foreign_artifacts::SelectedForeignInstallError> {
        let host = crate::ForeignUnitHost::from_session_adapters(
            std::rc::Rc::clone(&self.foreign_runtime),
            std::rc::Rc::clone(&self.native_ffi_adapter),
            std::rc::Rc::clone(&self.java_adapter),
            std::rc::Rc::clone(&self.python_adapter),
            std::rc::Rc::clone(&self.mex_runtime),
        );
        let plan =
            crate::foreign_artifacts::preflight_selected_interop(&host, manifest, materialized)?;
        crate::foreign_artifacts::prepare_selected_cuda(&plan)?;
        let replacement =
            crate::foreign_artifacts::PreparedSelectedHostReplacement::prepare(&host, plan)?;
        self.retire_project_native_extensions().await?;
        let retirement = replacement
            .retire_project_hosts(Some(&self.runtime_context))
            .await?;
        let publication_fence = replacement
            .admit_commit_after_retirement(retirement)?
            .commit();
        drop(publication_fence);
        Ok(())
    }

    /// Browser project handoff has no native MEX lifecycle and remains synchronous.
    #[cfg(target_arch = "wasm32")]
    pub fn install_project_handoff(
        &mut self,
        handoff: runmat_package::FrozenProjectHandoff,
    ) -> std::result::Result<
        runmat_package::ProjectRevision,
        runmat_package::FrozenProjectHandoffError,
    > {
        handoff.validate()?;
        self.publish_validated_project_handoff(handoff)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline(never)]
    fn retire_project_native_extensions(
        &self,
    ) -> impl std::future::Future<Output = std::result::Result<(), RuntimeError>> + '_ {
        // This lifecycle body contains concrete foreign teardown futures. Do
        // not embed their full state in every project-install caller. Admit F
        // before construction and let the shared deferred poll owner construct
        // and pin it only after the caller's construction frame has unwound.
        let prepared = self
            .runtime_context
            .prepare_poll_scope()
            .and_then(|scope| scope.prepare_future())
            .map(|scope| {
                scope.defer_result(move || async move {
                    self.source_evaluator
                        .retire_selected_foreign_hosts(&self.runtime_context, &self.local_tasks)
                        .await?;
                    self.mex_runtime
                        .clear_gracefully(
                            &runmat_runtime::user_functions::DynamicFunctionClearRequest::NativeExtensions,
                            self.runtime_context.clone(),
                        )
                        .await
                })
            });
        async move {
            self.local_tasks.participate_foreground(prepared?).await??;
            Ok(())
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn publish_validated_project_handoff(
        &mut self,
        handoff: runmat_package::FrozenProjectHandoff,
    ) -> std::result::Result<
        runmat_package::ProjectRevision,
        runmat_package::FrozenProjectHandoffError,
    > {
        self.publish_project_source_handoff(handoff)
    }

    #[cfg(target_arch = "wasm32")]
    fn publish_project_source_handoff(
        &mut self,
        handoff: runmat_package::FrozenProjectHandoff,
    ) -> std::result::Result<
        runmat_package::ProjectRevision,
        runmat_package::FrozenProjectHandoffError,
    > {
        let revision = handoff.revision();
        let program_revision = self.program_revision_for_handoff(&revision)?;
        self.commit_project_source_handoff(handoff, program_revision);
        Ok(revision)
    }

    fn program_revision_for_handoff(
        &self,
        revision: &runmat_package::ProjectRevision,
    ) -> std::result::Result<
        runmat_execution::ProgramRevision,
        runmat_package::FrozenProjectHandoffError,
    > {
        runmat_execution::ProgramRevision::new(
            runmat_execution::Digest::from_bytes(*revision.graph_digest.bytes()),
            runmat_execution::Digest::from_bytes(*revision.source_revision.bytes()),
            crate::program_environment(self.compat_mode),
        )
        .map_err(project_revision_error)
    }

    fn commit_project_source_handoff(
        &mut self,
        handoff: runmat_package::FrozenProjectHandoff,
        program_revision: runmat_execution::ProgramRevision,
    ) {
        self.runtime_context = self
            .runtime_context
            .clone()
            .with_program_revision(Some(program_revision));
        self.source_evaluator.set_project_sources(Some(&handoff));
        self.project_handoff = Some(handoff);
        self.checked_fea_sources = Default::default();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.private_project_root = None;
            self.prepared_project_foreign = None;
            self.prepared_selected_foreign_input = None;
        }
        self.pending_companion_source_discovery = None;
        self.dynamic_function_cache.clear();
    }

    /// Remove the host-frozen snapshot and restore normal project discovery.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn clear_project_handoff(
        &mut self,
    ) -> std::result::Result<(), runmat_package::FrozenProjectHandoffError> {
        let Some(_) = self.project_handoff.as_ref() else {
            // Working-source checks also retain project-scoped products when
            // no frozen handoff is installed. Clearing the project retires
            // those products without starting an unnecessary host retirement.
            self.checked_fea_sources = Default::default();
            return Ok(());
        };
        let cache = self
            .generic_native_cache
            .prepare_project_revision(None)
            .map_err(project_revision_error)?;
        let host = crate::ForeignUnitHost::from_session_adapters(
            std::rc::Rc::clone(&self.foreign_runtime),
            std::rc::Rc::clone(&self.native_ffi_adapter),
            std::rc::Rc::clone(&self.java_adapter),
            std::rc::Rc::clone(&self.python_adapter),
            std::rc::Rc::clone(&self.mex_runtime),
        );
        let replacement =
            crate::foreign_artifacts::PreparedSelectedHostReplacement::prepare_clear(&host)
                .map_err(project_artifact_error)?;
        self.retire_project_native_extensions()
            .await
            .map_err(|error| {
                runmat_package::FrozenProjectHandoffError::Revision(format!(
                    "could not clear loaded MEX modules before clearing project: {error}"
                ))
            })?;
        let retirement = replacement
            .retire_project_hosts(Some(&self.runtime_context))
            .await
            .map_err(project_artifact_error)?;
        let cache = cache
            .admit_commit(&mut self.generic_native_cache)
            .map_err(project_revision_error)?;
        let replacement = replacement
            .admit_commit_after_retirement(retirement)
            .map_err(project_artifact_error)?;
        let publication_fence = replacement.commit();
        cache.commit();
        self.forget_project_handoff();
        drop(publication_fence);
        Ok(())
    }

    /// Browser project handoff has no native MEX lifecycle and remains synchronous.
    #[cfg(target_arch = "wasm32")]
    pub fn clear_project_handoff(
        &mut self,
    ) -> std::result::Result<(), runmat_package::FrozenProjectHandoffError> {
        self.forget_project_handoff();
        Ok(())
    }

    fn forget_project_handoff(&mut self) {
        self.project_handoff = None;
        self.checked_fea_sources = Default::default();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.private_project_root = None;
            self.prepared_project_foreign = None;
            self.prepared_selected_foreign_input = None;
        }
        self.source_evaluator.set_project_sources(None);
        self.runtime_context = self.runtime_context.clone().with_program_revision(None);
        self.pending_companion_source_discovery = None;
        self.dynamic_function_cache.clear();
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

fn project_revision_error(
    error: impl std::fmt::Display,
) -> runmat_package::FrozenProjectHandoffError {
    runmat_package::FrozenProjectHandoffError::Revision(error.to_string())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "project/tests.rs"]
mod tests;
