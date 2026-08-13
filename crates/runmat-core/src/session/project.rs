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
        self.project_handoff = Some(handoff);
        self.pending_companion_source_discovery = None;
        self.dynamic_function_cache
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clear();
        Ok(revision)
    }

    /// Remove the host-frozen snapshot and restore normal project discovery.
    pub fn clear_project_handoff(&mut self) {
        self.project_handoff = None;
        self.runtime_context = self.runtime_context.clone().with_program_revision(None);
        self.pending_companion_source_discovery = None;
        self.dynamic_function_cache
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clear();
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
