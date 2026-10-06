//! Source replay workspace is temporary; original input reservations remain retained.
use super::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SourceWorkspaceKind {
    Replay,
    Transport,
}

#[derive(Clone, Copy)]
pub(super) struct SourceWorkspaceCharge {
    kind: SourceWorkspaceKind,
    resident_bytes: u64,
}

pub(in crate::analysis) struct SourceWorkspacePhase {
    active: Arc<std::sync::Mutex<Option<SourceWorkspaceCharge>>>,
}
impl SourceWorkspacePhase {
    /// Replay and ingestion are sequential, but share one uninterrupted lease.
    /// Installation runs under the same admission fence as inventory mutations.
    pub(super) fn with_transport<T>(
        &self,
        builder: &AnalysisPublicationBuilder,
        required: u64,
        install: impl FnOnce() -> Result<T, AnalysisPublicationError>,
    ) -> Result<T, AnalysisPublicationError> {
        if !Arc::ptr_eq(&self.active, &builder.source_phase) {
            return Err("source phase belongs to another publication".into());
        }
        let admission = builder
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?;
        if *admission {
            return Err("publication inventory is frozen for manifest preparation".into());
        }
        let previous = {
            let mut active = self
                .active
                .lock()
                .map_err(|_| "source phase lock poisoned")?;
            let previous = active
                .filter(|charge| charge.kind == SourceWorkspaceKind::Replay)
                .ok_or("source phase is not replay")?;
            *active = Some(SourceWorkspaceCharge {
                kind: SourceWorkspaceKind::Transport,
                resident_bytes: required,
            });
            previous
        };
        if let Err(error) = builder.admit(0) {
            *self
                .active
                .lock()
                .map_err(|_| "source phase lock poisoned")? = Some(previous);
            return Err(error);
        }
        install()
    }
}

impl Drop for SourceWorkspacePhase {
    fn drop(&mut self) {
        // A poisoned tracker remains a resource-authority error; never recover a
        // default budget or silently authorize another phase after that failure.
        if let Ok(mut active) = self.active.lock() {
            *active = None;
        }
    }
}

impl AnalysisPublicationBuilder {
    pub(in crate::analysis) fn begin_source_profile(
        &self,
        maximum: u64,
    ) -> Result<SourceWorkspacePhase, AnalysisPublicationError> {
        self.begin_source_workspace(maximum)
    }
    pub(in crate::analysis) fn begin_source_phase(
        &self,
    ) -> Result<SourceWorkspacePhase, AnalysisPublicationError> {
        self.begin_source_workspace(self.source_workspace_bytes)
    }

    pub(super) fn begin_catalog_phase(
        &self,
        objects: &[runmat_execution::value::ValueRef],
    ) -> Result<SourceWorkspacePhase, AnalysisPublicationError> {
        self.begin_source_workspace(self.planned_source_workspace(objects)?)
    }

    /// Hold this guard across the complete synchronous source operation. Source
    /// demand is charged alongside retained input/output and concurrent mutations.
    fn begin_source_workspace(
        &self,
        required: u64,
    ) -> Result<SourceWorkspacePhase, AnalysisPublicationError> {
        let admission = self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?;
        if *admission {
            return Err("publication inventory is frozen for manifest preparation".into());
        }
        let mut active = self
            .source_phase
            .lock()
            .map_err(|_| "source phase lock poisoned")?;
        if active.is_some() {
            return Err("a publication source phase is already active".into());
        }
        *active = Some(SourceWorkspaceCharge {
            kind: SourceWorkspaceKind::Replay,
            resident_bytes: required,
        });
        drop(active);
        let phase = SourceWorkspacePhase {
            active: self.source_phase.clone(),
        };
        self.admit(0)?;
        Ok(phase)
    }

    pub(super) fn active_source_workspace(&self) -> Result<u64, AnalysisPublicationError> {
        Ok(self
            .source_phase
            .lock()
            .map_err(|_| "source phase lock poisoned")?
            .map_or(0, |charge| charge.resident_bytes))
    }

    pub(super) fn source_replay_active(&self) -> Result<bool, AnalysisPublicationError> {
        Ok(self
            .source_phase
            .lock()
            .map_err(|_| "source phase lock poisoned")?
            .is_some_and(|charge| charge.kind == SourceWorkspaceKind::Replay))
    }

    pub(super) fn source_phase_active(&self) -> Result<bool, AnalysisPublicationError> {
        Ok(self
            .source_phase
            .lock()
            .map_err(|_| "source phase lock poisoned")?
            .is_some())
    }
}
