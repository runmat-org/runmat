use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Weak};

use runmat_execution::ExecutableSourceMapV2;
use runmat_types::{ProgramSourceId, SourceId};

use super::{
    portable_provenance::PortableSourceIdentity, source_info_in_context, ProgramSourceBinding,
    RuntimeSourceRef, SourceInfo,
};

/// The original issued source table, retained without its invocation's
/// RuntimeContext or execution ports. A definition may keep this owner after
/// its publishing request returns; it does not authenticate serialized code.
#[derive(Clone)]
pub struct SelectedProgramSourceOwner {
    facts: Rc<SelectedProgramSourceFacts>,
    durable_aliases: Rc<Vec<(RuntimeSourceRef, ProgramSourceId)>>,
    durable_ordinals: Rc<HashMap<RuntimeSourceRef, ProgramSourceId>>,
}

struct SelectedProgramSourceFacts {
    binding: Rc<ProgramSourceBinding>,
    session: Weak<()>,
    source_map: Arc<ExecutableSourceMapV2>,
    issued: Vec<(RuntimeSourceRef, SourceInfo)>,
    issued_ordinals: HashMap<RuntimeSourceRef, ProgramSourceId>,
}

impl std::fmt::Debug for SelectedProgramSourceOwner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectedProgramSourceOwner")
            .field("source_count", &self.facts.issued.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SelectedProgramSourceError {
    #[error("selected program source map is invalid: {0}")]
    InvalidMap(#[source] runmat_execution::ContractError),
    #[error("selected invocation has no bound portable source table")]
    MissingBinding,
    #[error("selected source table differs from its admitted portable map")]
    MapMismatch,
    #[error("selected program source ordinal {0:?} has retired")]
    Retired(ProgramSourceId),
    #[error("selected program source ordinal {0:?} differs from its issued record")]
    RecordMismatch(ProgramSourceId),
    #[error("published source alias is not paired with an issued program source")]
    InvalidAlias,
    #[error("published source alias belongs to a different or retired session")]
    ForeignSession,
}

impl SelectedProgramSourceOwner {
    /// Capture the already-issued binding. This never reissues a source ID or
    /// substitutes a later source map with matching display text.
    pub fn capture(
        runtime: &crate::context::RuntimeContext,
        map: &ExecutableSourceMapV2,
    ) -> Result<Self, SelectedProgramSourceError> {
        map.validate()
            .map_err(SelectedProgramSourceError::InvalidMap)?;
        let binding = runtime
            .selected_program_source_binding()
            .ok_or(SelectedProgramSourceError::MissingBinding)?;
        let source_map = binding
            .portable_map()
            .filter(|selected| selected.as_ref() == map)
            .cloned()
            .ok_or(SelectedProgramSourceError::MapMismatch)?;
        let mut issued = Vec::with_capacity(map.entries.len());
        let mut issued_ordinals = HashMap::with_capacity(map.entries.len());
        for record in &map.entries {
            let reference = binding
                .source_ref(record.source_id)
                .ok_or(SelectedProgramSourceError::Retired(record.source_id))?;
            let source = reference
                .resolve_in(runtime)
                .ok_or(SelectedProgramSourceError::Retired(record.source_id))?;
            let info = source_info_in_context(runtime, source)
                .ok_or(SelectedProgramSourceError::Retired(record.source_id))?;
            if info.name.as_ref() != record.display_name.as_str()
                || info.text.as_ref() != record.text.as_str()
                || info.portable_identity.as_ref()
                    != Some(&PortableSourceIdentity::from_record(record))
            {
                return Err(SelectedProgramSourceError::RecordMismatch(record.source_id));
            }
            if issued_ordinals
                .insert(reference, record.source_id)
                .is_some()
            {
                return Err(SelectedProgramSourceError::InvalidAlias);
            }
            issued.push((reference, info));
        }
        Ok(Self {
            facts: Rc::new(SelectedProgramSourceFacts {
                binding,
                session: runtime.session_lease(),
                source_map,
                issued,
                issued_ordinals,
            }),
            durable_aliases: Rc::new(Vec::new()),
            durable_ordinals: Rc::new(HashMap::new()),
        })
    }

    pub fn owns_source(&self, source: RuntimeSourceRef) -> bool {
        self.source_ordinal_for(source).is_some()
    }

    /// Compare a source fact with the exact issued or admitted durable ref
    /// without retaining its RuntimeContext. The opaque ref's SourceId is
    /// inspected only here, together with this owner's ordinal and live
    /// session proof; numeric SourceId equality alone is not authority.
    pub fn matches_issued_source_id(&self, source: RuntimeSourceRef, id: SourceId) -> bool {
        self.is_live() && source.id == id && self.source_ordinal_for(source).is_some()
    }

    pub fn same_issuance(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.facts.binding, &other.facts.binding)
            && Arc::ptr_eq(&self.facts.source_map, &other.facts.source_map)
    }

    pub fn source_ordinal_for(&self, source: RuntimeSourceRef) -> Option<ProgramSourceId> {
        self.facts
            .issued_ordinals
            .get(&source)
            .or_else(|| self.durable_ordinals.get(&source))
            .copied()
    }

    /// Constant-time exact issued presentation lookup for an admitted source
    /// ordinal. A durable alias maps to that same record, never new text.
    pub fn issued_source_info_for(&self, ordinal: ProgramSourceId) -> Option<&SourceInfo> {
        let index = usize::try_from(ordinal.0).ok()?;
        self.facts
            .source_map
            .entries
            .get(index)
            .filter(|record| record.source_id == ordinal)?;
        self.facts.issued.get(index).map(|(_, info)| info)
    }

    /// Local error-frame basis for a source actually issued by this admitted
    /// table. The token is never serialized and cannot select a unit alone.
    pub fn call_frame_source_owner_for(
        &self,
        ordinal: ProgramSourceId,
    ) -> Option<runmat_async::CallFrameSourceOwner> {
        self.facts.binding.frame_owner_for(ordinal)
    }

    /// Add only aliases paired by the successful canonical definition
    /// publication. The caller proves exact body/origin/definition identity
    /// before passing each issued and durable source reference here.
    pub fn with_durable_aliases(
        &self,
        runtime: &crate::context::RuntimeContext,
        pairs: impl IntoIterator<Item = (RuntimeSourceRef, RuntimeSourceRef)>,
    ) -> Result<Self, SelectedProgramSourceError> {
        let issuing = self
            .facts
            .session
            .upgrade()
            .ok_or(SelectedProgramSourceError::ForeignSession)?;
        let requested = runtime
            .session_lease()
            .upgrade()
            .ok_or(SelectedProgramSourceError::ForeignSession)?;
        if !Arc::ptr_eq(&issuing, &requested) {
            return Err(SelectedProgramSourceError::ForeignSession);
        }
        let mut aliases = self.durable_aliases.as_ref().clone();
        let mut alias_ordinals = self.durable_ordinals.as_ref().clone();
        for (issued, durable) in pairs {
            let ordinal = self
                .facts
                .issued_ordinals
                .get(&issued)
                .copied()
                .ok_or(SelectedProgramSourceError::InvalidAlias)?;
            let source = durable
                .resolve_in(runtime)
                .ok_or(SelectedProgramSourceError::InvalidAlias)?;
            let info = source_info_in_context(runtime, source)
                .ok_or(SelectedProgramSourceError::InvalidAlias)?;
            let record = self
                .facts
                .source_map
                .entries
                .get(
                    usize::try_from(ordinal.0)
                        .map_err(|_| SelectedProgramSourceError::InvalidAlias)?,
                )
                .ok_or(SelectedProgramSourceError::InvalidAlias)?;
            if info.name.as_ref() != record.display_name.as_str()
                || info.text.as_ref() != record.text.as_str()
                || self
                    .source_ordinal_for(durable)
                    .is_some_and(|selected| selected != ordinal)
                || aliases
                    .iter()
                    .any(|(existing, selected)| *existing == durable && *selected != ordinal)
            {
                return Err(SelectedProgramSourceError::InvalidAlias);
            }
            if !aliases.contains(&(durable, ordinal)) {
                aliases.push((durable, ordinal));
                alias_ordinals.insert(durable, ordinal);
            }
        }
        let mut selected = self.clone();
        selected.durable_aliases = Rc::new(aliases);
        selected.durable_ordinals = Rc::new(alias_ordinals);
        Ok(selected)
    }

    pub fn portable_map(&self) -> &ExecutableSourceMapV2 {
        &self.facts.source_map
    }

    /// Exact issued records, indexed by their admitted portable ordinals.
    /// Consumers may project a graph-local source table without retaining the
    /// invocation context or reissuing any source lease.
    pub fn selected_sources(
        &self,
    ) -> impl Iterator<Item = (ProgramSourceId, RuntimeSourceRef, &SourceInfo)> {
        self.facts
            .source_map
            .entries
            .iter()
            .zip(&self.facts.issued)
            .map(|(record, (reference, info))| (record.source_id, *reference, info))
    }

    pub fn durable_aliases(
        &self,
    ) -> impl Iterator<Item = (RuntimeSourceRef, ProgramSourceId)> + '_ {
        self.durable_aliases.iter().copied()
    }

    pub fn is_live(&self) -> bool {
        self.facts.session.upgrade().is_some()
    }

    /// Compare an explicit runtime with this exact issuing session. This is
    /// only a lifetime/scope check; it does not authenticate an executable or
    /// turn a diagnostic source reference into callable authority.
    pub fn is_issued_in(&self, runtime: &crate::context::RuntimeContext) -> bool {
        self.is_live() && Weak::ptr_eq(&self.facts.session, &runtime.session_lease())
    }

    /// This exact source table, not merely this session, is active for the
    /// invocation. Dynamic foreign admission must not borrow a sibling
    /// selected unit's source provenance.
    pub fn is_bound_in(&self, runtime: &crate::context::RuntimeContext) -> bool {
        self.is_issued_in(runtime)
            && runtime
                .selected_program_source_binding()
                .is_some_and(|binding| Rc::ptr_eq(&binding, &self.facts.binding))
    }

    /// Re-enter this already-issued portable source table in its original
    /// session. A retained executable/foreign product may use the resulting
    /// short-lived context for callbacks or orderly shutdown. The owner
    /// remains code-port-free and cannot be rebound into another session.
    /// This installs source provenance only; the caller separately selects
    /// the executable/graph product and foreign host for that unit.
    pub fn bind_runtime(
        &self,
        runtime: crate::context::RuntimeContext,
    ) -> Result<crate::context::RuntimeContext, SelectedProgramSourceError> {
        if !self.is_issued_in(&runtime) {
            return Err(SelectedProgramSourceError::ForeignSession);
        }
        Ok(runtime.with_selected_program_source_binding(Rc::clone(&self.facts.binding)))
    }

    /// Issuing-session lifetime only. This token cannot authenticate a
    /// source ordinal or executable product without the owner's checked map.
    pub fn session_lease(&self) -> Weak<()> {
        self.facts.session.clone()
    }
}

#[cfg(test)]
#[path = "selected_program/tests.rs"]
mod tests;
