use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Weak;

use runmat_execution::{Digest, ExecutableSourceMapV2, ProgramRevision};
use runmat_types::{ProgramSourceId, SourceId};

use super::{invalid, ValueCodecError};

/// Exact, invocation-selected source facts needed after a Future is created.
/// No execution service, RuntimeContext, or process-local ID enters the wire.
#[derive(Clone, Debug)]
pub struct GraphSourceIssuer {
    session: Weak<()>,
    sources: HashMap<SourceId, Vec<SelectedSource>>,
    frame_sources: Vec<(SourceId, crate::CallFrameSourceOwner)>,
    _source_owners: Vec<crate::source_context::SelectedProgramSourceOwner>,
}

#[derive(Clone, Debug)]
struct SelectedSource {
    ordinal: ProgramSourceId,
    info: crate::source_context::SourceInfo,
    basis: crate::CallFrameSourceOwner,
}

/// One already-admitted V3 unit's source owner. The program key is derived
/// from its complete validated revision, not a name or graph digest.
#[derive(Clone, Debug)]
pub struct GraphSourceUnitFacts {
    program: Digest,
    session: Weak<()>,
    source_map: Arc<ExecutableSourceMapV2>,
    owner: crate::source_context::SelectedProgramSourceOwner,
    issued: Vec<(SourceId, ProgramSourceId, crate::source_context::SourceInfo)>,
    frame_sources: Vec<(SourceId, crate::CallFrameSourceOwner)>,
}

impl GraphSourceUnitFacts {
    /// Capture only source leases and already-verified issued records from the
    /// selected bound V3 unit. No execution port or RuntimeContext survives.
    pub fn capture_owner(
        owner: crate::source_context::SelectedProgramSourceOwner,
        revision: &ProgramRevision,
        source_map: Arc<ExecutableSourceMapV2>,
    ) -> Result<Self, ValueCodecError> {
        let program = revision
            .identity_digest()
            .map_err(|error| invalid(error.to_string()))?;
        source_map
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        if !owner.is_live() {
            return Err(invalid("selected V3 source owner has retired"));
        }
        if owner.portable_map() != source_map.as_ref() {
            return Err(invalid(
                "selected V3 source map differs from its bound issuer",
            ));
        }
        let mut issued = Vec::with_capacity(source_map.entries.len());
        let mut frame_sources = vec![None; source_map.entries.len()];
        for (ordinal, source, info) in owner.selected_sources() {
            let record = source_map
                .entries
                .get(usize::try_from(ordinal.0).map_err(|_| invalid("source ordinal overflow"))?)
                .ok_or_else(|| invalid("selected V3 source ordinal is absent"))?;
            if record.source_id != ordinal {
                return Err(invalid("selected V3 source ordinal differs from issuer"));
            }
            issued.push((source.source_id(), ordinal, info.clone()));
            let selected = &mut frame_sources
                [usize::try_from(ordinal.0).map_err(|_| invalid("source ordinal overflow"))?];
            if selected.is_some() {
                return Err(invalid("selected source ordinal is repeated"));
            }
            *selected = Some((
                source.source_id(),
                owner
                    .call_frame_source_owner_for(ordinal)
                    .ok_or_else(|| invalid("selected source has no issued frame basis"))?,
            ));
        }
        if issued.len() != source_map.entries.len() {
            return Err(invalid("selected V3 source table is incomplete"));
        }
        let frame_sources = frame_sources
            .into_iter()
            .map(|selected| {
                selected.ok_or_else(|| invalid("selected source frame basis is absent"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (alias, ordinal) in owner.durable_aliases() {
            let info = owner
                .issued_source_info_for(ordinal)
                .cloned()
                .ok_or_else(|| invalid("durable source alias has no issued record"))?;
            issued.push((alias.source_id(), ordinal, info));
        }
        Ok(Self {
            program,
            session: owner.session_lease(),
            source_map,
            owner,
            issued,
            frame_sources,
        })
    }

    pub fn program(&self) -> Digest {
        self.program
    }

    /// One selected product and one local issuance, irrespective of how many
    /// callable roots reach it. Equal canonical V3 bytes issued separately
    /// are deliberately not the same source unit.
    pub fn same_selected_issuance(&self, other: &Self) -> bool {
        self.program == other.program && self.owner.same_issuance(&other.owner)
    }
}

impl GraphSourceIssuer {
    pub fn capture_session(runtime: &crate::context::RuntimeContext) -> Self {
        Self {
            session: runtime.session_lease(),
            sources: HashMap::new(),
            frame_sources: Vec::new(),
            _source_owners: Vec::new(),
        }
    }

    /// Retain only the selected issuer's weak session and its source leases.
    /// The execution service and RuntimeContext are never captured.
    pub fn capture_owner(owner: &crate::source_context::SelectedProgramSourceOwner) -> Self {
        Self {
            session: owner.session_lease(),
            sources: HashMap::new(),
            frame_sources: Vec::new(),
            _source_owners: vec![owner.clone()],
        }
    }

    /// Build graph-global ordinals from exactly the selected V3 products.
    /// Both client and worker run this deterministic mapping independently;
    /// only admitted unit source maps may supply its records.
    pub fn capture_selected_units(
        units: &[GraphSourceUnitFacts],
    ) -> Result<(Self, ExecutableSourceMapV2), ValueCodecError> {
        let first = units
            .first()
            .ok_or_else(|| invalid("graph source selection is empty"))?;
        if units.len() > runmat_execution::value::GRAPH_MAX_SELECTED_UNITS {
            return Err(invalid("graph selects too many admitted V3 units"));
        }
        let session = first.session.clone();
        let mut records = Vec::new();
        let mut sources = HashMap::new();
        let mut frame_sources = Vec::new();
        for (position, unit) in units.iter().enumerate() {
            if !Weak::ptr_eq(&session, &unit.session) {
                return Err(invalid("selected graph units belong to different sessions"));
            }
            if units[..position]
                .iter()
                .any(|prior| prior.same_selected_issuance(unit))
            {
                return Err(invalid("selected graph source unit is repeated"));
            }
            let first_ordinal = u32::try_from(records.len())
                .map_err(|_| invalid("graph source ordinal count exceeds its ABI"))?;
            for record in &unit.source_map.entries {
                let ordinal = ProgramSourceId(
                    u32::try_from(records.len())
                        .map_err(|_| invalid("graph source ordinal count exceeds its ABI"))?,
                );
                let mut selected = record.clone();
                selected.source_id = ordinal;
                records.push(selected);
            }
            frame_sources.extend(unit.frame_sources.iter().cloned());
            for (source, original, info) in &unit.issued {
                let ordinal = ProgramSourceId(
                    first_ordinal
                        .checked_add(original.0)
                        .ok_or_else(|| invalid("graph source ordinal count exceeds its ABI"))?,
                );
                let basis = unit
                    .owner
                    .call_frame_source_owner_for(*original)
                    .ok_or_else(|| invalid("selected source has no issued frame basis"))?;
                let candidates: &mut Vec<SelectedSource> = sources.entry(*source).or_default();
                if let Some(prior) = candidates
                    .iter()
                    .find(|candidate| candidate.basis.same_issuance(&basis))
                {
                    if prior.ordinal != ordinal {
                        return Err(invalid(
                            "one source has conflicting ordinals within its issued table",
                        ));
                    }
                    continue;
                }
                candidates.push(SelectedSource {
                    ordinal,
                    info: info.clone(),
                    basis,
                });
            }
        }
        let map =
            ExecutableSourceMapV2::new(records).map_err(|error| invalid(error.to_string()))?;
        Ok((
            Self {
                session,
                sources,
                frame_sources,
                _source_owners: units.iter().map(|unit| unit.owner.clone()).collect(),
            },
            map,
        ))
    }

    pub(in crate::execution::value_codec) fn session_lease(&self) -> &Weak<()> {
        &self.session
    }

    pub(in crate::execution::value_codec) fn selected_unit_count(&self) -> usize {
        self._source_owners.len().max(1)
    }

    pub(crate) fn ensure_live(&self) -> Result<(), ValueCodecError> {
        self.session
            .upgrade()
            .map(|_| ())
            .ok_or_else(|| invalid("selected graph session has retired"))
    }

    pub(crate) fn source_ordinal(
        &self,
        source: SourceId,
        basis: Option<&crate::CallFrameSourceOwner>,
    ) -> Option<ProgramSourceId> {
        let candidates = self.sources.get(&source)?;
        match basis {
            Some(basis) => candidates
                .iter()
                .find(|candidate| candidate.basis.same_issuance(basis))
                .map(|candidate| candidate.ordinal),
            None if candidates.len() == 1 => Some(candidates[0].ordinal),
            None => None,
        }
    }

    pub(crate) fn frame_source(
        &self,
        ordinal: ProgramSourceId,
    ) -> Option<(SourceId, crate::CallFrameSourceOwner)> {
        self.frame_sources
            .get(usize::try_from(ordinal.0).ok()?)
            .cloned()
    }

    pub(crate) fn frame_diagnostic_source(
        &self,
        ordinal: ProgramSourceId,
    ) -> Option<crate::CallFrameDiagnosticSource> {
        let (source, basis) = self.frame_source(ordinal)?;
        let info = self.sources.get(&source)?.iter().find(|candidate| {
            candidate.ordinal == ordinal && candidate.basis.same_issuance(&basis)
        })?;
        Some(crate::CallFrameDiagnosticSource::new(
            Arc::clone(&info.info.name),
            Arc::clone(&info.info.text),
        ))
    }

    pub(crate) fn is_issued_in(&self, runtime: &crate::context::RuntimeContext) -> bool {
        self.session.upgrade().is_some() && Weak::ptr_eq(&self.session, &runtime.session_lease())
    }

    pub(crate) fn render_call_stack(&self, error: &crate::RuntimeError) -> Vec<String> {
        crate::execution::errors::render_source_call_stack_with(error, |source| {
            self.sources
                .get(&source)
                .filter(|candidates| candidates.len() == 1)
                .map(|candidates| candidates[0].info.clone())
        })
    }
}
