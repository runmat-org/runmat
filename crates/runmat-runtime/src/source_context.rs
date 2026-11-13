use runmat_execution::identity::{Digest, DigestStream};
use runmat_thread_local::runmat_thread_local;
use runmat_types::SourceId;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

mod lexical;
mod portable_provenance;
mod current_overlay;
mod selected_program;
use current_overlay::CurrentSourceOverlay;
use lexical::SourceFileIdentity;
pub use lexical::{lease_nested_source_context_in_context, NestedSourceIdentityError};
pub use portable_provenance::{
    lease_portable_source_context_in_context, portable_source_identity_in_context,
    PortableSourceIdentity, PortableSourceIdentityError, PortableSourceRecordAdmission,
    ValidatedPortableSourceMap,
};
pub use selected_program::{SelectedProgramSourceError, SelectedProgramSourceOwner};

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub source_id: Option<SourceId>,
    pub name: Arc<str>,
    pub fullpath_name: Option<Arc<str>>,
    pub text: Arc<str>,
    lexical_file: Option<SourceFileIdentity>,
    portable_identity: Option<PortableSourceIdentity>,
}

#[derive(Debug)]
pub(crate) struct SourceCurrentProjection {
    base: Option<SourceInfo>,
    overlays: Vec<(Rc<Cell<bool>>, Option<SourceInfo>)>,
}

impl SourceCurrentProjection {
    pub(crate) fn new(base: Option<SourceInfo>) -> Self {
        Self {
            base,
            overlays: Vec::new(),
        }
    }

    pub(crate) fn snapshot(&self) -> Option<SourceInfo> {
        self.overlays
            .iter()
            .rev()
            .find(|(live, _)| live.get())
            .map_or_else(|| self.base.clone(), |(_, source)| source.clone())
    }

    fn push(&mut self, source: Option<SourceInfo>) -> Rc<Cell<bool>> {
        let live = Rc::new(Cell::new(true));
        self.overlays.push((Rc::clone(&live), source));
        live
    }

    fn retire(&mut self, live: &Rc<Cell<bool>>) {
        live.set(false);
        while self.overlays.last().is_some_and(|(live, _)| !live.get()) {
            self.overlays.pop();
        }
    }
}

/// Each async invocation owns this foreground view. A guard removes only its
/// own overlay, even when another guard finishes first.
#[derive(Debug)]
pub(crate) struct SourceCatalogProjection {
    base: Rc<HashMap<SourceId, SourceInfo>>,
    overlays: Vec<(usize, Rc<HashMap<SourceId, SourceInfo>>)>,
    next_token: Option<usize>,
}

impl SourceCatalogProjection {
    pub(crate) fn new(base: Rc<HashMap<SourceId, SourceInfo>>) -> Self {
        Self {
            base,
            overlays: Vec::new(),
            next_token: Some(0),
        }
    }

    pub(crate) fn snapshot_catalog(&self) -> Rc<HashMap<SourceId, SourceInfo>> {
        self.overlays
            .last()
            .map_or_else(|| Rc::clone(&self.base), |(_, catalog)| Rc::clone(catalog))
    }

    /// A parallel lab has its own catalog, not the parent's weak leases. The
    /// state snapshot has copied each still-live issued source into owned
    /// records; retain those records alongside the poll-local foreground view.
    pub(crate) fn fork_with_inherited_sources(&self, source: &crate::context::SourceState) -> Self {
        let mut catalog = self.snapshot_catalog();
        for id in source.issued_authorities.keys() {
            if let Some(info) = source.catalog.get(id) {
                // An issued identity cannot be replaced by an untyped
                // foreground row, even if a caller reused its integer.
                Rc::make_mut(&mut catalog).insert(*id, info.clone());
            }
        }
        Self::new(catalog)
    }

    fn current(&self) -> &HashMap<SourceId, SourceInfo> {
        self.overlays
            .last()
            .map_or_else(|| self.base.as_ref(), |(_, catalog)| catalog.as_ref())
    }

    fn push(
        &mut self,
        catalog: HashMap<SourceId, SourceInfo>,
    ) -> Result<usize, SourceCatalogAdmissionError> {
        let token = self
            .next_token
            .ok_or(SourceCatalogAdmissionError::ProjectionExhausted)?;
        self.next_token = token.checked_add(1);
        self.overlays.push((token, Rc::new(catalog)));
        Ok(token)
    }

    fn remove(&mut self, token: usize) {
        if let Some(index) = self.overlays.iter().position(|(owned, _)| *owned == token) {
            self.overlays.remove(index);
        }
    }
}

fn imported_identity(info: &SourceInfo) -> Digest {
    let mut digest = DigestStream::new();
    for field in [
        Some(info.name.as_ref()),
        info.fullpath_name.as_deref(),
        Some(info.text.as_ref()),
    ] {
        match field {
            Some(value) => {
                digest.update(&[1]);
                digest.update(&(value.len() as u64).to_le_bytes());
                digest.update(value.as_bytes());
            }
            None => digest.update(&[0]),
        }
    }
    lexical::update_identity_digest(info, &mut digest);
    digest.finish()
}

/// Provenance issued by the source owner, never reconstructed from a numeric
/// `SourceId` or carried in a portable artifact. Forks retain this exact
/// authority only for source records present in their captured snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeSourceAuthority {
    issuer: crate::context::RuntimeContextLocalId,
    content: Digest,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RuntimeSourceRef {
    id: SourceId,
    issuer: crate::context::RuntimeContextLocalId,
    content: Digest,
}

impl RuntimeSourceRef {
    /// Stable semantic content of this issued source. This is not a portable
    /// source identity: only the complete reference may authorize a lookup in
    /// its issuing runtime (or an admitted fork).
    pub fn semantic_content_revision(self) -> Digest {
        self.content
    }

    pub fn resolve_in(self, runtime: &crate::context::RuntimeContext) -> Option<SourceId> {
        let sources = runtime.state().source.borrow();
        let authority = sources.issued_authorities.get(&self.id).copied()?;
        if authority.issuer != self.issuer || authority.content != self.content {
            return None;
        }
        // The lease is immutable and its content digest was checked when
        // issued. A fork retains that verified digest in its admitted catalog.
        // Overlay imports are checked against it once at admission, not on
        // every bytecode/source lookup (which may occur for every call).
        let active = crate::context::legacy::active()
            .filter(|active| active.same_source_projection_owner(runtime));
        let catalog = active
            .as_ref()
            .and_then(crate::context::RuntimeContext::source_catalog_view)
            .or_else(|| runtime.source_catalog_view());
        let catalog = catalog.as_ref().map(|catalog| catalog.borrow());
        let catalog = catalog
            .as_ref()
            .map_or(sources.catalog.as_ref(), |view| view.current());
        if sources
            .leased_catalog
            .get(&self.id)
            .and_then(LeasedSource::live_info)
            .is_some()
        {
            return (!catalog.contains_key(&self.id)).then_some(self.id);
        }
        (sources.inherited_authorities.contains(&self.id)
            && sources.imported_identities.get(&self.id) == Some(&self.content)
            && catalog.contains_key(&self.id))
        .then_some(self.id)
    }

    pub fn source_id(self) -> SourceId {
        self.id
    }
}

pub fn issued_source_ref_in_context(
    runtime: &crate::context::RuntimeContext,
    id: SourceId,
) -> Option<RuntimeSourceRef> {
    let authority = *runtime
        .state()
        .source
        .borrow()
        .issued_authorities
        .get(&id)?;
    let reference = RuntimeSourceRef {
        id,
        issuer: authority.issuer,
        content: authority.content,
    };
    reference.resolve_in(runtime).map(|_| reference)
}

#[derive(Debug)]
pub(crate) struct LeasedSource {
    owner: Weak<LeaseInner>,
}

impl LeasedSource {
    fn live_info(&self) -> Option<SourceInfo> {
        self.owner.upgrade().map(|owner| {
            let mut info = owner.info.clone();
            info.portable_identity = owner.portable_identity.get().cloned();
            info
        })
    }
}

fn resolved_source_info(
    source: &crate::context::SourceState,
    catalog: &HashMap<SourceId, SourceInfo>,
    id: SourceId,
) -> Option<SourceInfo> {
    let leased = source
        .leased_catalog
        .get(&id)
        .and_then(LeasedSource::live_info);
    // An untyped foreground import can use the same integer, but it cannot
    // decide which source an issued lease denotes. Fail closed rather than
    // attributing either caller's span to the other's text.
    if leased.is_some() && catalog.contains_key(&id) {
        return None;
    }
    leased.or_else(|| catalog.get(&id).cloned())
}

#[derive(Debug)]
struct LeaseInner {
    info: SourceInfo,
    authority: RuntimeSourceAuthority,
    state: Weak<crate::context::RuntimeContextState>,
    portable_identity: std::cell::OnceCell<PortableSourceIdentity>,
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let Some(id) = self.info.source_id else {
            return;
        };
        // A lease may be the last owner dropped while an outer catalog
        // operation holds this RefCell. The weak row retains no source text;
        // skip eager removal and let the next issue prune it instead of
        // panicking from a destructor.
        let Ok(mut sources) = state.source.try_borrow_mut() else {
            return;
        };
        if sources
            .leased_catalog
            .get(&id)
            .is_some_and(|entry| std::ptr::eq(entry.owner.as_ptr(), self))
        {
            sources.leased_catalog.remove(&id);
            if sources.issued_authorities.get(&id) == Some(&self.authority) {
                sources.issued_authorities.remove(&id);
            }
        }
    }
}

/// Session-issued provenance. Clones retain text; the last clone retires only
/// this record, never an unrelated foreground catalog or later source.
#[derive(Clone, Debug)]
pub struct RuntimeSourceLease(Rc<LeaseInner>);

/// The immutable ordinal-to-session projection owned for one invocation.
/// Keeping the leases here prevents a pending call from losing its source text.
#[derive(Debug)]
pub(crate) struct ProgramSourceBinding {
    sources: HashMap<runmat_types::ProgramSourceId, RuntimeSourceRef>,
    _leases: Vec<RuntimeSourceLease>,
    portable_map: Option<Arc<runmat_execution::ExecutableSourceMapV2>>,
    frame_owner: runmat_async::CallFrameSourceOwner,
}

impl ProgramSourceBinding {
    pub(crate) fn frame_owner_for(
        &self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Option<runmat_async::CallFrameSourceOwner> {
        self.sources
            .contains_key(&ordinal)
            .then(|| self.frame_owner.clone())
    }

    pub(crate) fn get(&self, ordinal: runmat_types::ProgramSourceId) -> Option<SourceId> {
        self.sources.get(&ordinal).map(|source| source.source_id())
    }

    pub(crate) fn source_ref(
        &self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Option<RuntimeSourceRef> {
        self.sources.get(&ordinal).copied()
    }

    pub(crate) fn source_ordinal(
        &self,
        source: RuntimeSourceRef,
    ) -> Option<runmat_types::ProgramSourceId> {
        self.sources
            .iter()
            .find_map(|(ordinal, selected)| (*selected == source).then_some(*ordinal))
    }

    pub(crate) fn portable_map(&self) -> Option<&Arc<runmat_execution::ExecutableSourceMapV2>> {
        self.portable_map.as_ref()
    }

    pub(crate) fn retain_portable_map(
        &mut self,
        admitted: &ValidatedPortableSourceMap,
    ) -> Result<(), PortableSourceIdentityError> {
        let map = admitted.map_arc();
        for record in &map.entries {
            let selected = admitted
                .record(record.source_id)
                .ok_or(PortableSourceIdentityError::RecordMismatch)?;
            let lease = self
                ._leases
                .iter()
                .find(|lease| self.sources.get(&record.source_id) == Some(&lease.source_ref()))
                .ok_or(PortableSourceIdentityError::RecordMismatch)?;
            portable_provenance::attach_admitted_identity(lease, &selected)?;
        }
        self.portable_map = Some(map);
        Ok(())
    }

    /// A worker owns copied source records in its SourceState catalog. Its
    /// projection therefore retains no lease on the parent's source issuer.
    pub(crate) fn fork_snapshot(&self) -> Self {
        Self {
            sources: self.sources.clone(),
            _leases: Vec::new(),
            portable_map: self.portable_map.clone(),
            frame_owner: self.frame_owner.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProgramSourceRecord {
    pub ordinal: runmat_types::ProgramSourceId,
    pub name: Arc<str>,
    pub fullpath_name: Option<Arc<str>>,
    pub text: Arc<str>,
}
#[path = "source_context/portable.rs"]
mod portable;
pub use portable::{
    bind_portable_source_map_v2, bind_portable_source_map_v2_with_existing_lease,
    bind_portable_source_map_v2_with_records, PortableSourceBindingError,
};
#[cfg(test)]
#[path = "source_context/authority_tests.rs"]
mod authority_tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramSourceBindingError {
    DuplicateOrdinal(runmat_types::ProgramSourceId),
    Exhausted,
    ForeignOrRetired,
}

impl std::fmt::Display for ProgramSourceBindingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateOrdinal(ordinal) => {
                write!(formatter, "duplicate program source ordinal {}", ordinal.0)
            }
            Self::Exhausted => formatter.write_str("source identity space is exhausted"),
            Self::ForeignOrRetired => formatter.write_str(
                "existing source lease is retired or belongs to another runtime session",
            ),
        }
    }
}

impl std::error::Error for ProgramSourceBindingError {}

pub(crate) fn bind_program_sources(
    runtime: &crate::context::RuntimeContext,
    records: impl IntoIterator<Item = ProgramSourceRecord>,
) -> Result<ProgramSourceBinding, ProgramSourceBindingError> {
    let mut records = records.into_iter().collect::<Vec<_>>();
    records.sort_by_key(|record| record.ordinal);
    if let Some(duplicate) = records
        .windows(2)
        .find(|pair| pair[0].ordinal == pair[1].ordinal)
    {
        return Err(ProgramSourceBindingError::DuplicateOrdinal(
            duplicate[0].ordinal,
        ));
    }
    let mut sources = HashMap::with_capacity(records.len());
    let mut leases = Vec::with_capacity(records.len());
    for record in records {
        let lease = lease_source_context_in_context(
            runtime,
            record.name,
            record.fullpath_name,
            record.text,
        )
        .map_err(|_| ProgramSourceBindingError::Exhausted)?;
        sources.insert(record.ordinal, lease.source_ref());
        leases.push(lease);
    }
    Ok(ProgramSourceBinding {
        sources,
        _leases: leases,
        portable_map: None,
        frame_owner: runmat_async::CallFrameSourceOwner::issue(),
    })
}

pub(crate) fn bind_existing_program_source(
    runtime: &crate::context::RuntimeContext,
    ordinal: runmat_types::ProgramSourceId,
    lease: RuntimeSourceLease,
) -> Result<ProgramSourceBinding, ProgramSourceBindingError> {
    let reference = lease.source_ref();
    if reference.resolve_in(runtime).is_none() {
        return Err(ProgramSourceBindingError::ForeignOrRetired);
    }
    Ok(ProgramSourceBinding {
        sources: HashMap::from([(ordinal, reference)]),
        _leases: vec![lease],
        portable_map: None,
        frame_owner: runmat_async::CallFrameSourceOwner::issue(),
    })
}

impl RuntimeSourceLease {
    pub fn source_id(&self) -> SourceId {
        self.0
            .info
            .source_id
            .expect("issued source has an identity")
    }

    pub fn source_info(&self) -> SourceInfo {
        let mut info = self.0.info.clone();
        info.portable_identity = self.0.portable_identity.get().cloned();
        info
    }

    pub fn source_ref(&self) -> RuntimeSourceRef {
        RuntimeSourceRef {
            id: self.source_id(),
            issuer: self.0.authority.issuer,
            content: self.0.authority.content,
        }
    }
}

impl crate::context::SourceEvaluationProvenance for RuntimeSourceLease {
    fn source_id(&self) -> SourceId {
        RuntimeSourceLease::source_id(self)
    }

    fn runtime_source_ref(&self) -> Option<RuntimeSourceRef> {
        Some(self.source_ref())
    }

    fn runtime_source_lease(&self) -> Option<RuntimeSourceLease> {
        Some(self.clone())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceIdExhausted;

impl std::fmt::Display for SourceIdExhausted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("source identity space is exhausted")
    }
}

impl std::error::Error for SourceIdExhausted {}

impl crate::context::SourceState {
    fn issue_id(&mut self) -> Result<SourceId, SourceIdExhausted> {
        self.leased_catalog
            .retain(|_, entry| entry.owner.upgrade().is_some());
        let inherited = &self.inherited_authorities;
        let leased = &self.leased_catalog;
        self.issued_authorities
            .retain(|id, _| inherited.contains(id) || leased.contains_key(id));
        loop {
            let next = self.next_issued_id.ok_or(SourceIdExhausted)?;
            self.next_issued_id = next.checked_add(1);
            let id = SourceId(next);
            if !self.catalog.contains_key(&id)
                && !self.imported_identities.contains_key(&id)
                && !self.leased_catalog.contains_key(&id)
                && !self.issued_authorities.contains_key(&id)
            {
                return Ok(id);
            }
        }
    }

    /// A fork gets a provenance snapshot, not a copy of the parent's lease
    /// authority. Expired leases contribute neither records nor retained text.
    pub(crate) fn fork_snapshot(&self) -> Self {
        let mut snapshot = Self {
            current: self.current.clone(),
            catalog: self.catalog.clone(),
            imported_identities: self.imported_identities.clone(),
            ..Self::default()
        };
        for (id, entry) in &self.leased_catalog {
            if let Some(info) = entry.live_info() {
                if Rc::make_mut(&mut snapshot.catalog).remove(id).is_none() {
                    Rc::make_mut(&mut snapshot.catalog).insert(*id, info);
                }
            }
        }
        snapshot.issued_authorities = self
            .issued_authorities
            .iter()
            .filter(|(id, _)| snapshot.catalog.contains_key(id))
            .map(|(id, authority)| (*id, *authority))
            .collect();
        snapshot.inherited_authorities = snapshot.issued_authorities.keys().copied().collect();
        for (id, authority) in &snapshot.issued_authorities {
            snapshot.imported_identities.insert(*id, authority.content);
        }
        snapshot
    }
}

/// Issue source identity in the selected session. Raw imported numeric IDs do
/// not advance the issuer; an occupied numeric slot is skipped without reuse.
pub fn lease_source_context_in_context(
    runtime: &crate::context::RuntimeContext,
    name: impl Into<Arc<str>>,
    fullpath_name: Option<Arc<str>>,
    text: impl Into<Arc<str>>,
) -> Result<RuntimeSourceLease, SourceIdExhausted> {
    lease_source_context_with_lexical_file(runtime, name, fullpath_name, text, None)
}

fn lease_source_context_with_lexical_file(
    runtime: &crate::context::RuntimeContext,
    name: impl Into<Arc<str>>,
    fullpath_name: Option<Arc<str>>,
    text: impl Into<Arc<str>>,
    lexical_file: Option<SourceFileIdentity>,
) -> Result<RuntimeSourceLease, SourceIdExhausted> {
    let id = runtime.state().source.borrow_mut().issue_id()?;
    let info = SourceInfo {
        source_id: Some(id),
        name: name.into(),
        fullpath_name,
        text: text.into(),
        lexical_file,
        portable_identity: None,
    };
    let authority = RuntimeSourceAuthority {
        issuer: runtime.local_identity(),
        content: imported_identity(&info),
    };
    let inner = Rc::new(LeaseInner {
        info,
        authority,
        state: Rc::downgrade(runtime.state()),
        portable_identity: std::cell::OnceCell::new(),
    });
    {
        let mut source = runtime.state().source.borrow_mut();
        source.leased_catalog.insert(
            id,
            LeasedSource {
                owner: Rc::downgrade(&inner),
            },
        );
        source.issued_authorities.insert(id, authority);
    }
    Ok(RuntimeSourceLease(inner))
}

runmat_thread_local! {
    static CURRENT_SOURCE: RefCell<Option<SourceInfo>> = const { RefCell::new(None) };
    static SOURCE_CATALOG: RefCell<HashMap<SourceId, SourceInfo>> = RefCell::new(HashMap::new());
}

pub struct SourceContextGuard {
    prev: Option<SourceInfo>,
    invocation_current: Option<CurrentSourceOverlay>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for SourceContextGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        if let Some(overlay) = &self.invocation_current {
            overlay.retire();
        } else if let Some(state) = &self.state {
            state.source.borrow_mut().current = prev;
        } else {
            CURRENT_SOURCE.with(|slot| {
                *slot.borrow_mut() = prev;
            });
        }
    }
}

pub struct SourceCatalogGuard {
    prev: Option<Rc<HashMap<SourceId, SourceInfo>>>,
    invocation_catalog: Option<(
        crate::context::poll_projection::PollProjection<SourceCatalogProjection>,
        usize,
    )>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
    registered_ids: Vec<SourceId>,
}

impl Drop for SourceCatalogGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        if let Some((catalog, token)) = &self.invocation_catalog {
            catalog.borrow_mut().remove(*token);
        } else if let Some(state) = &self.state {
            state.source.borrow_mut().catalog =
                prev.expect("session catalog guard owns prior generation");
        } else {
            SOURCE_CATALOG.with(|catalog| {
                let prev = prev.expect("direct catalog guard owns prior generation");
                *catalog.borrow_mut() =
                    Rc::try_unwrap(prev).unwrap_or_else(|shared| shared.as_ref().clone());
            });
        }
        if let Some(state) = &self.state {
            let mut source = state.source.borrow_mut();
            for id in &self.registered_ids {
                let Some(count) = source.active_imports.get_mut(id) else {
                    continue;
                };
                *count -= 1;
                if *count == 0 {
                    source.active_imports.remove(id);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceCatalogAdmissionError {
    DuplicateSourceId(SourceId),
    LeasedSourceId(SourceId),
    PreviouslyIssuedSourceId(SourceId),
    ConflictingImport(SourceId),
    TooManyImports(SourceId),
    ProjectionExhausted,
}

impl std::fmt::Display for SourceCatalogAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateSourceId(id) => write!(formatter, "duplicate source identity {}", id.0),
            Self::LeasedSourceId(id) => write!(formatter, "source identity {} is leased", id.0),
            Self::PreviouslyIssuedSourceId(id) => {
                write!(formatter, "source identity {} was already issued", id.0)
            }
            Self::ConflictingImport(id) => {
                write!(
                    formatter,
                    "source identity {} has different imported text",
                    id.0
                )
            }
            Self::TooManyImports(id) => {
                write!(
                    formatter,
                    "source identity {} has too many active imports",
                    id.0
                )
            }
            Self::ProjectionExhausted => {
                formatter.write_str("source catalog projection is exhausted")
            }
        }
    }
}

impl std::error::Error for SourceCatalogAdmissionError {}

/// Replace the current source text for this thread.
///
/// This is used for UX features like "show the original expression" in legends and for
/// diagnostics that need to slice the source by byte-span.
pub fn replace_current_source(source: Option<&str>) -> SourceContextGuard {
    replace_current_source_context(None, source)
}

pub fn replace_current_source_context(
    name: Option<&str>,
    source: Option<&str>,
) -> SourceContextGuard {
    let next = source.map(|text| SourceInfo {
        source_id: None,
        name: Arc::<str>::from(name.unwrap_or_default()),
        fullpath_name: None,
        text: Arc::<str>::from(text),
        lexical_file: None,
        portable_identity: None,
    });
    if let Some(runtime) = crate::context::legacy::active() {
        if let Some(current) = runtime.source_current() {
            let current = current.clone();
            let overlay = CurrentSourceOverlay::push(current, next);
            return SourceContextGuard {
                prev: None,
                invocation_current: Some(overlay),
                state: None,
            };
        }
    }
    if let Some(state) = active_state() {
        let prev = std::mem::replace(&mut state.source.borrow_mut().current, next);
        SourceContextGuard {
            prev,
            invocation_current: None,
            state: Some(state),
        }
    } else {
        let prev = CURRENT_SOURCE.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), next));
        SourceContextGuard {
            prev,
            invocation_current: None,
            state: None,
        }
    }
}

pub fn replace_current_source_id(source_id: Option<SourceId>) -> SourceContextGuard {
    if let Some(runtime) = crate::context::legacy::active() {
        if runtime.source_current().is_some() {
            return replace_current_source_id_in_context(&runtime, source_id);
        }
    }
    if let Some(state) = active_state() {
        let next = source_id.and_then(source_info);
        let prev = std::mem::replace(&mut state.source.borrow_mut().current, next);
        SourceContextGuard {
            prev,
            invocation_current: None,
            state: Some(state),
        }
    } else {
        let next = source_id
            .and_then(|id| SOURCE_CATALOG.with(|catalog| catalog.borrow().get(&id).cloned()));
        let prev = CURRENT_SOURCE.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), next));
        SourceContextGuard {
            prev,
            invocation_current: None,
            state: None,
        }
    }
}

pub fn replace_current_source_id_in_context(
    runtime: &crate::context::RuntimeContext,
    source_id: Option<SourceId>,
) -> SourceContextGuard {
    let next = source_id.and_then(|source_id| source_info_in_context(runtime, source_id));
    let active = crate::context::legacy::active()
        .filter(|active| active.same_source_projection_owner(runtime));
    let current = active
        .as_ref()
        .and_then(crate::context::RuntimeContext::source_current)
        .or_else(|| runtime.source_current());
    if let Some(current) = current {
        let current = current.clone();
        let overlay = CurrentSourceOverlay::push(current, next);
        SourceContextGuard {
            prev: None,
            invocation_current: Some(overlay),
            state: None,
        }
    } else {
        let state = Rc::clone(runtime.state());
        let prev = std::mem::replace(&mut state.source.borrow_mut().current, next);
        SourceContextGuard {
            prev,
            invocation_current: None,
            state: Some(state),
        }
    }
}

/// Admit a context-free embedding catalog or a poll-local foreground view.
///
/// An imported numeric identity may never alias a session-issued lease or
/// later denote different source text. The returned guard is synchronous;
/// async callers must create it inside `RuntimeContext::scope` so each future
/// owns its view across suspension.
pub fn replace_source_catalog(
    entries: Vec<(SourceId, String, String)>,
) -> Result<SourceCatalogGuard, SourceCatalogAdmissionError> {
    replace_source_catalog_with_fullpaths(
        entries
            .into_iter()
            .map(|(source_id, name, text)| (source_id, name, None, text))
            .collect(),
    )
}

pub fn replace_source_catalog_with_fullpaths(
    entries: Vec<(SourceId, String, Option<String>, String)>,
) -> Result<SourceCatalogGuard, SourceCatalogAdmissionError> {
    let mut next = HashMap::with_capacity(entries.len());
    for (source_id, name, fullpath_name, text) in entries {
        if next
            .insert(
                source_id,
                SourceInfo {
                    source_id: Some(source_id),
                    name: Arc::<str>::from(name),
                    fullpath_name: fullpath_name.map(Arc::<str>::from),
                    text: Arc::<str>::from(text),
                    lexical_file: None,
                    portable_identity: None,
                },
            )
            .is_some()
        {
            return Err(SourceCatalogAdmissionError::DuplicateSourceId(source_id));
        }
    }
    let projection =
        crate::context::legacy::active().and_then(|runtime| runtime.source_catalog_view().cloned());
    if projection
        .as_ref()
        .is_some_and(|catalog| catalog.borrow().next_token.is_none())
    {
        return Err(SourceCatalogAdmissionError::ProjectionExhausted);
    }
    let state = active_state();
    if let Some(state) = &state {
        let source = state.source.borrow();
        for (id, info) in &next {
            if source
                .leased_catalog
                .get(id)
                .and_then(LeasedSource::live_info)
                .is_some()
            {
                return Err(SourceCatalogAdmissionError::LeasedSourceId(*id));
            }
            if source.next_issued_id.is_none_or(|next| id.0 < next)
                && !source.imported_identities.contains_key(id)
            {
                return Err(SourceCatalogAdmissionError::PreviouslyIssuedSourceId(*id));
            }
            if source
                .imported_identities
                .get(id)
                .is_some_and(|identity| *identity != imported_identity(info))
            {
                return Err(SourceCatalogAdmissionError::ConflictingImport(*id));
            }
            if source.active_imports.get(id) == Some(&usize::MAX) {
                return Err(SourceCatalogAdmissionError::TooManyImports(*id));
            }
        }
    }
    let registered_ids = next.keys().copied().collect::<Vec<_>>();
    if let Some(state) = &state {
        let mut source = state.source.borrow_mut();
        for id in &registered_ids {
            *source.active_imports.entry(*id).or_default() += 1;
            source
                .imported_identities
                .entry(*id)
                .or_insert_with(|| imported_identity(&next[id]));
        }
    }
    if let Some(catalog) = projection {
        let token = catalog.borrow_mut().push(next)?;
        return Ok(SourceCatalogGuard {
            prev: None,
            invocation_catalog: Some((catalog, token)),
            state,
            registered_ids,
        });
    }
    if let Some(state) = state {
        let prev = std::mem::replace(&mut state.source.borrow_mut().catalog, Rc::new(next));
        Ok(SourceCatalogGuard {
            prev: Some(prev),
            invocation_catalog: None,
            state: Some(state),
            registered_ids,
        })
    } else {
        let prev =
            SOURCE_CATALOG.with(|catalog| std::mem::replace(&mut *catalog.borrow_mut(), next));
        Ok(SourceCatalogGuard {
            prev: Some(Rc::new(prev)),
            invocation_catalog: None,
            state: None,
            registered_ids: Vec::new(),
        })
    }
}

pub fn source_catalog_entries() -> Vec<(SourceId, String, String)> {
    source_catalog_entries_with_fullpaths()
        .into_iter()
        .map(|(source_id, name, _fullpath_name, text)| (source_id, name, text))
        .collect()
}

pub fn source_catalog_entries_with_fullpaths() -> Vec<(SourceId, String, Option<String>, String)> {
    if let Some(runtime) = crate::context::legacy::active() {
        if let Some(catalog) = runtime.source_catalog_view() {
            return catalog_entries(catalog.borrow().current());
        }
    }
    if let Some(state) = active_state() {
        return catalog_entries(&state.source.borrow().catalog);
    }
    SOURCE_CATALOG.with(|catalog| catalog_entries(&catalog.borrow()))
}

pub fn current_source() -> Option<Arc<str>> {
    if let Some(runtime) = crate::context::legacy::active() {
        if let Some(current) = runtime.source_current() {
            return current.borrow().snapshot().map(|source| source.text);
        }
    }
    if let Some(state) = active_state() {
        return state
            .source
            .borrow()
            .current
            .as_ref()
            .map(|source| Arc::clone(&source.text));
    }
    CURRENT_SOURCE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|source| Arc::clone(&source.text))
    })
}

pub fn current_source_info() -> Option<SourceInfo> {
    if let Some(runtime) = crate::context::legacy::active() {
        if let Some(current) = runtime.source_current() {
            return current.borrow().snapshot();
        }
    }
    if let Some(state) = active_state() {
        return state.source.borrow().current.clone();
    }
    CURRENT_SOURCE.with(|slot| slot.borrow().clone())
}

pub fn source_info(source_id: SourceId) -> Option<SourceInfo> {
    if let Some(runtime) = crate::context::legacy::active() {
        if let Some(catalog) = runtime.source_catalog_view() {
            let source = runtime.state().source.borrow();
            return resolved_source_info(&source, catalog.borrow().current(), source_id);
        }
    }
    if let Some(state) = active_state() {
        let source = state.source.borrow();
        return resolved_source_info(&source, &source.catalog, source_id);
    }
    SOURCE_CATALOG.with(|catalog| catalog.borrow().get(&source_id).cloned())
}

pub fn source_info_in_context(
    runtime: &crate::context::RuntimeContext,
    source_id: SourceId,
) -> Option<SourceInfo> {
    let active = crate::context::legacy::active()
        .filter(|active| active.same_source_projection_owner(runtime));
    let catalog = active
        .as_ref()
        .and_then(crate::context::RuntimeContext::source_catalog_view)
        .or_else(|| runtime.source_catalog_view());
    let source = runtime.state().source.borrow();
    if let Some(catalog) = catalog {
        resolved_source_info(&source, catalog.borrow().current(), source_id)
    } else {
        resolved_source_info(&source, &source.catalog, source_id)
    }
}

/// Seed a newly-created standalone runtime context from the ambient source
/// state that preceded it.
///
/// Embedders normally install source state directly on an active context. The
/// standalone VM compatibility boundary creates that context after callers
/// have installed a legacy catalog, so it must transfer the catalog and
/// current source exactly once before entering the context scope.
pub fn inherit_legacy_source_context(context: &crate::context::RuntimeContext) {
    if active_state().is_some() {
        return;
    }
    let current = CURRENT_SOURCE.with(|slot| slot.borrow().clone());
    let catalog = SOURCE_CATALOG.with(|catalog| catalog.borrow().clone());
    let mut state = context.state().source.borrow_mut();
    state.current = current;
    state.imported_identities = catalog
        .iter()
        .map(|(id, info)| (*id, imported_identity(info)))
        .collect();
    state.catalog = Rc::new(catalog);
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

fn catalog_entries(
    catalog: &HashMap<SourceId, SourceInfo>,
) -> Vec<(SourceId, String, Option<String>, String)> {
    catalog
        .iter()
        .map(|(source_id, source)| {
            (
                *source_id,
                source.name.to_string(),
                source.fullpath_name.as_ref().map(ToString::to_string),
                source.text.to_string(),
            )
        })
        .collect()
}

#[cfg(test)]
mod lease_tests {
    use super::*;
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
    use std::future::Future;
    use std::task::{Context, Poll, Wake};

    struct NoopWake;

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    fn runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[test]
    fn poll_catalog_snapshot_retains_exact_map_generation() {
        let id = SourceId(11);
        let mut map = HashMap::new();
        map.insert(
            id,
            SourceInfo {
                source_id: Some(id),
                name: Arc::from("snapshot.m"),
                fullpath_name: None,
                text: Arc::from("one"),
                lexical_file: None,
                portable_identity: None,
            },
        );
        let mut view = SourceCatalogProjection::new(Rc::new(map));
        let first = view.snapshot_catalog();
        assert!(Rc::ptr_eq(&first, &view.snapshot_catalog()));
        let token = view.push(HashMap::new()).expect("overlay token");
        assert!(view.current().is_empty());
        view.remove(token);
        assert_eq!(
            first.get(&id).expect("retained source").text.as_ref(),
            "one"
        );
        assert!(Rc::ptr_eq(&first, &view.snapshot_catalog()));
    }

    #[test]
    fn imported_large_ids_cannot_poison_the_session_issuer() {
        let runtime = runtime();
        Rc::make_mut(&mut runtime.state().source.borrow_mut().catalog).insert(
            SourceId(usize::MAX - 1),
            SourceInfo {
                source_id: Some(SourceId(usize::MAX - 1)),
                name: Arc::from("imported"),
                fullpath_name: None,
                text: Arc::from("old"),
                lexical_file: None,
                portable_identity: None,
            },
        );
        let first = lease_source_context_in_context(&runtime, "first", None, "one").unwrap();
        let second = lease_source_context_in_context(&runtime, "second", None, "two").unwrap();
        assert_eq!(first.source_id(), SourceId(0));
        assert_eq!(second.source_id(), SourceId(1));
        assert_eq!(
            source_info_in_context(&runtime, SourceId(usize::MAX - 1))
                .unwrap()
                .text
                .as_ref(),
            "old"
        );
    }

    #[test]
    fn exhausted_issuer_never_wraps_or_reuses_retired_identity() {
        let runtime = runtime();
        runtime.state().source.borrow_mut().next_issued_id = Some(usize::MAX);
        let last = lease_source_context_in_context(&runtime, "last", None, "last text")
            .expect("final representable identity");
        assert_eq!(last.source_id(), SourceId(usize::MAX));
        drop(last);
        assert!(matches!(
            lease_source_context_in_context(&runtime, "extra", None, "no"),
            Err(SourceIdExhausted)
        ));
    }

    #[test]
    fn program_binding_maps_ordinals_to_issued_ids_and_retains_pending_text() {
        let runtime = runtime();
        let unrelated =
            lease_source_context_in_context(&runtime, "unrelated", None, "older").unwrap();
        let selected = runtime
            .clone()
            .with_program_sources([
                ProgramSourceRecord {
                    ordinal: runmat_types::ProgramSourceId(0),
                    name: Arc::from("main.m"),
                    fullpath_name: None,
                    text: Arc::from("primary"),
                },
                ProgramSourceRecord {
                    ordinal: runmat_types::ProgramSourceId(1),
                    name: Arc::from("helper.m"),
                    fullpath_name: None,
                    text: Arc::from("companion"),
                },
            ])
            .unwrap();
        let primary = selected
            .program_source_id(runmat_types::ProgramSourceId(0))
            .unwrap();
        let companion = selected
            .program_source_id(runmat_types::ProgramSourceId(1))
            .unwrap();
        assert_ne!(primary, unrelated.source_id());
        assert_ne!(companion, unrelated.source_id());
        drop(unrelated);
        assert_eq!(
            source_info_in_context(&runtime, companion)
                .unwrap()
                .text
                .as_ref(),
            "companion"
        );
        drop(selected);
        assert!(source_info_in_context(&runtime, primary).is_none());
        assert!(source_info_in_context(&runtime, companion).is_none());
    }

    #[test]
    fn duplicate_program_ordinals_fail_before_issuing_any_source() {
        let runtime = runtime();
        let records = ["first", "second"].map(|text| ProgramSourceRecord {
            ordinal: runmat_types::ProgramSourceId(0),
            name: Arc::from("main.m"),
            fullpath_name: None,
            text: Arc::from(text),
        });
        assert!(matches!(
            runtime.clone().with_program_sources(records),
            Err(ProgramSourceBindingError::DuplicateOrdinal(_))
        ));
        let first = lease_source_context_in_context(&runtime, "later", None, "later").unwrap();
        assert_eq!(first.source_id(), SourceId(0));
    }

    #[test]
    fn alternating_pending_invocations_keep_distinct_current_sources() {
        let session = runtime();
        let bind = |name: &'static str| {
            session
                .clone()
                .with_program_sources([ProgramSourceRecord {
                    ordinal: runmat_types::ProgramSourceId(0),
                    name: Arc::from(name),
                    fullpath_name: None,
                    text: Arc::from(name),
                }])
                .unwrap()
        };
        let first = bind("first");
        let second = bind("second");
        let pending = |runtime: RuntimeContext, expected: &'static str| {
            let source_id = runtime
                .program_source_id(runmat_types::ProgramSourceId(0))
                .unwrap();
            let scoped = runtime.clone();
            runtime.scope(async move {
                let _source = replace_current_source_id_in_context(&scoped, Some(source_id));
                std::future::poll_fn(|_| {
                    assert_eq!(current_source().as_deref(), Some(expected));
                    Poll::<()>::Pending
                })
                .await;
            })
        };
        let mut first_future = Box::pin(pending(first, "first"));
        let mut second_future = Box::pin(pending(second, "second"));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        drop(first_future);
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        drop(second_future);
        assert!(current_source().is_none());
    }

    #[test]
    fn sibling_futures_from_one_bound_program_keep_separate_source_views() {
        let selected = runtime()
            .with_program_sources([
                ProgramSourceRecord {
                    ordinal: runmat_types::ProgramSourceId(0),
                    name: Arc::from("primary.m"),
                    fullpath_name: None,
                    text: Arc::from("primary"),
                },
                ProgramSourceRecord {
                    ordinal: runmat_types::ProgramSourceId(1),
                    name: Arc::from("helper.m"),
                    fullpath_name: None,
                    text: Arc::from("helper"),
                },
            ])
            .unwrap();
        let pending = |ordinal: runmat_types::ProgramSourceId,
                       imported: SourceId,
                       expected_program: &'static str,
                       expected_import: &'static str| {
            let scoped = selected.clone();
            selected.scope(async move {
                let source_id = scoped.program_source_id(ordinal).unwrap();
                let _source = replace_current_source_id_in_context(&scoped, Some(source_id));
                let _catalog = replace_source_catalog(vec![(
                    imported,
                    expected_import.into(),
                    expected_import.into(),
                )])
                .expect("sibling catalog admission");
                std::future::poll_fn(|_| {
                    assert_eq!(current_source().as_deref(), Some(expected_program));
                    assert_eq!(
                        source_info(imported).unwrap().text.as_ref(),
                        expected_import
                    );
                    assert_eq!(
                        source_info_in_context(&scoped, imported)
                            .unwrap()
                            .text
                            .as_ref(),
                        expected_import
                    );
                    Poll::<()>::Pending
                })
                .await;
            })
        };
        let mut first = Box::pin(pending(
            runmat_types::ProgramSourceId(0),
            SourceId(31),
            "primary",
            "first",
        ));
        let mut second = Box::pin(pending(
            runmat_types::ProgramSourceId(1),
            SourceId(32),
            "helper",
            "second",
        ));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        assert!(first.as_mut().poll(&mut cx).is_pending());
        assert!(second.as_mut().poll(&mut cx).is_pending());
        assert!(first.as_mut().poll(&mut cx).is_pending());
        drop(first);
        assert!(second.as_mut().poll(&mut cx).is_pending());
        drop(second);
        assert!(selected
            .source_current()
            .unwrap()
            .borrow()
            .snapshot()
            .is_none());
    }

    #[test]
    fn current_source_guard_out_of_order_drop_preserves_newer_source() {
        let selected = runtime()
            .with_program_sources([
                ProgramSourceRecord {
                    ordinal: runmat_types::ProgramSourceId(0),
                    name: Arc::from("first.m"),
                    fullpath_name: None,
                    text: Arc::from("first"),
                },
                ProgramSourceRecord {
                    ordinal: runmat_types::ProgramSourceId(1),
                    name: Arc::from("second.m"),
                    fullpath_name: None,
                    text: Arc::from("second"),
                },
            ])
            .unwrap();
        let first_id = selected
            .program_source_id(runmat_types::ProgramSourceId(0))
            .unwrap();
        let second_id = selected
            .program_source_id(runmat_types::ProgramSourceId(1))
            .unwrap();
        let _scope = selected.enter();
        let first = replace_current_source_id_in_context(&selected, Some(first_id));
        let second = replace_current_source_id_in_context(&selected, Some(second_id));
        assert_eq!(current_source().as_deref(), Some("second"));
        drop(first);
        assert_eq!(current_source().as_deref(), Some("second"));
        drop(second);
        assert!(current_source().is_none());
    }

    #[test]
    fn pending_foreground_catalog_guards_are_poll_local() {
        let session = runtime();
        let pending = |id: SourceId, text: &'static str| {
            session.scope(async move {
                let _catalog = replace_source_catalog(vec![(id, text.into(), text.into())])
                    .expect("poll-local source catalog admission");
                std::future::poll_fn(|_| {
                    assert_eq!(source_info(id).unwrap().text.as_ref(), text);
                    Poll::<()>::Pending
                })
                .await;
            })
        };
        let mut first = Box::pin(pending(SourceId(11), "first"));
        let mut second = Box::pin(pending(SourceId(12), "second"));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        assert!(first.as_mut().poll(&mut cx).is_pending());
        assert!(second.as_mut().poll(&mut cx).is_pending());
        assert!(first.as_mut().poll(&mut cx).is_pending());
        drop(first);
        assert!(second.as_mut().poll(&mut cx).is_pending());
        drop(second);
        assert!(source_info_in_context(&session, SourceId(11)).is_none());
        assert!(source_info_in_context(&session, SourceId(12)).is_none());
    }

    #[test]
    fn out_of_order_foreground_guard_drop_preserves_newer_overlay() {
        let runtime = runtime();
        futures::executor::block_on(runtime.scope(async {
            let first = replace_source_catalog(vec![(SourceId(21), "first".into(), "one".into())])
                .expect("first catalog");
            let second =
                replace_source_catalog(vec![(SourceId(22), "second".into(), "two".into())])
                    .expect("second catalog");
            assert!(source_info(SourceId(21)).is_none());
            assert_eq!(source_info(SourceId(22)).unwrap().text.as_ref(), "two");
            drop(first);
            assert_eq!(source_info(SourceId(22)).unwrap().text.as_ref(), "two");
            drop(second);
            assert!(source_info(SourceId(21)).is_none());
            assert!(source_info(SourceId(22)).is_none());
        }));
    }

    #[test]
    fn interleaved_foreground_guards_do_not_remove_leased_sources() {
        let runtime = runtime();
        let _scope = runtime.enter();
        let first = lease_source_context_in_context(&runtime, "first", None, "one").unwrap();
        let guard = replace_source_catalog_with_fullpaths(vec![(
            SourceId(5),
            "foreground".into(),
            None,
            "statement".into(),
        )])
        .expect("foreground source catalog admission");
        let second = lease_source_context_in_context(&runtime, "second", None, "two").unwrap();
        drop(guard);
        assert_eq!(source_info(first.source_id()).unwrap().text.as_ref(), "one");
        assert_eq!(
            source_info(second.source_id()).unwrap().text.as_ref(),
            "two"
        );
        drop(first);
        assert!(source_info(SourceId(0)).is_none());
        assert_eq!(
            source_info(second.source_id()).unwrap().text.as_ref(),
            "two"
        );
    }

    #[test]
    fn colliding_raw_import_cannot_relabel_a_live_issued_source() {
        let runtime = runtime();
        let lease =
            lease_source_context_in_context(&runtime, "issued", None, "issued text").unwrap();
        let _scope = runtime.enter();
        let error = replace_source_catalog_with_fullpaths(vec![(
            lease.source_id(),
            "untrusted".into(),
            None,
            "untrusted text".into(),
        )])
        .err()
        .expect("live issued source must reject a raw import before publication");
        assert_eq!(
            error,
            SourceCatalogAdmissionError::LeasedSourceId(lease.source_id())
        );
        assert_eq!(
            source_info(lease.source_id()).unwrap().text.as_ref(),
            "issued text"
        );
        assert_eq!(lease.source_info().text.as_ref(), "issued text");
    }

    #[test]
    fn raw_import_rejects_retired_ids_and_conflicting_reuse_without_issuer_poisoning() {
        let runtime = runtime();
        let issued = lease_source_context_in_context(&runtime, "issued", None, "first").unwrap();
        let retired = issued.source_id();
        drop(issued);
        let _scope = runtime.enter();
        let error = replace_source_catalog(vec![(retired, "old".into(), "wrong".into())])
            .err()
            .expect("retired runtime identity must remain reserved");
        assert_eq!(
            error,
            SourceCatalogAdmissionError::PreviouslyIssuedSourceId(retired)
        );
        let future = SourceId(usize::MAX - 1);
        let first = replace_source_catalog(vec![(future, "imported".into(), "text".into())])
            .expect("independent raw identity");
        drop(first);
        let conflict = replace_source_catalog(vec![(future, "imported".into(), "changed".into())])
            .err()
            .expect("one raw identity cannot denote different text later");
        assert_eq!(
            conflict,
            SourceCatalogAdmissionError::ConflictingImport(future)
        );
        let same = replace_source_catalog(vec![(future, "imported".into(), "text".into())])
            .expect("identical raw provenance may be scoped again");
        drop(same);
        let next = lease_source_context_in_context(&runtime, "next", None, "second").unwrap();
        assert_eq!(next.source_id(), SourceId(1));
    }

    #[test]
    fn live_raw_import_reserves_exact_id_until_and_after_guard_drop() {
        let runtime = runtime();
        runtime.state().source.borrow_mut().next_issued_id = Some(5);
        let _scope = runtime.enter();
        let guard = replace_source_catalog(vec![(SourceId(5), "raw".into(), "text".into())])
            .expect("raw import admission");
        let first = lease_source_context_in_context(&runtime, "issued", None, "other").unwrap();
        assert_eq!(first.source_id(), SourceId(6));
        drop(guard);
        let second = lease_source_context_in_context(&runtime, "later", None, "later").unwrap();
        assert_eq!(second.source_id(), SourceId(7));
        assert!(source_info(SourceId(5)).is_none());
    }

    #[test]
    fn duplicate_raw_rows_reject_before_catalog_mutation() {
        let runtime = runtime();
        let _scope = runtime.enter();
        let error = replace_source_catalog(vec![
            (SourceId(5), "first".into(), "one".into()),
            (SourceId(5), "second".into(), "two".into()),
        ])
        .err()
        .expect("duplicate source identity");
        assert_eq!(
            error,
            SourceCatalogAdmissionError::DuplicateSourceId(SourceId(5))
        );
        assert!(source_info(SourceId(5)).is_none());
        assert!(runtime
            .state()
            .source
            .borrow()
            .imported_identities
            .is_empty());
    }

    #[test]
    fn lease_retirement_keeps_cloned_text_and_fork_snapshot_is_independent() {
        let runtime = runtime();
        let lease =
            lease_source_context_in_context(&runtime, "callback", None, "retained").unwrap();
        let id = lease.source_id();
        let retained = lease.source_info();
        let child = runtime
            .fork_parallel_lab(RuntimeServicePorts::default())
            .expect("idle session fork");
        drop(lease);
        assert!(source_info_in_context(&runtime, id).is_none());
        assert_eq!(retained.text.as_ref(), "retained");
        assert_eq!(
            source_info_in_context(&child, id).unwrap().text.as_ref(),
            "retained"
        );
        let child_lease = lease_source_context_in_context(&child, "child", None, "new").unwrap();
        assert_ne!(child_lease.source_id(), id);
    }

    #[test]
    fn parallel_lab_keeps_program_projection_after_parent_invocation_retires() {
        let session = runtime();
        let selected = session
            .clone()
            .with_program_sources([
                ProgramSourceRecord {
                    ordinal: runmat_types::ProgramSourceId(0),
                    name: Arc::from("main.m"),
                    fullpath_name: None,
                    text: Arc::from("main source"),
                },
                ProgramSourceRecord {
                    ordinal: runmat_types::ProgramSourceId(1),
                    name: Arc::from("helper.m"),
                    fullpath_name: None,
                    text: Arc::from("helper source"),
                },
            ])
            .unwrap();
        let primary = selected
            .program_source_id(runmat_types::ProgramSourceId(0))
            .unwrap();
        let companion = selected
            .program_source_id(runmat_types::ProgramSourceId(1))
            .unwrap();
        let current = replace_current_source_id_in_context(&selected, Some(companion));
        let child = selected
            .fork_parallel_lab(RuntimeServicePorts::default())
            .expect("idle session fork");
        drop(current);
        drop(selected);
        assert!(source_info_in_context(&session, primary).is_none());
        assert!(source_info_in_context(&session, companion).is_none());
        assert_eq!(
            child.program_source_id(runmat_types::ProgramSourceId(0)),
            Some(primary)
        );
        assert_eq!(
            child.program_source_id(runmat_types::ProgramSourceId(1)),
            Some(companion)
        );
        assert_eq!(
            source_info_in_context(&child, primary)
                .unwrap()
                .text
                .as_ref(),
            "main source"
        );
        assert_eq!(
            source_info_in_context(&child, companion)
                .unwrap()
                .text
                .as_ref(),
            "helper source"
        );
        assert_eq!(
            child
                .source_current()
                .unwrap()
                .borrow()
                .snapshot()
                .as_ref()
                .unwrap()
                .text
                .as_ref(),
            "helper source"
        );
        let later = lease_source_context_in_context(&child, "later", None, "other").unwrap();
        assert_ne!(later.source_id(), primary);
        assert_ne!(later.source_id(), companion);
    }

    #[test]
    fn last_lease_drop_under_catalog_borrow_is_nonpanicking_and_pruned() {
        let runtime = runtime();
        let lease = lease_source_context_in_context(&runtime, "held", None, "text").unwrap();
        let retired = lease.source_id();
        let borrow = runtime.state().source.borrow_mut();
        drop(lease);
        drop(borrow);
        assert!(source_info_in_context(&runtime, retired).is_none());
        let next = lease_source_context_in_context(&runtime, "next", None, "next").unwrap();
        assert_ne!(next.source_id(), retired);
        assert!(!runtime
            .state()
            .source
            .borrow()
            .leased_catalog
            .contains_key(&retired));
    }
}
