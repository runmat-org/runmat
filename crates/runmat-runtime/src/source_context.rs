use runmat_thread_local::runmat_thread_local;
use runmat_types::SourceId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub source_id: Option<SourceId>,
    pub name: Arc<str>,
    pub fullpath_name: Option<Arc<str>>,
    pub text: Arc<str>,
}

#[derive(Debug)]
pub(crate) struct LeasedSource {
    owner: Weak<LeaseInner>,
}

impl LeasedSource {
    fn live_info(&self) -> Option<SourceInfo> {
        self.owner.upgrade().map(|owner| owner.info.clone())
    }
}

fn resolved_source_info(source: &crate::context::SourceState, id: SourceId) -> Option<SourceInfo> {
    let leased = source
        .leased_catalog
        .get(&id)
        .and_then(LeasedSource::live_info);
    // An untyped foreground import can use the same integer, but it cannot
    // decide which source an issued lease denotes. Fail closed rather than
    // attributing either caller's span to the other's text.
    if leased.is_some() && source.catalog.contains_key(&id) {
        return None;
    }
    leased.or_else(|| source.catalog.get(&id).cloned())
}

#[derive(Debug)]
struct LeaseInner {
    info: SourceInfo,
    state: Weak<crate::context::RuntimeContextState>,
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
            .is_some_and(|entry| entry.owner.as_ptr() == self as *const LeaseInner)
        {
            sources.leased_catalog.remove(&id);
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
    sources: HashMap<runmat_types::ProgramSourceId, SourceId>,
    _leases: Vec<RuntimeSourceLease>,
}

impl ProgramSourceBinding {
    pub(crate) fn get(&self, ordinal: runmat_types::ProgramSourceId) -> Option<SourceId> {
        self.sources.get(&ordinal).copied()
    }
}

#[derive(Clone, Debug)]
pub struct ProgramSourceRecord {
    pub ordinal: runmat_types::ProgramSourceId,
    pub name: Arc<str>,
    pub fullpath_name: Option<Arc<str>>,
    pub text: Arc<str>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramSourceBindingError {
    DuplicateOrdinal(runmat_types::ProgramSourceId),
    Exhausted,
}

impl std::fmt::Display for ProgramSourceBindingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateOrdinal(ordinal) => {
                write!(formatter, "duplicate program source ordinal {}", ordinal.0)
            }
            Self::Exhausted => formatter.write_str("source identity space is exhausted"),
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
        sources.insert(record.ordinal, lease.source_id());
        leases.push(lease);
    }
    Ok(ProgramSourceBinding {
        sources,
        _leases: leases,
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
        self.0.info.clone()
    }
}

impl crate::context::SourceEvaluationProvenance for RuntimeSourceLease {
    fn source_id(&self) -> SourceId {
        RuntimeSourceLease::source_id(self)
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
        loop {
            let next = self.next_issued_id.ok_or(SourceIdExhausted)?;
            self.next_issued_id = next.checked_add(1);
            let id = SourceId(next);
            if !self.catalog.contains_key(&id) && !self.leased_catalog.contains_key(&id) {
                return Ok(id);
            }
        }
    }

    /// A fork gets a provenance snapshot, not a copy of the parent's lease
    /// authority. Expired leases contribute neither records nor retained text.
    pub(crate) fn fork_snapshot(&self) -> Self {
        let mut snapshot = Self::default();
        snapshot.current = self.current.clone();
        snapshot.catalog = self.catalog.clone();
        for (id, entry) in &self.leased_catalog {
            if let Some(info) = entry.live_info() {
                if snapshot.catalog.remove(id).is_none() {
                    snapshot.catalog.insert(*id, info);
                }
            }
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
    let id = runtime.state().source.borrow_mut().issue_id()?;
    let inner = Rc::new(LeaseInner {
        info: SourceInfo {
            source_id: Some(id),
            name: name.into(),
            fullpath_name,
            text: text.into(),
        },
        state: Rc::downgrade(runtime.state()),
    });
    runtime.state().source.borrow_mut().leased_catalog.insert(
        id,
        LeasedSource {
            owner: Rc::downgrade(&inner),
        },
    );
    Ok(RuntimeSourceLease(inner))
}

runmat_thread_local! {
    static CURRENT_SOURCE: RefCell<Option<SourceInfo>> = const { RefCell::new(None) };
    static SOURCE_CATALOG: RefCell<HashMap<SourceId, SourceInfo>> = RefCell::new(HashMap::new());
}

pub struct SourceContextGuard {
    prev: Option<SourceInfo>,
    invocation_current: Option<Rc<RefCell<Option<SourceInfo>>>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for SourceContextGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        if let Some(current) = &self.invocation_current {
            *current.borrow_mut() = prev;
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
    prev: HashMap<SourceId, SourceInfo>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for SourceCatalogGuard {
    fn drop(&mut self) {
        let prev = std::mem::take(&mut self.prev);
        if let Some(state) = &self.state {
            state.source.borrow_mut().catalog = prev;
        } else {
            SOURCE_CATALOG.with(|catalog| {
                *catalog.borrow_mut() = prev;
            });
        }
    }
}

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
    });
    if let Some(runtime) = crate::context::legacy::active() {
        if let Some(current) = runtime.source_current() {
            let current = Rc::clone(current);
            let prev = std::mem::replace(&mut *current.borrow_mut(), next);
            return SourceContextGuard {
                prev,
                invocation_current: Some(current),
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
        let next = source_id.and_then(|id| {
            let source = state.source.borrow();
            resolved_source_info(&source, id)
        });
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
    if let Some(current) = runtime.source_current() {
        let current = Rc::clone(current);
        let prev = std::mem::replace(&mut *current.borrow_mut(), next);
        SourceContextGuard {
            prev,
            invocation_current: Some(current),
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

pub fn replace_source_catalog(entries: Vec<(SourceId, String, String)>) -> SourceCatalogGuard {
    replace_source_catalog_with_fullpaths(
        entries
            .into_iter()
            .map(|(source_id, name, text)| (source_id, name, None, text))
            .collect(),
    )
}

pub fn replace_source_catalog_with_fullpaths(
    entries: Vec<(SourceId, String, Option<String>, String)>,
) -> SourceCatalogGuard {
    let next = entries
        .into_iter()
        .map(|(source_id, name, fullpath_name, text)| {
            (
                source_id,
                SourceInfo {
                    source_id: Some(source_id),
                    name: Arc::<str>::from(name),
                    fullpath_name: fullpath_name.map(Arc::<str>::from),
                    text: Arc::<str>::from(text),
                },
            )
        })
        .collect::<HashMap<_, _>>();
    if let Some(state) = active_state() {
        let prev = std::mem::replace(&mut state.source.borrow_mut().catalog, next);
        SourceCatalogGuard {
            prev,
            state: Some(state),
        }
    } else {
        let prev =
            SOURCE_CATALOG.with(|catalog| std::mem::replace(&mut *catalog.borrow_mut(), next));
        SourceCatalogGuard { prev, state: None }
    }
}

pub fn source_catalog_entries() -> Vec<(SourceId, String, String)> {
    source_catalog_entries_with_fullpaths()
        .into_iter()
        .map(|(source_id, name, _fullpath_name, text)| (source_id, name, text))
        .collect()
}

pub fn source_catalog_entries_with_fullpaths() -> Vec<(SourceId, String, Option<String>, String)> {
    if let Some(state) = active_state() {
        return catalog_entries(&state.source.borrow().catalog);
    }
    SOURCE_CATALOG.with(|catalog| catalog_entries(&catalog.borrow()))
}

pub fn current_source() -> Option<Arc<str>> {
    if let Some(runtime) = crate::context::legacy::active() {
        if let Some(current) = runtime.source_current() {
            return current
                .borrow()
                .as_ref()
                .map(|source| Arc::clone(&source.text));
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
            return current.borrow().clone();
        }
    }
    if let Some(state) = active_state() {
        return state.source.borrow().current.clone();
    }
    CURRENT_SOURCE.with(|slot| slot.borrow().clone())
}

pub fn source_info(source_id: SourceId) -> Option<SourceInfo> {
    if let Some(state) = active_state() {
        let source = state.source.borrow();
        return resolved_source_info(&source, source_id);
    }
    SOURCE_CATALOG.with(|catalog| catalog.borrow().get(&source_id).cloned())
}

pub fn source_info_in_context(
    runtime: &crate::context::RuntimeContext,
    source_id: SourceId,
) -> Option<SourceInfo> {
    let source = runtime.state().source.borrow();
    resolved_source_info(&source, source_id)
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
    state.catalog = catalog;
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
    fn imported_large_ids_cannot_poison_the_session_issuer() {
        let runtime = runtime();
        runtime.state().source.borrow_mut().catalog.insert(
            SourceId(usize::MAX - 1),
            SourceInfo {
                source_id: Some(SourceId(usize::MAX - 1)),
                name: Arc::from("imported"),
                fullpath_name: None,
                text: Arc::from("old"),
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
    fn interleaved_foreground_guards_do_not_remove_leased_sources() {
        let runtime = runtime();
        let _scope = runtime.enter();
        let first = lease_source_context_in_context(&runtime, "first", None, "one").unwrap();
        let guard = replace_source_catalog_with_fullpaths(vec![(
            SourceId(5),
            "foreground".into(),
            None,
            "statement".into(),
        )]);
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
        let guard = replace_source_catalog_with_fullpaths(vec![(
            lease.source_id(),
            "untrusted".into(),
            None,
            "untrusted text".into(),
        )]);
        assert!(source_info(lease.source_id()).is_none());
        assert_eq!(lease.source_info().text.as_ref(), "issued text");
        drop(guard);
        assert_eq!(
            source_info(lease.source_id()).unwrap().text.as_ref(),
            "issued text"
        );
    }

    #[test]
    fn lease_retirement_keeps_cloned_text_and_fork_snapshot_is_independent() {
        let runtime = runtime();
        let lease =
            lease_source_context_in_context(&runtime, "callback", None, "retained").unwrap();
        let id = lease.source_id();
        let retained = lease.source_info();
        let child = runtime.fork_parallel_lab(RuntimeServicePorts::default());
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
