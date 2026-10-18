//! Canonical operation-owned backing events. Each issued scope has its own
//! ordered phase plan; tickets retain that identity through late retirement.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
#[path = "resources/storage.rs"]
mod storage;
use storage::{PhaseStorage, ScopeStorage};

/// A prepared sequence of maximum active-owner backing bytes. Domain owners
/// name the positions; the common ledger knows only issued order and limits.
pub struct ProviderResourcePhasePlan {
    limits: PhaseStorage<u64>,
}

impl ProviderResourcePhasePlan {
    pub fn new<const N: usize>(limits: [u64; N]) -> Result<Self, ProviderObservationFailure> {
        if N == 0 || limits.contains(&0) {
            return Err(ProviderObservationFailure::TrackingOverflow);
        }
        Ok(Self {
            limits: PhaseStorage::from_array(limits),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderResourcePhaseToken {
    scope_id: u64,
    index: usize,
}

impl ProviderResourcePhaseToken {
    pub fn scope_id(self) -> u64 {
        self.scope_id
    }

    pub fn index(self) -> usize {
        self.index
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderScopedResourceSnapshot {
    pub scope_id: u64,
    pub requested_live_bytes: u64,
    pub requested_peak_bytes: u64,
    pub issued: u64,
    pub retired: u64,
    pub pooled_handoffs: u64,
    pub phase_peaks: Vec<Option<u64>>,
    pub phase_limits: Vec<u64>,
    pub finished: bool,
}

/// Active probe-owner requested backing extents, not driver VRAM. Each scope
/// has independent peaks; aggregate peak includes simultaneous scopes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderResourceSnapshot {
    pub requested_live_bytes: u64,
    pub requested_peak_bytes: u64,
    pub issued: u64,
    pub retired: u64,
    pub pooled_handoffs: u64,
    pub scopes: Vec<ProviderScopedResourceSnapshot>,
}

struct ScopeState {
    id: u64,
    phase_limits: PhaseStorage<u64>,
    phase_peaks: PhaseStorage<Option<u64>>,
    next_phase: usize,
    active_phase: Option<usize>,
    finished: bool,
    issued: u64,
    retired: u64,
    live_bytes: u64,
    peak_bytes: u64,
    pooled_handoffs: u64,
}

impl ScopeState {
    fn begin_next_phase(
        &mut self,
    ) -> Result<ProviderResourcePhaseToken, ProviderObservationFailure> {
        if self.finished || self.next_phase >= self.phase_limits.len() {
            return Err(ProviderObservationFailure::TrackingOverflow);
        }
        let index = self.next_phase;
        if self.live_bytes > self.phase_limits[index] {
            return Err(ProviderObservationFailure::ResourceLimit);
        }
        self.phase_peaks[index] = Some(self.live_bytes);
        self.active_phase = Some(index);
        self.next_phase += 1;
        Ok(ProviderResourcePhaseToken {
            scope_id: self.id,
            index,
        })
    }

    fn finish(&mut self) -> Result<(), ProviderObservationFailure> {
        if self.finished || self.next_phase != self.phase_limits.len() {
            return Err(ProviderObservationFailure::TrackingOverflow);
        }
        self.finished = true;
        self.active_phase = None;
        Ok(())
    }

    fn snapshot(&self) -> ProviderScopedResourceSnapshot {
        ProviderScopedResourceSnapshot {
            scope_id: self.id,
            requested_live_bytes: self.live_bytes,
            requested_peak_bytes: self.peak_bytes,
            issued: self.issued,
            retired: self.retired,
            pooled_handoffs: self.pooled_handoffs,
            phase_peaks: self.phase_peaks.to_vec(),
            phase_limits: self.phase_limits.to_vec(),
            finished: self.finished,
        }
    }
}

#[derive(Default)]
pub(super) struct ResourceState {
    scopes: ScopeStorage<ScopeState>,
    unfinished_scopes: u64,
    issued: u64,
    retired: u64,
    live_bytes: u64,
    peak_bytes: u64,
    pooled_handoffs: u64,
}

impl ResourceState {
    pub(super) fn has_issued_resources(&self) -> bool {
        self.issued != 0
    }

    pub(super) fn live_count(&self) -> u64 {
        self.issued - self.retired
    }

    pub(super) fn scope_pending(&self) -> bool {
        self.unfinished_scopes != 0
    }

    fn scope_mut(&mut self, id: u64) -> Result<&mut ScopeState, ProviderObservationFailure> {
        let index = id
            .checked_sub(1)
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        self.scopes
            .get_mut(index)
            .filter(|scope| scope.id == id)
            .ok_or(ProviderObservationFailure::TrackingOverflow)
    }

    fn begin_scope(
        &mut self,
        plan: ProviderResourcePhasePlan,
    ) -> Result<u64, ProviderObservationFailure> {
        let id = u64::try_from(self.scopes.len())
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        let unfinished = self
            .unfinished_scopes
            .checked_add(1)
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        self.scopes.push(ScopeState {
            id,
            phase_peaks: PhaseStorage::filled(None, plan.limits.len()),
            phase_limits: plan.limits,
            next_phase: 0,
            active_phase: None,
            finished: false,
            issued: 0,
            retired: 0,
            live_bytes: 0,
            peak_bytes: 0,
            pooled_handoffs: 0,
        });
        self.unfinished_scopes = unfinished;
        Ok(id)
    }

    fn finish_scope(&mut self, id: u64) -> Result<(), ProviderObservationFailure> {
        if self.unfinished_scopes == 0 {
            return Err(ProviderObservationFailure::TrackingOverflow);
        }
        self.scope_mut(id)?.finish()?;
        self.unfinished_scopes -= 1;
        Ok(())
    }

    fn issue(&mut self, id: u64, bytes: u64) -> Result<(), ProviderObservationFailure> {
        let aggregate_issued = self
            .issued
            .checked_add(1)
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        let aggregate_live = self
            .live_bytes
            .checked_add(bytes)
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        let scope = self.scope_mut(id)?;
        let phase = scope
            .active_phase
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        let scoped_issued = scope
            .issued
            .checked_add(1)
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        let scoped_live = scope
            .live_bytes
            .checked_add(bytes)
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        if scoped_live > scope.phase_limits[phase] {
            return Err(ProviderObservationFailure::ResourceLimit);
        }
        scope.issued = scoped_issued;
        scope.live_bytes = scoped_live;
        scope.peak_bytes = scope.peak_bytes.max(scoped_live);
        scope.phase_peaks[phase] = Some(scope.phase_peaks[phase].unwrap_or(0).max(scoped_live));
        self.issued = aggregate_issued;
        self.live_bytes = aggregate_live;
        self.peak_bytes = self.peak_bytes.max(aggregate_live);
        Ok(())
    }

    fn retire(
        &mut self,
        id: u64,
        bytes: u64,
        pooled: bool,
    ) -> Result<(), ProviderObservationFailure> {
        let aggregate_retired = self
            .retired
            .checked_add(1)
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        let aggregate_live = self
            .live_bytes
            .checked_sub(bytes)
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        let aggregate_pooled = self
            .pooled_handoffs
            .checked_add(u64::from(pooled))
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        if aggregate_retired > self.issued {
            return Err(ProviderObservationFailure::TrackingOverflow);
        }
        let scope = self.scope_mut(id)?;
        let scoped_retired = scope
            .retired
            .checked_add(1)
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        let scoped_live = scope
            .live_bytes
            .checked_sub(bytes)
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        let scoped_pooled = scope
            .pooled_handoffs
            .checked_add(u64::from(pooled))
            .ok_or(ProviderObservationFailure::TrackingOverflow)?;
        if scoped_retired > scope.issued {
            return Err(ProviderObservationFailure::TrackingOverflow);
        }
        scope.retired = scoped_retired;
        scope.live_bytes = scoped_live;
        scope.pooled_handoffs = scoped_pooled;
        self.retired = aggregate_retired;
        self.live_bytes = aggregate_live;
        self.pooled_handoffs = aggregate_pooled;
        Ok(())
    }

    pub(super) fn snapshot(&self) -> Option<ProviderResourceSnapshot> {
        (!self.scopes.is_empty()).then(|| ProviderResourceSnapshot {
            requested_live_bytes: self.live_bytes,
            requested_peak_bytes: self.peak_bytes,
            issued: self.issued,
            retired: self.retired,
            pooled_handoffs: self.pooled_handoffs,
            scopes: self.scopes.iter().map(ScopeState::snapshot).collect(),
        })
    }
}

struct TicketInner {
    shared: Arc<Shared>,
    scope_id: u64,
    bytes: u64,
    pooled: AtomicBool,
}

impl Drop for TicketInner {
    fn drop(&mut self) {
        self.shared.with(|state| {
            if let Err(error) = state.resources.retire(
                self.scope_id,
                self.bytes,
                self.pooled.load(Ordering::Acquire),
            ) {
                state.fail(error);
            }
        });
    }
}

/// Clone only when two handles share the same physical owner. Final clone
/// retirement records owner disposition, not GPU completion or destruction.
#[derive(Clone)]
pub struct ProviderResourceTicket(Arc<TicketInner>);

impl ProviderResourceTicket {
    /// One physical ticket's control allocation; cloned handles share it.
    pub fn host_metadata_upper_bytes() -> Option<u64> {
        super::host_metadata::arc_host_upper_bytes::<TicketInner>()
    }

    pub fn requested_bytes(&self) -> u64 {
        self.0.bytes
    }

    /// A failed physical cleanup remains a canonical failure even if Rust
    /// subsequently drops the owner and balances its active-byte ticket.
    pub fn fail_owner(&self) {
        self.0
            .shared
            .with(|state| state.fail(ProviderObservationFailure::ProviderFailed));
    }

    /// The backing passed to provider residency; it remains allocated there.
    pub fn mark_pooled_handoff(&self) {
        self.0.pooled.store(true, Ordering::Release);
    }
}

struct ScopeInner {
    shared: Arc<Shared>,
    id: u64,
}

impl Drop for ScopeInner {
    fn drop(&mut self) {
        self.shared.with(|state| {
            if state
                .resources
                .scope_mut(self.id)
                .is_ok_and(|scope| !scope.finished)
            {
                state.fail(ProviderObservationFailure::Abandoned);
            }
        });
    }
}

/// Explicit opt-in operation sub-scope. The captured ID follows tickets and
/// nested future polls; ambient context never determines late retirement.
#[derive(Clone)]
pub struct ProviderResourceScope(Arc<ScopeInner>);

impl ProviderResourceScope {
    /// The first scope and its single phase have inline ledger storage.
    /// Only the scope's shared control allocation is newly allocated here.
    /// Additional scopes and multi-phase plans require independent quotes.
    pub fn first_single_phase_host_metadata_upper_bytes() -> Option<u64> {
        super::host_metadata::arc_host_upper_bytes::<ScopeInner>()
    }

    pub fn id(&self) -> u64 {
        self.0.id
    }

    pub fn fail(&self, reason: ProviderObservationFailure) {
        self.0.shared.with(|state| state.fail(reason));
    }

    /// Advances the immutable prepared sequence; the returned token carries
    /// this scope's identity and can be checked by the domain owner.
    pub fn begin_next_phase(
        &self,
    ) -> Result<ProviderResourcePhaseToken, ProviderObservationFailure> {
        self.0.shared.with(|state| {
            state
                .resources
                .scope_mut(self.0.id)
                .and_then(ScopeState::begin_next_phase)
                .inspect_err(|error| state.fail(*error))
        })
    }

    pub fn finish(&self) -> Result<(), ProviderObservationFailure> {
        self.0.shared.with(|state| {
            state
                .resources
                .finish_scope(self.0.id)
                .inspect_err(|error| state.fail(*error))
        })
    }

    /// Issue at an actual buffer allocation owner, using its backing extent.
    /// Borrowed inputs and published outputs are not probe-owned tickets.
    pub fn issue_resource_ticket(
        &self,
        requested_bytes: u64,
    ) -> Result<ProviderResourceTicket, ProviderObservationFailure> {
        self.0.shared.with(|state| {
            if let Some(failure) = state.failure {
                return Err(failure);
            }
            if state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::PublicationFailed);
                return Err(ProviderObservationFailure::PublicationFailed);
            }
            if let Err(error) = state.resources.issue(self.0.id, requested_bytes) {
                state.fail(error);
                return Err(error);
            }
            Ok(ProviderResourceTicket(Arc::new(TicketInner {
                shared: self.0.shared.clone(),
                scope_id: self.0.id,
                bytes: requested_bytes,
                pooled: AtomicBool::new(false),
            })))
        })
    }
}

impl ProviderOperationRecorder {
    /// Begin another issued sub-scope on this canonical operation. Sequential
    /// and nested probes retain one outer submission/timing authority while
    /// phase peaks and owner lifetimes remain isolated by scope identity.
    pub fn begin_resource_scope(
        &self,
        plan: ProviderResourcePhasePlan,
    ) -> Result<ProviderResourceScope, ProviderObservationFailure> {
        self.0.with(|state| {
            if let Some(failure) = state.failure {
                return Err(failure);
            }
            if state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::PublicationFailed);
                return Err(ProviderObservationFailure::PublicationFailed);
            }
            let id = state
                .resources
                .begin_scope(plan)
                .inspect_err(|error| state.fail(*error))?;
            Ok(ProviderResourceScope(Arc::new(ScopeInner {
                shared: self.0.clone(),
                id,
            })))
        })
    }
}

impl ProviderOperationReceipt {
    /// None means no resource scope was issued, never zero resource demand.
    pub fn resource_snapshot(&self) -> Option<ProviderResourceSnapshot> {
        self.0.with(|state| state.resources.snapshot())
    }
}

#[cfg(test)]
#[path = "resources/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "resources/retirement_tests.rs"]
mod retirement_tests;
