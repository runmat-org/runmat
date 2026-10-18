//! Retained owner and two-phase protocol for canonical workspace publication.

use std::alloc::Layout;
use std::fmt;
use std::rc::Rc;

use allocator_api2::boxed::Box as FallibleBox;
use runmat_execution::resource::ExactShared;
use runmat_value::{ExactHostAllocationClaim, ExactHostAllocationPermit};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::RuntimeError;

use super::{WorkspacePublicationBatch, WorkspacePublicationMutation};

/// Domain-owned two-phase participant in canonical workspace publication.
///
/// `prepare` may allocate and validate, but it must not mutate the observer's
/// visible state. It stores one private prepared transaction in the protocol
/// object. `checkpoint` closes any revocable authority immediately before the
/// workspace commit. `commit` applies the prepared transaction infallibly;
/// `abort` retires it without publication.
///
/// Runtime intentionally does not type-erase a newly allocated transaction on
/// each publication. Implementations own and quote the concrete storage used
/// for their prepared transaction.
pub trait WorkspacePublicationProtocol: fmt::Debug {
    /// Prepare one prospective source effect before the projection changes.
    /// A participating observer owns and checkpoints its short allocation
    /// transaction here, but does not install its pending metadata yet.
    /// `None` retains the final-publication protocol for external observers.
    fn prepare_mutation(
        &self,
        _context: Option<&RuntimeContext>,
        _mutation: WorkspacePublicationMutation<'_>,
    ) -> Result<Option<WorkspacePublicationToken>, RuntimeError> {
        Ok(None)
    }

    /// Prepare an entire authentic completion as one transaction, rather
    /// than invoking the observer separately for every completed binding.
    fn prepare_completion(
        &self,
        _context: Option<&RuntimeContext>,
        _changes: WorkspacePublicationBatch<'_>,
    ) -> Result<Option<WorkspacePublicationToken>, RuntimeError> {
        Ok(None)
    }

    /// Prepare a completion that is also the terminal canonical publication.
    /// The participant must stage and checkpoint its replacement without
    /// exposing it until Runtime has installed the matching Base revision.
    /// Legacy final-publication observers retain their ordinary `prepare`
    /// behavior; Core overrides this with one atomic prepared transaction.
    fn prepare_completion_publication(
        &self,
        context: Option<&RuntimeContext>,
        changes: WorkspacePublicationBatch<'_>,
    ) -> Result<WorkspacePublicationToken, RuntimeError> {
        self.prepare(context, changes)
    }

    /// Install a previously prepared mutation/completion after Runtime has
    /// installed its projection. This must not allocate, fail, or call back
    /// into execution. Every returned prospective token requires this hook.
    fn commit_mutation(&self, _token: WorkspacePublicationToken) {}

    /// Retire only this prospective update, preserving any earlier prefix.
    fn abort_mutation(&self, _token: WorkspacePublicationToken) {}

    fn prepare(
        &self,
        context: Option<&RuntimeContext>,
        changes: WorkspacePublicationBatch<'_>,
    ) -> Result<WorkspacePublicationToken, RuntimeError>;

    fn checkpoint(
        &self,
        token: WorkspacePublicationToken,
        context: Option<&RuntimeContext>,
    ) -> Result<(), RuntimeError>;

    fn commit(&self, token: WorkspacePublicationToken);

    fn abort(&self, token: WorkspacePublicationToken);
}

/// Identity of one domain-owned prepared publication transaction.
///
/// A protocol chooses a monotonically unique generation after preparation is
/// fully installed. Runtime returns that exact generation on checkpoint,
/// commit, or abort, so a failed nested prepare cannot retire an outer
/// transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkspacePublicationToken(u64);

impl WorkspacePublicationToken {
    pub const fn from_generation(generation: u64) -> Self {
        Self(generation)
    }

    pub const fn generation(self) -> u64 {
        self.0
    }
}

struct ObserverOwner {
    // Field order is physical lifetime order: protocol allocation, its claim,
    // then the claim funding this payload's ExactShared header.
    protocol: Box<dyn WorkspacePublicationProtocol>,
    _protocol_claim: ExactHostAllocationClaim,
    _header_claim: ExactHostAllocationClaim,
}

impl fmt::Debug for ObserverOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ObserverOwner")
    }
}

#[derive(Clone)]
enum ObserverBacking {
    Legacy(Rc<dyn WorkspacePublicationProtocol>),
    Managed(ExactShared<ObserverOwner>),
}

impl ObserverBacking {
    fn protocol(&self) -> &dyn WorkspacePublicationProtocol {
        match self {
            Self::Legacy(protocol) => protocol.as_ref(),
            Self::Managed(owner) => owner.get().protocol.as_ref(),
        }
    }
}

#[derive(Clone)]
pub struct WorkspacePublicationObserver(ObserverBacking);

impl fmt::Debug for WorkspacePublicationObserver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("WorkspacePublicationObserver")
    }
}

impl WorkspacePublicationObserver {
    pub(in crate::workspace::base::projection_service) fn prepare_mutation(
        &self,
        context: Option<&RuntimeContext>,
        mutation: WorkspacePublicationMutation<'_>,
    ) -> Result<PreparedWorkspaceMutation<'_>, RuntimeError> {
        let token = self.0.protocol().prepare_mutation(context, mutation)?;
        Ok(PreparedWorkspaceMutation {
            observer: self,
            token,
            installed: false,
        })
    }

    pub(in crate::workspace::base::projection_service) fn prepare_completion(
        &self,
        context: Option<&RuntimeContext>,
        changes: WorkspacePublicationBatch<'_>,
    ) -> Result<PreparedWorkspaceMutation<'_>, RuntimeError> {
        let token = self.0.protocol().prepare_completion(context, changes)?;
        Ok(PreparedWorkspaceMutation {
            observer: self,
            token,
            installed: false,
        })
    }

    /// Construct an explicitly unbounded observer for legacy integrations.
    pub fn new(observer: impl WorkspacePublicationProtocol + 'static) -> Self {
        Self(ObserverBacking::Legacy(Rc::new(observer)))
    }

    /// Construct an observer under the current invocation's host-allocation
    /// policy. An explicitly unbounded policy retains the legacy Rc owner;
    /// finite policies fund the concrete protocol box and exact shared owner.
    pub fn new_with_context<P>(context: &RuntimeContext, protocol: P) -> Result<Self, RuntimeError>
    where
        P: WorkspacePublicationProtocol + 'static,
    {
        let protocol_bytes = Layout::new::<P>().size();
        let owner_bytes = ExactShared::<ObserverOwner>::requested_bytes();
        let raw_bytes = protocol_bytes
            .checked_add(owner_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| admission_error(context, HostAllocationError::Overflow))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(raw_bytes, 2)
            .map_err(|cause| admission_error(context, cause))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| admission_error(context, cause))?;
        let Some(receipt) = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|cause| admission_error(context, cause))?
        else {
            return Ok(Self::new(protocol));
        };
        let permit = receipt
            .output_permit(context)
            .map_err(|cause| admission_error(context, cause))?;
        let protocol_claim = permit
            .reserve_exact_claim(protocol_bytes)
            .map_err(HostAllocationError::from)
            .map_err(|cause| admission_error(context, cause))?;
        let allocated = FallibleBox::try_new(protocol).map_err(|_| {
            admission_error(
                context,
                HostAllocationError::from(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        })?;
        let pointer = FallibleBox::into_raw(allocated);
        // SAFETY: allocator-api2's Global allocator uses the same global
        // allocation/deallocation contract as std::Box. The fresh allocation
        // has exactly Layout::new::<P>(), and ownership transfers once here.
        let protocol: Box<P> = unsafe { Box::from_raw(pointer) };
        let protocol: Box<dyn WorkspacePublicationProtocol> = protocol;
        let header_claim = permit
            .reserve_exact_claim(owner_bytes)
            .map_err(HostAllocationError::from)
            .map_err(|cause| admission_error(context, cause))?;
        let payload = ObserverOwner {
            protocol,
            _protocol_claim: protocol_claim,
            _header_claim: header_claim,
        };
        let owner = ExactShared::try_new(payload).map_err(|_| {
            admission_error(
                context,
                HostAllocationError::from(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        })?;
        receipt
            .checkpoint(context)
            .map_err(|cause| admission_error(context, cause))?;
        Ok(Self(ObserverBacking::Managed(owner)))
    }

    pub(in crate::workspace::base::projection_service) fn prepare(
        &self,
        context: Option<&RuntimeContext>,
        changes: WorkspacePublicationBatch<'_>,
    ) -> Result<PreparedWorkspacePublication, RuntimeError> {
        let token = self.0.protocol().prepare(context, changes)?;
        Ok(PreparedWorkspacePublication {
            observer: self.clone(),
            token,
            checkpointed: false,
            committed: false,
        })
    }

    pub(in crate::workspace::base::projection_service) fn prepare_completion_publication(
        &self,
        context: Option<&RuntimeContext>,
        changes: WorkspacePublicationBatch<'_>,
    ) -> Result<PreparedWorkspacePublication, RuntimeError> {
        let token = self
            .0
            .protocol()
            .prepare_completion_publication(context, changes)?;
        Ok(PreparedWorkspacePublication {
            observer: self.clone(),
            token,
            checkpointed: false,
            committed: false,
        })
    }
}

/// Synchronous rollback guard: it never owns a RuntimeContext or a producer
/// receipt, and must not cross an execution suspension point.
pub(in crate::workspace::base::projection_service) struct PreparedWorkspaceMutation<'a> {
    observer: &'a WorkspacePublicationObserver,
    token: Option<WorkspacePublicationToken>,
    installed: bool,
}

impl PreparedWorkspaceMutation<'_> {
    pub(in crate::workspace::base::projection_service) fn commit(mut self) {
        self.installed = true;
        if let Some(token) = self.token {
            self.observer.0.protocol().commit_mutation(token);
        }
    }
}

impl Drop for PreparedWorkspaceMutation<'_> {
    fn drop(&mut self) {
        if !self.installed {
            if let Some(token) = self.token {
                self.observer.0.protocol().abort_mutation(token);
            }
        }
    }
}

pub(in crate::workspace::base::projection_service) struct PreparedWorkspacePublication {
    observer: WorkspacePublicationObserver,
    token: WorkspacePublicationToken,
    checkpointed: bool,
    committed: bool,
}

impl PreparedWorkspacePublication {
    pub(in crate::workspace::base::projection_service) fn checkpoint(
        &mut self,
        context: Option<&RuntimeContext>,
    ) -> Result<(), RuntimeError> {
        self.observer.0.protocol().checkpoint(self.token, context)?;
        self.checkpointed = true;
        Ok(())
    }

    pub(in crate::workspace::base::projection_service) fn commit(mut self) {
        debug_assert!(self.checkpointed);
        self.committed = true;
        self.observer.0.protocol().commit(self.token);
    }
}

impl Drop for PreparedWorkspacePublication {
    fn drop(&mut self) {
        if !self.committed {
            self.observer.0.protocol().abort(self.token);
        }
    }
}

fn admission_error(context: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    super::super::workspace_admission_error(context, cause)
}
