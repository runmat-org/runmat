//! Nonpersistable completion authority for one VM-installed fusion region.
//! `install` is a trusted Rust-host integration API, not a cryptographic or
//! sealed capability against hostile embedding code. The normal VM caller
//! checks its retained, store-reconciled plan before issuing a right.
//! A region key is session-local registration, not serialized code proof.
#[path = "vm_fusion_completion/authority.rs"]
mod authority;
pub use authority::{
    RuntimeVmFusionGenerationPermit, VmFusionFeedbackLease, VmFusionSemanticIdentity,
    VmFusionSessionRegistration,
};

use super::vm_fusion_host_route::VmFusionHostOrigin;
use super::{RuntimeContext, RuntimeContextLocalId, VmFusionHostRoute};
use runmat_builtins::BuiltinBindingIdentity;
use runmat_execution::{
    cancellation::CancellationToken, PlacementExecutionAuthorityRevision, PlacementRevision,
    ProgramRevision,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

static NEXT_INSTALLATION: AtomicU64 = AtomicU64::new(1);

/// Tagged exact VM group in one installed interpreter plan. The numeric ID is
/// process-local and has no public constructor or deserialization path.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VmFusionRegionRegistration {
    installation: u64,
    semantic: VmFusionSemanticIdentity,
    group_id: usize,
    span_start: usize,
    span_end: usize,
}

/// Retained by the interpreter beside its reconciled `Rc<FusionPlan>`. The
/// issuing VM must check the actual group against that retained plan before
/// asking for a completion right. Runtime cannot inspect VM bytecode.
pub struct VmFusionInstalledRegion {
    registration: VmFusionRegionRegistration,
    owner: RuntimeContextLocalId,
    program: Option<ProgramRevision>,
    cancellation: CancellationToken,
    issuer_live: Arc<()>,
    feedback_live: Arc<()>,
}

/// A single selected VM attempt. This right is deliberately not Clone.
pub struct VmFusionCompletionRight {
    registration: VmFusionRegionRegistration,
    owner: RuntimeContextLocalId,
    program: Option<ProgramRevision>,
    cancellation: CancellationToken,
    issuer_live: Weak<()>,
    feedback_live: Weak<()>,
    session: Option<VmFusionSessionRegistration>,
}

impl VmFusionInstalledRegion {
    /// Trusted Rust-host issuance point. The VM path registers only its
    /// already-reconciled plan; external hosts using this public API are
    /// responsible for the same installation provenance.
    pub fn install(
        runtime: &RuntimeContext,
        semantic: VmFusionSemanticIdentity,
        group_id: usize,
        span_start: usize,
        span_end: usize,
    ) -> Option<Self> {
        if span_start > span_end || runtime.is_cancelled() {
            return None;
        }
        let installation = NEXT_INSTALLATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("VM fusion installation identity space exhausted");
        Some(Self {
            registration: VmFusionRegionRegistration {
                installation,
                semantic,
                group_id,
                span_start,
                span_end,
            },
            owner: runtime.local_identity(),
            program: runtime.program_revision().cloned(),
            cancellation: runtime.cancellation(),
            issuer_live: Arc::new(()),
            feedback_live: Arc::new(()),
        })
    }

    pub fn matches_group(&self, group_id: usize, span_start: usize, span_end: usize) -> bool {
        self.registration.group_id == group_id
            && self.registration.span_start == span_start
            && self.registration.span_end == span_end
    }

    pub fn right(&self, runtime: &RuntimeContext) -> Option<VmFusionCompletionRight> {
        if runtime.local_identity() != self.owner
            || runtime.program_revision() != self.program.as_ref()
            || !runtime.cancellation().same_generation(&self.cancellation)
            || runtime.is_cancelled()
        {
            return None;
        }
        let session = runtime.service_ports().placement().and_then(|service| {
            service.whole_work_vm_fusion_region_session_revision(
                self.program.clone(),
                self.registration,
                Arc::downgrade(&self.feedback_live),
            )
        });
        let session = session.filter(|session| {
            matches!(
                session.revision().authority,
                PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
            ) && session.is_current()
        });
        Some(VmFusionCompletionRight {
            registration: self.registration,
            owner: self.owner,
            program: self.program.clone(),
            cancellation: self.cancellation.clone(),
            issuer_live: Arc::downgrade(&self.issuer_live),
            feedback_live: Arc::downgrade(&self.feedback_live),
            session,
        })
    }
}

impl VmFusionCompletionRight {
    /// A borrowed instruction route never issues a second completion right.
    /// The one-shot right must still be consumed after VM publication.
    pub fn host_route(
        &self,
        runtime: &RuntimeContext,
        expected_bindings: Vec<BuiltinBindingIdentity>,
    ) -> Option<VmFusionHostRoute> {
        let session = self.revalidated_session(runtime)?;
        let placement = runtime.service_ports().placement()?;
        VmFusionHostRoute::new(
            VmFusionHostOrigin {
                registration: self.registration,
                owner: self.owner,
                program: self.program.clone(),
                cancellation: self.cancellation.clone(),
                placement_owner: std::rc::Rc::downgrade(placement),
                issuer_live: self.issuer_live.clone(),
                generation: session.generation_permit(),
            },
            expected_bindings,
        )
    }
    /// Selection may borrow the exact session revision, never manufacture one.
    pub fn placement_revision(&self) -> Option<&PlacementRevision> {
        self.issuer_live.upgrade()?;
        let session = self.session.as_ref()?;
        session.is_current().then(|| session.revision())
    }

    /// Read-only planning check; no retained feedback owner or second right
    /// is minted. The consuming completion check is repeated at publication.
    pub fn planning_revision(&self, runtime: &RuntimeContext) -> Option<PlacementRevision> {
        self.revalidated_session(runtime)
            .map(|session| session.revision().clone())
    }

    pub fn completion_lease(self, runtime: &RuntimeContext) -> Option<VmFusionFeedbackLease> {
        let session = self.revalidated_session(runtime)?;
        // Pending feedback may retain its own lease after VM teardown, but it
        // must not keep an unconsumed sibling right capable of new issuance.
        self.issuer_live.upgrade()?;
        let live = self.feedback_live.upgrade()?;
        Some(session.feedback_lease(live))
    }

    fn revalidated_session(
        &self,
        runtime: &RuntimeContext,
    ) -> Option<&VmFusionSessionRegistration> {
        self.issuer_live.upgrade()?;
        if runtime.local_identity() != self.owner
            || runtime.program_revision() != self.program.as_ref()
            || !runtime.cancellation().same_generation(&self.cancellation)
            || runtime.is_cancelled()
        {
            return None;
        }
        let selected = self.session.as_ref()?;
        let current = runtime.service_ports().placement().and_then(|service| {
            service.whole_work_vm_fusion_region_session_revision(
                self.program.clone(),
                self.registration,
                self.feedback_live.clone(),
            )
        })?;
        if !current.is_current()
            || current.revision() != selected.revision()
            || !current.same_generation(selected)
        {
            return None;
        }
        Some(selected)
    }

    pub fn registration(&self) -> VmFusionRegionRegistration {
        self.registration
    }
}

impl VmFusionRegionRegistration {
    pub(crate) fn span_start(self) -> usize {
        self.span_start
    }

    pub(crate) fn span_end(self) -> usize {
        self.span_end
    }

    pub fn contains_pc(self, pc: usize) -> bool {
        self.span_start <= pc && pc <= self.span_end
    }
    pub fn semantic_identity(self) -> VmFusionSemanticIdentity {
        self.semantic
    }

    pub fn installation(self) -> u64 {
        self.installation
    }
}

#[cfg(test)]
#[path = "vm_fusion_completion/tests.rs"]
mod tests;
