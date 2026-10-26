//! Original allocation authority, separate from source/binding provenance.
use crate::context::host_allocation::{
    HostAllocationCheckpoint, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
    HostTypedWorkspace,
};
use crate::context::{RuntimeContext, RuntimeContextLocalId};
use runmat_execution::cancellation::CancellationToken;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Weak,
};
#[cfg(test)]
mod tests;

/// A deferred quote capability, not a capacity grant. It retains neither a
/// RuntimeContext nor an execution service. Ordinary no-scratch schemas never
/// consult it, so missing/denied optional authority cannot change those calls.
#[derive(Clone)]
pub struct ValidationWorkspaceOwner {
    context: Option<RuntimeContextLocalId>,
    cancellation: Option<CancellationToken>,
    generation: Option<(Arc<AtomicU64>, u64)>,
    session: Option<Weak<()>>,
    authority: Result<Option<HostAllocationCheckpoint>, HostAllocationError>,
}
impl ValidationWorkspaceOwner {
    pub fn capture(runtime: &RuntimeContext) -> Self {
        Self {
            context: Some(runtime.local_identity()),
            cancellation: Some(runtime.cancellation()),
            generation: Some(runtime.host_effect_session_generation()),
            session: Some(runtime.session_lease()),
            authority: HostAllocationReceipt::capture_owner(Some(runtime)),
        }
    }
    /// Only existing context-free synchronous codec entry points use the
    /// ordinary policy. Issuer-only/delayed calls cannot manufacture it.
    pub(super) fn ordinary() -> Self {
        Self {
            context: None,
            cancellation: None,
            generation: None,
            session: None,
            authority: Ok(None),
        }
    }
    pub(crate) fn for_runtime(runtime: Option<&RuntimeContext>) -> Self {
        runtime.map_or_else(Self::ordinary, Self::capture)
    }
    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        if self.context != other.context {
            return false;
        }
        let token = match (&self.cancellation, &other.cancellation) {
            (Some(a), Some(b)) => a.same_generation(b),
            (None, None) => true,
            _ => false,
        };
        let generation = match (&self.generation, &other.generation) {
            (Some((a, x)), Some((b, y))) => x == y && Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        let session = match (&self.session, &other.session) {
            (Some(a), Some(b)) => Weak::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        let authority = match (&self.authority, &other.authority) {
            (Ok(Some(a)), Ok(Some(b))) => a.same_owner(b),
            (Ok(None), Ok(None)) => true,
            (Err(a), Err(b)) => a == b,
            _ => false,
        };
        token && generation && session && authority
    }
    fn check(&self) -> Result<(), HostAllocationError> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(HostAllocationError::Cancelled);
        }
        if self
            .session
            .as_ref()
            .is_some_and(|session| session.upgrade().is_none())
            || self
                .generation
                .as_ref()
                .is_some_and(|(owner, generation)| owner.load(Ordering::Acquire) != *generation)
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        match &self.authority {
            Ok(Some(checkpoint)) => checkpoint.check_original(),
            Ok(None) => Ok(()),
            Err(error) => Err(*error),
        }
    }
    pub(crate) fn with_f64_scratch<T>(
        &self,
        len: usize,
        operation: impl FnOnce(&mut [f64]) -> T,
    ) -> Result<T, HostAllocationError> {
        self.check()?;
        let result = if let Some(checkpoint) = self.authority.as_ref().map_err(|e| *e)? {
            let plan = HostAllocationPlan::new(HostTypedWorkspace::<f64>::quoted_bytes(len)?, 0)?;
            let receipt = HostAllocationReceipt::admit_captured(checkpoint.clone(), plan)?;
            let mut scratch = receipt.temporary_typed_captured::<f64>(len)?;
            let result = operation(scratch.as_mut_slice());
            receipt.checkpoint_captured()?;
            result
        } else {
            let mut scratch =
                runmat_value::exact_capacity_vec(len).map_err(HostAllocationError::Numeric)?;
            scratch.resize(len, 0.0);
            operation(&mut scratch)
        };
        self.check()?;
        Ok(result)
    }
}
