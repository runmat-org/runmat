use super::{Lease, LeaseId, LeaseOwner};
use crate::state::CacheState;
use crate::CacheError;

pub fn renew(
    state: &mut CacheState,
    id: &LeaseId,
    owner: &LeaseOwner,
    generation: u64,
    now_ms: u64,
    ttl_ms: u64,
) -> Result<Lease, CacheError> {
    if ttl_ms == 0 {
        return Err(CacheError::Lease("lease TTL must be nonzero".to_string()));
    }
    let lease = state
        .leases
        .get_mut(id)
        .ok_or_else(|| CacheError::Lease(format!("lease {id} does not exist")))?;
    if &lease.owner != owner || lease.generation != generation || !lease.is_active_at(now_ms) {
        return Err(CacheError::Lease(format!(
            "lease {id} ownership, generation, or expiry check failed"
        )));
    }
    lease.expires_at_ms = now_ms
        .checked_add(ttl_ms)
        .ok_or_else(|| CacheError::Lease("lease expiration overflow".to_string()))?;
    Ok(lease.clone())
}
