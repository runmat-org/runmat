use super::{Lease, LeaseId, LeaseOwner};
use crate::state::CacheState;
use crate::CacheError;
use runmat_package::ContentDigest;
use std::collections::BTreeSet;

pub fn acquire(
    state: &mut CacheState,
    id: LeaseId,
    owner: LeaseOwner,
    objects: BTreeSet<ContentDigest>,
    now_ms: u64,
    ttl_ms: u64,
) -> Result<Lease, CacheError> {
    if ttl_ms == 0 {
        return Err(CacheError::Lease("lease TTL must be nonzero".to_string()));
    }
    if let Some(missing) = objects
        .iter()
        .find(|digest| !state.objects.contains_key(*digest))
    {
        return Err(CacheError::Miss(missing.clone()));
    }
    if state
        .leases
        .get(&id)
        .is_some_and(|lease| lease.is_active_at(now_ms))
    {
        return Err(CacheError::Lease(format!(
            "active lease {id} already exists"
        )));
    }
    let expires_at_ms = now_ms
        .checked_add(ttl_ms)
        .ok_or_else(|| CacheError::Lease("lease expiration overflow".to_string()))?;
    let generation = state
        .leases
        .get(&id)
        .map_or(0, |lease| lease.generation.saturating_add(1));
    let lease = Lease {
        id: id.clone(),
        owner,
        objects,
        acquired_at_ms: now_ms,
        expires_at_ms,
        generation,
    };
    state.leases.insert(id, lease.clone());
    Ok(lease)
}
