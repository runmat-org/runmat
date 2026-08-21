use super::{LeaseId, LeaseOwner};
use crate::state::CacheState;
use crate::CacheError;

pub fn release(
    state: &mut CacheState,
    id: &LeaseId,
    owner: &LeaseOwner,
    generation: u64,
) -> Result<(), CacheError> {
    let lease = state
        .leases
        .get(id)
        .ok_or_else(|| CacheError::Lease(format!("lease {id} does not exist")))?;
    if &lease.owner != owner || lease.generation != generation {
        return Err(CacheError::Lease(format!(
            "lease {id} ownership or generation check failed"
        )));
    }
    state.leases.remove(id);
    state
        .materializations
        .retain(|_, materialization| &materialization.lease != id);
    Ok(())
}
