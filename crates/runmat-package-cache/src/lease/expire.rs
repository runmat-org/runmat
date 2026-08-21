use super::LeaseId;
use crate::state::CacheState;

pub fn expire(state: &mut CacheState, now_ms: u64) -> Vec<LeaseId> {
    let expired: Vec<_> = state
        .leases
        .iter()
        .filter(|(_, lease)| !lease.is_active_at(now_ms))
        .map(|(id, _)| id.clone())
        .collect();
    for id in &expired {
        state.leases.remove(id);
        state
            .materializations
            .retain(|_, materialization| &materialization.lease != id);
    }
    expired
}
