use super::{MaterializationRecord, MaterializationState};
use crate::lease::LeaseId;
use crate::state::CacheState;
use crate::CacheError;
use runmat_package::ContentDigest;

pub fn begin(
    state: &mut CacheState,
    digest: &ContentDigest,
    lease: LeaseId,
    attempt: impl Into<String>,
    now_ms: u64,
) -> Result<MaterializationRecord, CacheError> {
    if !state.objects.contains_key(digest) || !state.leases.contains_key(&lease) {
        return Err(CacheError::Materialization(
            "materialization requires an object and active lease".to_string(),
        ));
    }
    let next = MaterializationRecord {
        state: MaterializationState::Staging {
            attempt: attempt.into(),
        },
        lease,
        updated_at_ms: now_ms,
    };
    if let Some(current) = state.materializations.get(digest) {
        if !current.state.can_transition_to(&next.state) {
            return Err(CacheError::Materialization(format!(
                "cannot restart materialization from {:?}",
                current.state
            )));
        }
    }
    state.materializations.insert(digest.clone(), next.clone());
    Ok(next)
}
