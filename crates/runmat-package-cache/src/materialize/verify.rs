use super::MaterializationState;
use crate::state::{CacheState, CorruptionRecord};
use crate::CacheError;
use runmat_package::ContentDigest;

pub fn verify(
    state: &mut CacheState,
    digest: &ContentDigest,
    now_ms: u64,
) -> Result<(), CacheError> {
    transition(state, digest, MaterializationState::Verified, now_ms)
}

pub fn mark_corrupt(
    state: &mut CacheState,
    digest: &ContentDigest,
    reason: impl Into<String>,
    now_ms: u64,
) -> Result<(), CacheError> {
    let reason = reason.into();
    transition(
        state,
        digest,
        MaterializationState::Corrupt {
            reason: reason.clone(),
        },
        now_ms,
    )?;
    state
        .corruptions
        .entry(digest.clone())
        .and_modify(|record| {
            record.detected_at_ms = now_ms;
            record.reason.clone_from(&reason);
            record.occurrences = record.occurrences.saturating_add(1);
        })
        .or_insert_with(|| CorruptionRecord::new(now_ms, reason));
    Ok(())
}

pub(crate) fn transition(
    state: &mut CacheState,
    digest: &ContentDigest,
    next: MaterializationState,
    now_ms: u64,
) -> Result<(), CacheError> {
    let record = state
        .materializations
        .get_mut(digest)
        .ok_or_else(|| CacheError::Materialization(format!("{digest} is not materializing")))?;
    if !record.state.can_transition_to(&next) {
        return Err(CacheError::Materialization(format!(
            "invalid transition from {:?} to {next:?}",
            record.state
        )));
    }
    record.state = next;
    record.updated_at_ms = now_ms;
    Ok(())
}
