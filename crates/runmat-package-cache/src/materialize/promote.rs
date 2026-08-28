use super::verify::transition;
use super::MaterializationState;
use crate::state::CacheState;
use crate::CacheError;
use runmat_package::ContentDigest;

pub fn promote(
    state: &mut CacheState,
    digest: &ContentDigest,
    now_ms: u64,
) -> Result<(), CacheError> {
    transition(state, digest, MaterializationState::Promoted, now_ms)
}
