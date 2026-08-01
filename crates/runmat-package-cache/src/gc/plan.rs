use super::mark::protected_closure;
use super::GcPolicy;
use crate::state::CacheState;
use runmat_package::ContentDigest;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GcPlan {
    pub delete: BTreeSet<ContentDigest>,
    pub reclaim_bytes: u64,
}

impl GcPlan {
    pub fn build(state: &CacheState, policy: GcPolicy) -> Self {
        let protected = protected_closure(state, policy.now_ms);
        let recent_cutoff = policy.now_ms.saturating_sub(policy.retain_recent_ms);
        let mut candidates: Vec<_> = state
            .objects
            .iter()
            .filter(|(digest, _)| !protected.contains(*digest))
            .filter(|(digest, _)| {
                state
                    .access
                    .get(*digest)
                    .is_none_or(|access| access.last_accessed_at_ms <= recent_cutoff)
            })
            .map(|(digest, object)| {
                (
                    state
                        .access
                        .get(digest)
                        .map_or(0, |access| access.last_accessed_at_ms),
                    digest.clone(),
                    object.stored_payload_bytes(),
                )
            })
            .collect();
        candidates.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

        let mut plan = Self::default();
        for (_, digest, byte_len) in candidates {
            if plan.reclaim_bytes >= policy.target_bytes {
                break;
            }
            if plan.delete.insert(digest.clone()) {
                plan.reclaim_bytes = plan.reclaim_bytes.saturating_add(byte_len);
            }
            add_reverse_dependents(state, &protected, &mut plan);
        }
        plan
    }
}

fn add_reverse_dependents(
    state: &CacheState,
    protected: &BTreeSet<ContentDigest>,
    plan: &mut GcPlan,
) {
    loop {
        let dependent = state.objects.iter().find_map(|(digest, object)| {
            (!protected.contains(digest)
                && !plan.delete.contains(digest)
                && object
                    .references()
                    .iter()
                    .any(|reference| plan.delete.contains(reference)))
            .then(|| (digest.clone(), object.stored_payload_bytes()))
        });
        match dependent {
            Some((digest, byte_len)) => {
                plan.delete.insert(digest);
                plan.reclaim_bytes = plan.reclaim_bytes.saturating_add(byte_len);
            }
            None => break,
        }
    }
}
