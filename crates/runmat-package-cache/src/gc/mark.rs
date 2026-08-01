use crate::state::CacheState;
use runmat_package::ContentDigest;
use std::collections::BTreeSet;

pub(crate) fn protected_closure(state: &CacheState, now_ms: u64) -> BTreeSet<ContentDigest> {
    let mut marked = BTreeSet::new();
    let mut pending: Vec<_> = state
        .leases
        .values()
        .filter(|lease| lease.is_active_at(now_ms))
        .flat_map(|lease| lease.objects.iter().cloned())
        .chain(
            state
                .pins
                .values()
                .flat_map(|pin| pin.objects.iter().cloned()),
        )
        .chain(state.materializations.keys().cloned())
        .collect();
    while let Some(digest) = pending.pop() {
        if !marked.insert(digest.clone()) {
            continue;
        }
        if let Some(object) = state.objects.get(&digest) {
            pending.extend(object.references());
        }
    }
    marked
}
