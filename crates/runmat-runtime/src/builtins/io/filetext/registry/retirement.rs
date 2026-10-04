use super::*;

pub(super) fn retire_session(identity: RuntimeContextLocalId) {
    let removed = {
        let mut registry = REGISTRY.lock().unwrap_or_else(|poison| poison.into_inner());
        registry.cleanup_registered.remove(&identity);
        let ids: Vec<i32> = registry
            .entries
            .iter_mut()
            .filter_map(|(id, entry)| {
                if entry.issuer.session_identity() == Some(identity) {
                    entry.retired = true;
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        ids.into_iter()
            .filter_map(|id| {
                let idle = registry.entries.get(&id).is_some_and(|entry| {
                    entry
                        .file_handle()
                        .is_some_and(|handle| handle.try_lock().is_ok_and(|slot| slot.is_some()))
                });
                idle.then(|| registry.entries.remove(&id)).flatten()
            })
            .collect::<Vec<_>>()
    };
    // Provider-owned handles are dropped after the registry lock is gone.
    drop(removed);
}
