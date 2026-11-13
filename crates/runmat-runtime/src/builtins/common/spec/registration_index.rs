//! Borrowed index over the spec owner's existing physical registrations.
use super::FusionSpecRegistration;

#[cfg(not(target_arch = "wasm32"))]
static INDEX: std::sync::LazyLock<Vec<FusionSpecRegistration>> = std::sync::LazyLock::new(|| {
    let mut entries: Vec<_> = super::builtin_fusion_spec_registrations().collect();
    entries.sort_by_key(|entry| entry.builtin_path);
    entries
});

pub(super) fn with_registrations<R>(
    owner: &str,
    visit: impl FnOnce(&[FusionSpecRegistration]) -> R,
) -> R {
    #[cfg(not(target_arch = "wasm32"))]
    {
        visit(matching(&INDEX, owner))
    }
    #[cfg(target_arch = "wasm32")]
    {
        super::wasm_registry::with_fusion_index(|entries| visit(matching(entries, owner)))
    }
}

fn matching<'a>(
    entries: &'a [FusionSpecRegistration],
    owner: &str,
) -> &'a [FusionSpecRegistration] {
    let first = entries.partition_point(|entry| entry.builtin_path < owner);
    let count = entries[first..].partition_point(|entry| entry.builtin_path == owner);
    &entries[first..first + count]
}

#[cfg(any(test, target_arch = "wasm32"))]
pub(super) fn insert(
    entries: &mut Vec<FusionSpecRegistration>,
    registration: FusionSpecRegistration,
) {
    let at = entries.partition_point(|entry| entry.builtin_path <= registration.builtin_path);
    entries.insert(at, registration);
}

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    pub(super) static INVENTORY_TRAVERSALS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn warm_spec_lookup_borrows_without_inventory_traversal() {
        let sample = super::super::builtin_fusion_spec_registrations()
            .next()
            .unwrap();
        let first = with_registrations(sample.builtin_path, |entries| entries.as_ptr());
        let before = INVENTORY_TRAVERSALS.with(std::cell::Cell::get);
        for _ in 0..64 {
            with_registrations(sample.builtin_path, |entries| {
                assert!(!entries.is_empty());
                assert_eq!(entries.as_ptr(), first);
            });
        }
        assert_eq!(INVENTORY_TRAVERSALS.with(std::cell::Cell::get), before);
    }

    #[test]
    fn late_spec_registration_retains_duplicates_after_early_miss() {
        let sample = super::super::builtin_fusion_spec_registrations()
            .next()
            .unwrap();
        let mut index = Vec::new();
        assert!(matching(&index, sample.builtin_path).is_empty());
        insert(&mut index, sample);
        assert_eq!(matching(&index, sample.builtin_path).len(), 1);
        insert(&mut index, sample);
        assert_eq!(matching(&index, sample.builtin_path).len(), 2);
        assert!(matching(&index, "missing-physical-owner").is_empty());
    }
}
