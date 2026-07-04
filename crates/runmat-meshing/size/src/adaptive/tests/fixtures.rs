use crate::adaptive::{RefinementIndicatorAvailability, RefinementIndicatorKey};

pub(super) fn key(namespace: &str, name: &str) -> RefinementIndicatorKey {
    RefinementIndicatorKey::new(namespace, name)
}

pub(super) fn available(namespace: &str, name: &str) -> RefinementIndicatorAvailability {
    RefinementIndicatorAvailability {
        key: key(namespace, name),
        applicable: true,
        field_available: true,
    }
}
