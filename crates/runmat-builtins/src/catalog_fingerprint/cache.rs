//! One worker's immutable fingerprint between explicit registry mutations.
use std::cell::Cell;

pub(super) struct RegistryFingerprintCache(Cell<Option<[u8; 32]>>);

impl RegistryFingerprintCache {
    pub(super) const fn new() -> Self {
        Self(Cell::new(None))
    }

    pub(super) fn get_or_init(&self, compute: impl FnOnce() -> [u8; 32]) -> [u8; 32] {
        if let Some(value) = self.0.get() {
            return value;
        }
        let value = compute();
        self.0.set(Some(value));
        value
    }

    pub(super) fn invalidate(&self) {
        self.0.set(None);
    }
}
