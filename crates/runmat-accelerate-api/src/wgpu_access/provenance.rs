//! Device-local, monotone facts about effects outside the measured encoder.
//! These are not a command receipt or an alternate provider registry.

#[cfg(target_arch = "wasm32")]
use std::cell::Cell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
pub(super) struct DeviceEffectProvenance {
    #[cfg(target_arch = "wasm32")]
    raw_export: Cell<bool>,
    #[cfg(not(target_arch = "wasm32"))]
    raw_export: AtomicBool,
    #[cfg(target_arch = "wasm32")]
    implicit_transfer: Cell<bool>,
    #[cfg(not(target_arch = "wasm32"))]
    implicit_transfer: AtomicBool,
}

impl DeviceEffectProvenance {
    pub(super) fn mark_raw_export(&self) {
        #[cfg(target_arch = "wasm32")]
        self.raw_export.set(true);
        #[cfg(not(target_arch = "wasm32"))]
        self.raw_export.store(true, Ordering::SeqCst);
    }

    pub(super) fn has_raw_export(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            self.raw_export.get()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.raw_export.load(Ordering::SeqCst)
        }
    }

    pub(super) fn mark_implicit_transfer(&self) {
        #[cfg(target_arch = "wasm32")]
        self.implicit_transfer.set(true);
        #[cfg(not(target_arch = "wasm32"))]
        self.implicit_transfer.store(true, Ordering::SeqCst);
    }

    pub(super) fn has_implicit_transfer(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            self.implicit_transfer.get()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.implicit_transfer.load(Ordering::SeqCst)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceEffectProvenance;

    #[test]
    fn earlier_export_remains_visible_to_later_admission_and_seal() {
        let fact = DeviceEffectProvenance::default();
        assert!(!fact.has_raw_export());
        fact.mark_raw_export();
        assert!(fact.has_raw_export());
        fact.mark_raw_export();
        assert!(fact.has_raw_export());
        assert!(!fact.has_implicit_transfer());
        fact.mark_implicit_transfer();
        assert!(fact.has_implicit_transfer());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn concurrent_export_between_admission_and_seal_is_monotone() {
        use std::sync::{Arc, Barrier};
        let fact = Arc::new(DeviceEffectProvenance::default());
        let arrived = Arc::new(Barrier::new(2));
        let exported = Arc::new(Barrier::new(2));
        let worker = {
            let fact = fact.clone();
            let arrived = arrived.clone();
            let exported = exported.clone();
            std::thread::spawn(move || {
                arrived.wait();
                fact.mark_raw_export();
                fact.mark_implicit_transfer();
                exported.wait();
            })
        };
        assert!(!fact.has_raw_export());
        assert!(!fact.has_implicit_transfer());
        arrived.wait();
        exported.wait();
        assert!(fact.has_raw_export());
        assert!(fact.has_implicit_transfer());
        worker.join().unwrap();
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn browser_reentry_between_admission_and_seal_is_visible() {
        use std::rc::Rc;
        let fact = Rc::new(DeviceEffectProvenance::default());
        assert!(!fact.has_raw_export());
        assert!(!fact.has_implicit_transfer());
        let reentered = fact.clone();
        let callback = move || {
            reentered.mark_raw_export();
            reentered.mark_implicit_transfer();
        };
        callback();
        assert!(fact.has_raw_export());
        assert!(fact.has_implicit_transfer());
    }
}
