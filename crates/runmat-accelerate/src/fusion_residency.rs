use std::collections::HashSet;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use runmat_accelerate_api::{handle_identity, GpuHandleIdentity, GpuTensorHandle};

static RESIDENT_HANDLES: Lazy<Mutex<HashSet<GpuHandleIdentity>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));

pub fn mark(handle: &GpuTensorHandle) {
    if let Ok(mut guard) = RESIDENT_HANDLES.lock() {
        guard.insert(handle_identity(handle));
    }
}

pub fn clear(handle: &GpuTensorHandle) {
    if let Ok(mut guard) = RESIDENT_HANDLES.lock() {
        guard.remove(&handle_identity(handle));
    }
}

pub fn is_resident(handle: &GpuTensorHandle) -> bool {
    RESIDENT_HANDLES
        .lock()
        .map(|guard| guard.contains(&handle_identity(handle)))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle(device_id: u32) -> GpuTensorHandle {
        GpuTensorHandle {
            shape: vec![1, 1],
            device_id,
            buffer_id: 17,
        }
    }

    #[test]
    fn residency_is_namespaced_by_device_and_buffer() {
        let first = handle(101);
        let second = handle(202);

        mark(&first);
        assert!(is_resident(&first));
        assert!(!is_resident(&second));

        mark(&second);
        clear(&first);
        assert!(!is_resident(&first));
        assert!(is_resident(&second));
        clear(&second);
    }
}
