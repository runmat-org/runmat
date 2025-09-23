use std::collections::HashSet;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use runmat_accelerate_api::GpuTensorHandle;

static RESIDENT_HANDLES: Lazy<Mutex<HashSet<u64>>> = Lazy::new(|| Mutex::new(HashSet::new()));

pub fn mark(handle: &GpuTensorHandle) {
    if let Ok(mut guard) = RESIDENT_HANDLES.lock() {
        guard.insert(handle.buffer_id);
    }
}

pub fn clear(handle: &GpuTensorHandle) {
    if let Ok(mut guard) = RESIDENT_HANDLES.lock() {
        guard.remove(&handle.buffer_id);
    }
}

pub fn is_resident(handle: &GpuTensorHandle) -> bool {
    RESIDENT_HANDLES
        .lock()
        .map(|guard| guard.contains(&handle.buffer_id))
        .unwrap_or(false)
}
