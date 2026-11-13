use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HostCopyReason {
    CopyOnWriteMutation = 0,
    OwnedMaterialization = 1,
    ComplexLayoutConversion = 2,
    CharacterEncoding = 3,
    SparseLayoutConversion = 4,
    ForeignStorageConversion = 5,
    ProviderReadback = 6,
    ProcessSnapshot = 7,
    MemoryLayoutConversion = 8,
    ProviderUpload = 9,
}

const REASON_COUNT: usize = 10;
static COPY_COUNTS: [AtomicU64; REASON_COUNT] = [const { AtomicU64::new(0) }; REASON_COUNT];
static COPY_BYTES: [AtomicU64; REASON_COUNT] = [const { AtomicU64::new(0) }; REASON_COUNT];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCopyMetrics {
    pub operations: u64,
    pub bytes: u64,
}

pub fn record_host_copy(reason: HostCopyReason, bytes: usize) {
    let index = reason as usize;
    COPY_COUNTS[index].fetch_add(1, Ordering::Relaxed);
    COPY_BYTES[index].fetch_add(bytes as u64, Ordering::Relaxed);
}

pub fn host_copy_metrics(reason: HostCopyReason) -> HostCopyMetrics {
    let index = reason as usize;
    HostCopyMetrics {
        operations: COPY_COUNTS[index].load(Ordering::Relaxed),
        bytes: COPY_BYTES[index].load(Ordering::Relaxed),
    }
}
