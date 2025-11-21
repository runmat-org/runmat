use std::sync::atomic::{AtomicU64, Ordering};

pub struct WgpuMetrics {
    fused_cache_hits: AtomicU64,
    fused_cache_misses: AtomicU64,
    last_warmup_millis: AtomicU64,
}

impl Default for WgpuMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl WgpuMetrics {
    pub fn new() -> Self {
        Self {
            fused_cache_hits: AtomicU64::new(0),
            fused_cache_misses: AtomicU64::new(0),
            last_warmup_millis: AtomicU64::new(0),
        }
    }

    pub fn inc_hit(&self) {
        self.fused_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_miss(&self) {
        self.fused_cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn counters(&self) -> (u64, u64) {
        (
            self.fused_cache_hits.load(Ordering::Relaxed),
            self.fused_cache_misses.load(Ordering::Relaxed),
        )
    }

    pub fn reset(&self) {
        self.fused_cache_hits.store(0, Ordering::Relaxed);
        self.fused_cache_misses.store(0, Ordering::Relaxed);
        self.last_warmup_millis.store(0, Ordering::Relaxed);
    }

    pub fn set_last_warmup_millis(&self, ms: u64) {
        self.last_warmup_millis.store(ms, Ordering::Relaxed);
    }

    pub fn last_warmup_millis(&self) -> u64 {
        self.last_warmup_millis.load(Ordering::Relaxed)
    }
}
